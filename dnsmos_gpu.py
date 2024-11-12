"""
# @Author : echonoshy
# @File : dnsmos_gpu.py
# Usage:
# python dnsmos_local.py -mbo bak_ovr.onnx -t test_dir/ -o ovr_high_score_paths.txt

"""


import concurrent.futures
import glob
import os
import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort
import numpy.polynomial.polynomial as poly
from tqdm import tqdm

COEFS_OVR = np.array([8.924546794696789354e-01, 6.609981731940616223e-01, 7.600269530243179694e-02])

def audio_logpowspec(audio, nfft=320, hop_length=160, sr=16000):
    powspec = (np.abs(librosa.core.stft(audio, n_fft=nfft, hop_length=hop_length))) ** 2
    logpowspec = np.log10(np.maximum(powspec, 10 ** (-12)))
    return logpowspec.T

def process_file(fpath, session_bak_ovr, fs=16000, input_length=9):
    audio, file_fs = sf.read(fpath)
    if file_fs != fs:
        audio = librosa.resample(audio, orig_sr=file_fs, target_sr=fs)
    if len(audio) < 2 * fs:
        return None  # 音频片段太短

    len_samples = int(input_length * fs)
    while len(audio) < len_samples:
        audio = np.append(audio, audio)
    
    num_hops = int(np.floor(len(audio) / fs) - input_length) + 1
    hop_len_samples = fs
    predicted_mos_ovr_seg = []

    for idx in range(num_hops):
        audio_seg = audio[int(idx * hop_len_samples): int((idx + input_length) * hop_len_samples)]
        input_features = np.array(audio_logpowspec(audio=audio_seg, sr=fs)).astype('float32')[np.newaxis, :, :]
        onnx_inputs_bak_ovr = {inp.name: input_features for inp in session_bak_ovr.get_inputs()}
        mos_bak_ovr = session_bak_ovr.run(None, onnx_inputs_bak_ovr)
        mos_ovr = poly.polyval(mos_bak_ovr[0][0][2], COEFS_OVR)
        predicted_mos_ovr_seg.append(mos_ovr)

    mean_mos_ovr = np.mean(predicted_mos_ovr_seg)
    return (fpath, mean_mos_ovr) if mean_mos_ovr >= 4.0 else None

def main(args):
    providers = [('CUDAExecutionProvider', {'device_id': 0})]
    session_bak_ovr = ort.InferenceSession(args.bak_ovr_model_path, providers=providers)
    audio_clips_list = glob.glob(os.path.join(args.testset_dir, "*.wav"))
    output_path = args.output_path if args.output_path else "ovr_high_score_paths.txt"

    with open(output_path, mode='w') as output_file, concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, fpath, session_bak_ovr) for fpath in audio_clips_list]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                output_file.write(f"{result[0]}\n")

    print(f"Paths of audio files with OVR >= 4.0 have been saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-mbo', "--bak_ovr_model_path", default='bak_ovr.onnx', help='Path to ONNX model for BAK and OVR prediction')
    parser.add_argument('-t', "--testset_dir", default='.', help='Path to the dir containing audio clips in .wav to be evaluated')
    parser.add_argument('-o', "--output_path", default=None, help='File to save paths with OVR >= 4.0')
    parser.add_argument('-l', "--input_length", type=int, default=9)

    args = parser.parse_args()
    main(args)
