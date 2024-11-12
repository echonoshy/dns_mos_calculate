
from pydub import AudioSegment
import webrtcvad
import collections
import contextlib
import wave
import os
import numpy as np
from multiprocessing import Pool, cpu_count
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_wave(path):
    """Read a WAV file."""
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate, sample_width, num_channels

def write_wave(path, audio, sample_rate):
    """Write audio data to a WAV file."""
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(audio)

class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generate audio frames."""
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def get_segment_duration(segment, sample_rate):
    """Calculate segment duration (in seconds)."""
    total_bytes = sum(len(frame.bytes) for frame in segment)
    return total_bytes / (2 * sample_rate)  # 2 is for 16-bit audio where each sample takes 2 bytes

def process_audio(input_file):
    """Process a single audio file."""
    try:
        # Create output directory
        base_name = Path(input_file).stem
        output_dir = Path("output") / base_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Read audio using pydub and process it
        audio = AudioSegment.from_wav(input_file)
        
        # Ensure sample rate is 16 kHz
        audio = audio.set_frame_rate(16000)
        
        # Extract the right channel
        if audio.channels == 2:
            audio = audio.split_to_mono()[1]  # Get the right channel
        
        # Export as a temporary file
        temp_file = output_dir / "temp_processed.wav"
        audio.export(str(temp_file), format="wav")
        
        # 2. Perform VAD processing
        vad = webrtcvad.Vad(3)  # Set VAD aggressiveness (0-3)
        
        # Read the processed audio
        audio, sample_rate, sample_width, _ = read_wave(str(temp_file))
        
        # Generate 30ms frames
        frames = frame_generator(30, audio, sample_rate)
        frames = list(frames)
        
        # VAD detection
        segments = []
        current_segment = []
        triggered = False
        
        for frame in frames:
            is_speech = vad.is_speech(frame.bytes, sample_rate)
            
            if not triggered and is_speech:
                triggered = True
                current_segment = [frame]
            elif triggered and is_speech:
                current_segment.append(frame)
            elif triggered and not is_speech:
                if len(current_segment) > 0:
                    # Check segment duration
                    if get_segment_duration(current_segment, sample_rate) >= 1.0:  # Keep segments of 1 second or longer
                        segments.append(current_segment)
                triggered = False
                current_segment = []
        
        # If there's any remaining segment
        if current_segment and get_segment_duration(current_segment, sample_rate) >= 1.0:
            segments.append(current_segment)
        
        # 3. Save segmented audio parts
        for i, segment in enumerate(segments, 1):
            # Combine frame data
            segment_audio = b''.join([f.bytes for f in segment])
            
            # Generate output file name
            output_file = output_dir / f"{i:03d}.wav"
            
            # Save audio segment
            write_wave(str(output_file), segment_audio, sample_rate)
        
        # Delete temporary file
        temp_file.unlink()
        
        logging.info(f"Processing completed: {input_file} -> {len(segments)} segments")
        return len(segments)
    
    except Exception as e:
        logging.error(f"Error processing file {input_file}: {str(e)}")
        return 0

def process_audio_files(input_files, num_processes=None):
    """Process multiple audio files using multiprocessing."""
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)  # Leave one CPU core free
    
    logging.info(f"Processing with {num_processes} processes")
    
    # Create a process pool
    with Pool(num_processes) as pool:
        results = pool.map(process_audio, input_files)
    
    total_segments = sum(results)
    logging.info(f"All files processed, generated a total of {total_segments} segments")
    return total_segments


if __name__ == "__main__":
    input_directory = "input_files"
    input_files = [str(f) for f in Path(input_directory).glob("*.wav")]
    
    if not input_files:
        logging.warning(f"No WAV files found in the {input_directory} directory")
    else:
        process_audio_files(input_files)
