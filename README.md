# README

Code to calculate DNS_MOS score by calling the dns_mos interface provided by Microsoft.

Since the speed of CPU inferring MOS scores is too slow, GPU is used here for acceleration. 
And only recordings with OVR scores greater than 4.0 are used to screen high-quality recordings. 
Write the paths of the recordings that meet the requirements into a file.


## How to use:

1. cpu:
```
python dnsmos_local.py \
        --testset_dir $data_dir \
        --csv_path $output_dir \
        --run_name luoxiang_denoise \
        --sig_model_path $sig_model_path \
        --bak_ovr_model_path $bak_ovr_model_path

```

2. gpu:
```
# python dnsmos_local.py \ 
    -mbo bak_ovr.onnx \ 
    -t test_dir/ \ 
    -o ovr_high_score_paths.txt

```