#!/usr/bin/awk -f
## awk -v frame_shift=0.02 -f local/utils/splice_data_generation/ali2ctm.awk /mnt/lustre02/scratch/sjtu/home/ww089/fairseqs/fairseq-low-resource/examples/wavlm/dump/data_ls_960_model_wavlm_base_plus/km_L6_50/kernel_3_5_5_5/splice_data_generation/ctm

{
    start=2;
    uttid=$1;
    $1=$2;
    token=$1;
    for (i=2; i<=NF; ++i) {
        if ($i != $(i-1)) {
            printf("%s %.3f %.3f %s\n", uttid, (start - 2) * frame_shift, (i - 2) * frame_shift, token);
            token=$i;
            start = i;
        } 
    }
    printf("%s %.3f %.3f %s\n", uttid, (start - 2) * frame_shift, (NF - 1) * frame_shift, token);
}
