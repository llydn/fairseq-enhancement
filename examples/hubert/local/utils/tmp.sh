set -o pipefail
set -e

# mode and data
model=/mnt/lustre/sjtu/home/ww089/fairseqs/fairseq-220604/examples/hubert/download_models/hubert_base_ls960.pt
src=data/ls_960

# hyper parameters
km=500
layer=9
kernels="3_5_5_5"

# jobs
cmd=run.pl
nj=4
denoise_nj=10
device=0
stage=1
stop_stage=1

dumpdir=

. ./path.sh
. utils/parse_options.sh

if [ -z $dumpdir ]; then
    dumpdir=/mnt/lustre02/scratch/sjtu/home/ww089/fairseqs/fairseq-low-resource/examples/hubert/dump/data_`basename $src`_model_`basename $model .pt`/
fi

feat_dir=$dumpdir/feat_L$layer 

max_job_num=$((nj-1))

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
#     echo "dumping train feature for layer $layer"
#     local/utils/dump_hubert_feature.sh --src $src --split train --model $model --layer $layer --nj $nj --dst ${feat_dir} --device $device

    echo "dumping valid feature for layer $layer"
    local/utils/dump_hubert_feature.sh --src $src --split valid --model $model --layer $layer --nj $nj --dst $dumpdir/tmp --device $device
fi

