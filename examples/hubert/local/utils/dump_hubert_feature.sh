#!/usr/bin/env bash
## local/utils/dump_hubert_feature.sh --src data/ls_960/ --split valid --model simple_kmeans/hubert_base_ls960.pt --layer 9 --nj 4 --dst dump/raw/ls_960/feat_from_hubert_base_ls960

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

src=data/ls_960
dst=dump/feat_L9
split=train
cmd=run.pl
model=/mnt/lustre/sjtu/home/ww089/fairseqs/fairseq-220604/examples/hubert/download_models/hubert_base_ls960.pt
layer=9
nj=4
device=0


. ./path.sh
. utils/parse_options.sh
max_job_num=$((nj-1))

SECONDS=0
[ -L $dst/model.pt ] && rm $dst/model.pt

ln -s `realpath $model` $dst/model.pt

if [ -z $device ]; then
    $cmd JOB=0:${max_job_num} $dst/logdir/dump_hubert_feature.JOB.log \
        OPENBLAS_NUM_THREADS=5 python simple_kmeans/dump_hubert_feature.py $src $split $model $layer $nj JOB $dst
else
    $cmd JOB=0:${max_job_num} $dst/logdir/dump_hubert_feature.JOB.log \
        OPENBLAS_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=$device python simple_kmeans/dump_hubert_feature.py $src $split $model $layer $nj JOB $dst
fi

echo "dump feature done, ${SECONDS}s Elapsed. "

