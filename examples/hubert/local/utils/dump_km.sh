#!/usr/bin/env bash
set -o pipefail
set -e

# mode and data
#model=/mnt/lustre/sjtu/home/ww089/fairseqs/fairseq-220604/examples/hubert/download_models/hubert_base_ls960.pt
model=models/hubert_base_ls960.pt
src=data/ll_10

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
    dumpdir=dump/data_`basename $src`_model_`basename $model .pt`
fi

feat_dir=$dumpdir/feat_L$layer 
km_dir=$dumpdir/km_L${layer}_${km}
denoise_km_dir=${km_dir}/kernel_${kernels}/
mkdir -p ${feat_dir} ${km_dir}

max_job_num=$((nj-1))

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "dumping train feature for layer $layer"
    local/utils/dump_hubert_feature.sh --src $src --split train --model $model --layer $layer --nj $nj --dst ${feat_dir} --device $device

    echo "dumping valid feature for layer $layer"
    local/utils/dump_hubert_feature.sh --src $src --split valid --model $model --layer $layer --nj $nj --dst ${feat_dir} --device $device
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "learning kmeans for layer $layer km $km"
    OPENBLAS_NUM_THREADS=20 python simple_kmeans/learn_kmeans.py $dumpdir/feat_L$layer train $nj $dumpdir/L${layer}_${km}_p100.kmodel $km --percent 0.1

fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "dump train km label for layer $layer km $km"
    local/utils/dump_km_label.sh --feat_dir ${feat_dir} --split train --km_path $dumpdir/L${layer}_${km}_p100.kmodel --nj 4 --dst ${km_dir} --device $device

    for i in `seq 0 $max_job_num`; do cat ${km_dir}/train_${i}_${nj}.km; done > ${km_dir}/train.km

    echo "dump valid km label for layer $layer km $km"
    local/utils/dump_km_label.sh --feat_dir ${feat_dir} --split valid --km_path $dumpdir/L${layer}_${km}_p100.kmodel --nj 4 --dst ${km_dir} --device $device

    for i in `seq 0 $max_job_num`; do cat ${km_dir}/valid_${i}_${nj}.km; done > $km_dir/valid.km
fi

# rm $dumpdir/feat_L$layer/*npy


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "measure quality for raw layer $layer km $km labels"
    python local/utils/measure_teacher_quality.py data/ls_960/ ${km_dir} km --phn_dir /mnt/lustre/sjtu/home/ww089/fairseqs/fairseq-220604/examples/hubert/dump/raw/ls_960/phone_frame_align/ --phn_sets dev_clean dev_other --upsample 2 --verbose
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    mkdir -p ${denoise_km_dir}/logs
    for split in train valid; do
        rm ${denoise_km_dir}/${split}*.km &> /dev/null || true
        split -l$((`wc -l < ${km_dir}/${split}.km`/$(($denoise_nj)) + 1)) ${km_dir}/${split}.km ${denoise_km_dir}/${split}. --additional-suffix=".km" -da 3 --numeric-suffixes=1
        echo "logging to '${denoise_km_dir}/logs/${split}_denoise.n.log'"

        denoise_nj_actual=`ls -l ${denoise_km_dir}/${split}.*.km | wc -l`
        pushd ${denoise_km_dir} &> /dev/null
        for f in ${split}.*.km; do
            mv "$f" "`echo $f | sed -E 's/0+([1-9])/\1/'`" &> /dev/null || true
        done
        popd &> /dev/null

        ${cmd} JOB=1:"${denoise_nj_actual}" "${denoise_km_dir}"/logs/${split}_denoise.JOB.log \
            python local/utils/splice_data_generation/denoise_km.py \
                --input ${denoise_km_dir}/${split}.JOB.km \
                --output ${denoise_km_dir}/${split}_denoise.JOB.km \
                --kernels $kernels

        for i in `seq 1 ${denoise_nj_actual}`; do
            cat ${denoise_km_dir}/${split}_denoise.$i.km
        done >> ${denoise_km_dir}/${split}_denoise.km

        rm ${denoise_km_dir}/${split}.*.km ${denoise_km_dir}/${split}_denoise.*.km
    done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "measure quality for denoised layer $layer km $km labels (kernel = $kernels)"
    python local/utils/measure_teacher_quality.py data/ls_960/ ${denoise_km_dir} km --phn_dir /mnt/lustre/sjtu/home/ww089/fairseqs/fairseq-220604/examples/hubert/dump/raw/ls_960/phone_frame_align/ --lab_sets valid_denoise --phn_sets dev_clean dev_other --upsample 2 --verbose
fi

# if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
#     echo "dump km score for layer $layer km $km labels (kernel = $kernels)"
#     if [ -z $device ]; then
#         $cmd JOB=0:${max_job_num} $dst/logdir/dump_km_score.JOB.log \
#             OPENBLAS_NUM_THREADS=2 python simple_kmeans/dump_km_label.py ${feat_dir} ${split} ${km_path} $nj JOB $dst
#     else
#         $cmd JOB=0:${max_job_num} $dst/logdir/dump_hubert_feature.JOB.log \
#             OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$device python simple_kmeans/dump_km_label.py ${feat_dir} ${split} ${km_path} $nj JOB $dst
#     fi
# fi
