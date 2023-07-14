set -o pipefail
set -e

# mode and data
model=/mnt/lustre/sjtu/home/ww089/fairseqs/fairseq-220604/examples/hubert/download_models/hubert_base_ls960.pt
src=data/common_voice/german/tr_1000h
lang=german

# hyper parameters
km_path=/mnt/lustre02/scratch/sjtu/home/ww089/fairseqs/fairseq-low-resource/examples/hubert/dump/data_ls_960_model_hubert_base_ls960/L9_200.kmodel
km=200
layer=9
kernels="3_5_5_5"

# jobs
cmd=run.pl
nj=4
denoise_nj=10
device=0
stage=1
stop_stage=3

dumpdir=

. ./path.sh
. utils/parse_options.sh

if [ -z $dumpdir ]; then
    dumpdir=/mnt/lustre02/scratch/sjtu/home/ww089/fairseqs/fairseq-low-resource/examples/hubert/dump/data_${lang}_`basename $src`_model_`basename $model .pt`
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
    echo "dump train km label for layer $layer km $km"
    local/utils/dump_km_label.sh --feat_dir ${feat_dir} --split train --km_path ${km_path} --nj 4 --dst ${km_dir} --device $device

    for i in `seq 0 $max_job_num`; do cat ${km_dir}/train_${i}_${nj}.km; done > ${km_dir}/train.km

    echo "dump valid km label for layer $layer km $km"
    local/utils/dump_km_label.sh --feat_dir ${feat_dir} --split valid --km_path ${km_path} --nj 4 --dst ${km_dir} --device $device

    for i in `seq 0 $max_job_num`; do cat ${km_dir}/valid_${i}_${nj}.km; done > $km_dir/valid.km
fi

# rm $dumpdir/feat_L$layer/*npy

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
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
