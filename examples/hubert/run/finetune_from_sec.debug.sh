#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

config_name=base_100h
config_dir=config/finetune/1gpu
expdir=exp/finetune
hubert_tag=
data=data/ls_clean_100
init_dir=/mnt/lustre/sjtu/home/ww089/fairseqs/fairseq-dny/egs/sec/exp/hubert_base_librispeech_fusion_interpolate_residual_lr_2e-4_train_ls_960_enh_label_km_L9_500
checkpoint_name=checkpoint_20_50000.pt
fusion=interpolate

export NCCL_DEBUG=INFO
export NCCP_P2P_DISABLE=1

. ./path.sh
. utils/parse_options.sh

init=$init_dir/checkpoints/$checkpoint_name
data=`realpath $data`
init=`realpath $init`
checkpoint_base_name=${checkpoint_name%.*}
checkpoint_base_name=${checkpoint_base_name##*_}

if [ -z "${hubert_tag}" ]; then
    hubert_tag=${config_name}_finetune_$(basename "${data}")_from_fusion_${fusion}_${checkpoint_base_name}
fi

hubert_exp=$expdir/${hubert_tag}
fairseq-hydra-train \
  --config-dir ${config_dir} \
  --config-name ${config_name} \
  task.data=${data} task.label_dir=${data} \
  model.w2v_path=${init} hydra.run.dir=${hubert_exp}

