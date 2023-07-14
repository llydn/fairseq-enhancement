#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

config_name=hubert_base_librispeech
config_dir=config/pretrain/sec/4gpu
expdir=exp
hubert_tag=
data=data/ls_960
label=dump/data_ls_960_model_hubert_base_ls960/km_L9_500/
model=models/hubert_base_ls960.pt
enh_models=models/enh_models.json
enh_models=`realpath $enh_models`
fusion=

export NCCL_DEBUG=INFO
export NCCP_P2P_DISABLE=1


. ./path.sh
. utils/parse_options.sh

data=`realpath $data`
label=`realpath $label`
model=`realpath $model`

if [ -z "${hubert_tag}" ]; then
    hubert_tag=${config_name}_train_$(basename "${data}")_noisy_enh_label_$(basename "${label}")_librimix
fi
hubert_exp=$expdir/${hubert_tag}

fairseq-hydra-train \
  --config-dir ${config_dir} \
  --config-name ${config_name} \
  task.data=${data} task.label_dir=${label} task.labels='["km"]' \
  task.noises='["/mnt/lustre/sjtu/home/ww089/workspace/tmp/noise.manifest"]' \
  task.num_noise='[1,2]' task.aug_types='["additive"]' task.noise_snr='[5,10]' \
  model.label_rate=50 hydra.run.dir=${hubert_exp} model.init=${model} model.enh_models=${enh_models}

