#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

config_name=hubert_base_librispeech
config_dir=config/pretrain/phones/8gpu
expdir=exp
hubert_tag=hubert_base_librispeech_train_ls_960_label_phones
data=data/ls_960
#label=dump/data_ls_960_model_hubert_base_ls960/km_L9_500/
label=dump/data_ls_960_model_hubert_base_ls960/denoise/
model=models/hubert_base_ls960.pt

export NCCL_DEBUG=INFO
export NCCP_P2P_DISABLE=1


. ./path.sh
. utils/parse_options.sh

data=`realpath $data`
label=`realpath $label`
model=`realpath $model`

if [ -z "${hubert_tag}" ]; then
    hubert_tag=${config_name}_train_$(basename "${data}")_label_$(basename "${label}")
fi
hubert_exp=$expdir/${hubert_tag}

fairseq-hydra-train \
  --config-dir ${config_dir} \
  --config-name ${config_name} \
  task.data=${data} task.label_dir=${label} task.labels='["km"]' model.label_rate=50 hydra.run.dir=${hubert_exp} model.init=${model}

