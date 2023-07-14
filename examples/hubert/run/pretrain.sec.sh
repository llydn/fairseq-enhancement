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
debug_mode=3
noises='["/mnt/lustre/sjtu/home/dny20/fairseqs/fairseq-enhancement/examples/hubert/data/wham/noise.manifest"]'
# noises='["/mnt/lustre/sjtu/home/ww089/fairseqs/fairseq-220604/examples/hubert/data/musan/music.manifest", "/mnt/lustre/sjtu/home/ww089/fairseqs/fairseq-220604/examples/hubert/data/musan/noise.manifest"]' 
fusion=mean
fusion_head_nums=4
fusion_single=False
fusion_last=False

export NCCL_DEBUG=INFO
export NCCP_P2P_DISABLE=1


. ./path.sh
. utils/parse_options.sh

data=`realpath $data`
label=`realpath $label`
model=`realpath $model`

if [ -z "${hubert_tag}" ]; then
    hubert_tag=${config_name}_fusion_${fusion}_train_$(basename "${data}")_enh_label_$(basename "${label}")
fi
hubert_exp=$expdir/${hubert_tag}

fairseq-hydra-train \
  --config-dir ${config_dir} \
  --config-name ${config_name} \
  task.data=${data} task.label_dir=${label} task.labels='["km"]' \
  task.noises=$noises \
  task.num_noise='[1,2]' task.aug_types='["additive"]' task.noise_snr='[5,10]' \
  model.fusion_layer=$fusion \
  model.fusion_head_nums=$fusion_head_nums \
  model.fusion_single=$fusion_single \
  model.fusion_last=$fusion_last \
  model.label_rate=50 hydra.run.dir=${hubert_exp} model.init=${model} model.enh_models=${enh_models} model.debug_mode=${debug_mode}

