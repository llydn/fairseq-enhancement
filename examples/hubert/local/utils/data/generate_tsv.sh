#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

data_dir=$1
scp_name=$2  # wav.scp
prefix=$3 # /mnt/lustre/sjtu/home/dny20/fairseqs/fairseq-enhancement/examples/hubert/data/

python local/utils/data/scp2dur.py --wav_scp $data_dir/${scp_name} --utt2num_samples $data_dir/utt2num_samples

paste -d" " ${data_dir}/${scp_name} <(cut -d" " -f2 ${data_dir}/utt2num_samples) > ${data_dir}/test.manifest
echo $prefix > ${data_dir}/test.tsv
cut -d" " -f2- ${data_dir}/test.manifest | sed 's/ /\t/' | sed "s#${prefix}##" >> ${data_dir}/test.tsv


mv ${data_dir}/${scp_name} ${data_dir}/test.scp
exit 0