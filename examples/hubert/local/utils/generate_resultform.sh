#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

expdirs=$1 # "exp/finetune/base_100h_finetune_ls_clean_100_from_fusion_interpolate_50000 exp/finetune/base_100h_finetune_ls_clean_100_from_fusion_interpolate_40000"
output_dir=RESULTS
decode_subdir=viterbi_pad_audio
wer_file=wer.filtered

# decode_path fusion enhance_model finetune_set test_set checkpoint wer

. ./path.sh
. utils/parse_options.sh

cwd=$(pwd)
mkdir -p $output_dir

convert_to_relative_path() {
    local path=$1
    local prefix=$2
    echo $path | sed "s#${prefix}##" | sed "s#^/##"
}

output_file=$output_dir/result_$(date +'%m_%d_%H_%M').tsv
echo -e "decode_path\tfusion\tenhance_model\tfinetune_set\ttest_set\tcheckpoint\twer" > $output_file

if [ -z wer_suffix ]; then
    wer_file="wer.*"
fi

for expdir in $expdirs; do
    echo "Processing $expdir ..."
    for decode_wer_path in ${expdir}/decode/*/*/*/${decode_subdir}/${wer_file}; do
        if [ ! -e $decode_wer_path ]; then
            continue
        fi
        decode_wer_path=$(convert_to_relative_path $decode_wer_path $cwd)
        wer=$(head -n 1 $decode_wer_path | sed "s/WER: //")
        dir_base=$(echo $decode_wer_path | cut -d"/" -f3)
        fusion=$(echo $dir_base | awk -F "from_" '{print $2}'| sed -r "s#_[0-9]+##")
        finetune_set=$(echo $dir_base |  awk -F "finetune_|_from" '{print $2}')
        test_set=$(echo $decode_wer_path | cut -d"/" -f5)
        enhance_model=$(echo $decode_wer_path | cut -d"/" -f6)
        checkpoint=$(echo $decode_wer_path | cut -d"/" -f7)
        # line="${decode_wer_path} ${fusion} ${enhance_model} ${finetune_set} ${test_set} ${checkpoint} ${wer}"
        echo -e "${decode_wer_path}\t${fusion}\t${enhance_model}\t${finetune_set}\t${test_set}\t${checkpoint}\t${wer}" >> $output_file
    done
done

echo "Result saved to $output_file"