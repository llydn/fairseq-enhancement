set -e
set -u
set -o pipefail

log() {
    echo -e $*
}

text=/mnt/lustre/sjtu/home/ww089/espnets/espnet-text-adapt/egs2/librispeech/asr1/data/local/other_text/workspace/text.5-30.10k
uttids=/mnt/lustre02/scratch/sjtu/home/ww089/fairseqs/fairseq-low-resource/examples/hubert/dump/phn_sdg/text_10k_speech_ll_10h/utt2num_samples
dst=/mnt/lustre02/scratch/sjtu/home/ww089/fairseqs/fairseq-low-resource/examples/hubert/dump/phn_sdg/text_10k_speech_ll_10h/ltr

. ./path.sh
. utils/parse_options.sh

python local/utils/splice_data_generation/make_label.py $uttids $text $dst

