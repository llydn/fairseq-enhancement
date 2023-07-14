set -e
set -u
set -o pipefail

log() {
    echo -e $*
}

# directories
km=/mnt/lustre02/scratch/sjtu/home/ww089/fairseqs/fairseq-low-resource/examples/wavlm/dump/data_ls_960_model_wavlm_base_plus/km_L6_50/kernel_3_5_5_5/train_denoise.km
uttid=/mnt/lustre/sjtu/home/ww089/fairseqs/fairseq-220604/examples/hubert/data/ls_960/train.manifest


# hyper parameters
frame_shift=0.02
tokendb_min_n_token=3
tokendb_max_n_token=10
tokendb_token_variety_threshold=2
stage=0
stop_stage=1

. path.sh
. utils/parse_options.sh

workdir=`dirname $km`
ali=${km%%.*}.ali
ctm=${km%%.*}.ctm
ssl_unit=${km%%.*}.ssl_unit

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Stage 0: preprocess"

    log "generating $ali"
    paste -d" " <(cut -d" " -f1 $uttid) $km > $ali

    log "generating $ctm"
    awk -v frame_shift=${frame_shift} -f local/utils/splice_data_generation/ali2ctm.awk $ali > $ctm

    log "generating ${ssl_unit}"
    cat $ali | sed 's/$/ |/' | tr ' ' '\n' | uniq | tr '\n|' ' \n' | sed 's/^ //' | sed 's/ $//' > ${ssl_unit}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    tokendb_dir=${workdir}/sdg/tokendb
    mkdir -p ${tokendb_dir}
    for n_token in `seq ${tokendb_min_n_token} ${tokendb_max_n_token}`; do
        log "generating token stats for ${n_token} token"
        python local/utils/splice_data_generation/ctm2stats.py --ctm $ctm --n-token ${n_token} --output ${tokendb_dir}/${n_token}.pkl --token-variety-threshold ${tokendb_token_variety_threshold}
    done
fi
