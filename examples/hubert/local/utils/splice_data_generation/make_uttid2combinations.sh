set -e
set -u
set -o pipefail

log() {
    echo -e $*
}

dst=/mnt/lustre02/scratch/sjtu/home/ww089/fairseqs/fairseq-low-resource/examples/wavlm/dump/data_ls_960_model_wavlm_base_plus/km_L7_500/kernel_3_5_5_5/sdg/oracle_ls960
ssl_unit=/mnt/lustre02/scratch/sjtu/home/ww089/fairseqs/fairseq-low-resource/examples/hubert/dump/data_ls_960_model_hubert_base_ls960/km_L9_500/kernel_3_5_5_5/train_denoise.ssl_unit
wav_scp=/mnt/lustre/sjtu/home/ww089/espnets/espnet-text-adapt/egs2/gigaspeech/asr_you_no_overlap/dump/raw/ls_train_960/wav.scp
max_n_token=10
min_n_token=3
tokendb_dir=/mnt/lustre02/scratch/sjtu/home/ww089/fairseqs/fairseq-low-resource/examples/wavlm/dump/data_ls_960_model_wavlm_base_plus/km_L7_500/kernel_3_5_5_5/sdg/tokendb/
nj=4
cmd=run.pl
python=python3
forbid_self=false

. ./path.sh
. utils/parse_options.sh

mkdir -p ${dst}/logs
log "logging to '${dst}/logs/make_uttid2combinations.n.log'"
${cmd} JOB=1:"${nj}" "${dst}"/logs/make_uttid2combinations.JOB.log \
    ${python} local/utils/splice_data_generation/make_uttid2combinations.py \
        --nj ${nj} \
        --text_unit ${ssl_unit} \
        --forbid_self ${forbid_self} \
        --max_n_token ${max_n_token} \
        --min_n_token ${min_n_token} \
        --rank "JOB" \
        --tokendb_dir ${tokendb_dir} \
        --utt2num_samples "${dst}/utt2num_samples.JOB" \
        --token2segments "${dst}/token2segments.JOB.shelve" \
        --uttid2combinations "${dst}/uttid2combinations.JOB.shelve"

log "merging into ${dst}/uttid2combinations.shelve"
to_merge=""
for n in `seq 1 ${nj}`; do
    to_merge+=" ${dst}/uttid2combinations.$n.shelve";
done
${python} local/utils/splice_data_generation/merge_shelve.py --input ${to_merge} --output ${dst}/uttid2combinations.shelve

log "merging into ${dst}/token2segments.shelve"
to_merge=""
for n in `seq 1 ${nj}`; do
    to_merge+=" ${dst}/token2segments.$n.shelve";
done
${python} local/utils/splice_data_generation/merge_shelve.py --input ${to_merge} --output ${dst}/token2segments.shelve

[ -f ${dst}/utt2num_samples ] && rm ${dst}/utt2num_samples
for n in `seq 1 ${nj}`; do
    cat "${dst}/utt2num_samples.$n" >> ${dst}/utt2num_samples
done

cp $wav_scp $dst/db_wav.scp
