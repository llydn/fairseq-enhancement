set -e
set -u
set -o pipefail

log() {
    echo -e $*
}

src1=
src2=
dst=
python=python3

. ./path.sh
. utils/parse_options.sh



mkdir -p $dst
${python} local/utils/splice_data_generation/merge_shelve.py --input $src1/token2segments.shelve $src2/token2segments.shelve --output ${dst}/token2segments.shelve
${python} local/utils/splice_data_generation/merge_shelve.py --input $src1/uttid2combinations.shelve $src2/uttid2combinations.shelve --output ${dst}/uttid2combinations.shelve

for f in utt2num_samples ltr db_wav.scp; do
    cat $src1/$f $src2/$f > $dst/$f
    awk '{if (!($1 in a)) {a[$1]=1; print $0;}}' $dst/$f > $dst/$f.tmp
    mv $dst/$f.tmp $dst/$f
done
