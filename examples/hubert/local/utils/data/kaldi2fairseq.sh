# for lang in dutch german english frisian; do bash local/utils/data/kaldi2fairseq.sh --src /mnt/lustre02/scratch/sjtu/home/ww089/rawdata/common_voice/taslp_low_resource/dump/raw/train_${lang}_10h --dst data/common_voice/${lang}/tr_10h --split train --make_dict true; done
# for lang in german english frisian; do bash local/utils/data/kaldi2fairseq.sh --src /mnt/lustre02/scratch/sjtu/home/ww089/rawdata/common_voice/taslp_low_resource/dump/raw/dev_${lang} --dst data/common_voice/${lang}/tr_10h --split valid; done

set -o pipefail
set -e

src=
dst=
split=train
text=text_norm
prefix=/mnt/lustre02/scratch/sjtu/home/ww089/rawdata/common_voice/taslp_low_resource/
make_dict=false

. ./path.sh
. utils/parse_options.sh


mkdir -p $dst

paste -d" " $src/wav.scp <(cut -d" " -f2 $src/utt2num_samples) > $dst/${split}.manifest

echo $prefix > $dst/${split}.tsv
cut -d" " -f2- $dst/${split}.manifest | sed 's/ /\t/' | sed "s#${prefix}##" >> $dst/${split}.tsv

cp $src/$text $dst/${split}.text
python local/utils/data/make_labels.py --text $dst/${split}.text --dir ${dst} --split ${split}


if $make_dict; then
    awk '
    {
        split(toupper($0), chars, ""); 
        for (i=1; i<=length($0); i++)
            a[chars[i]] = (chars[i] in a) ? a[chars[i]] + 1 : 1;
    }
    END {
        for (c in a) 
            if (c != " ") 
                print c,a[c];
    }' $dst/${split}.ltr | sort -k2 -nr > $dst/dict.ltr.txt
fi
