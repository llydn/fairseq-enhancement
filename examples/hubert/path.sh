# activate conda env
if [ -z "${PS1:-}" ]; then
    PS1=__dummy__
fi
. /mnt/lustre/sjtu/home/dny20/espnets/espnet-baseline/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate fairseq-enhancement

module load cuda/10.2 cmake/3.12.0 gcc/7.3.0

export FAIRSEQ_ROOT=$PWD/../../
export PATH=$FAIRSEQ_ROOT/tools/src/kenlm/build/bin:$PWD/utils/:$PATH

