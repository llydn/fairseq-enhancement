if [ "$#" -eq 0  ]; then
    salloc -p 2080ti,gpu --gres=gpu:1 --nodes 8 --ntasks-per-node 1 --mem=6G -c 1
elif [ "$1" -eq 2 ]; then
    salloc -p 2080ti,gpu --gres=gpu:1 --nodes 8 --ntasks-per-node 1 --mem=8G -c 2
elif [ "$1" -eq 3 ]; then
    salloc -p 2080ti,gpu --gres=gpu:1 --nodes 8 --ntasks-per-node 1 --mem=10G -c 3
fi


