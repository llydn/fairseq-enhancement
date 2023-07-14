import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from argparse import ArgumentParser



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--result_tsv", type=str, help='Path to the result.tsv')
    args = parser.parse_args()

    tsv_name = os.path.basename(args.result_tsv).strip().split('.')[0]

    raw_df = pd.read_csv(args.result_tsv, sep="\t")
    raw_df = raw_df.set_index('decode_path')
    # decode_path fusion enhance_model finetune_set test_set checkpoint wer
    

    processed = raw_df.groupby(['fusion', 'enhance_model', 'finetune_set', 'test_set']).agg(wer_min=pd.NamedAgg(column='wer', aggfunc='min')).reset_index()


    test_sets = ['chime4_dt05_real', 'chime4_et05_real', \
                 'ls_test_clean', 'ls_test_other', \
                 'ls_test_clean_wham_-5_0db', 'ls_test_clean_wham_0_5db', 'ls_test_clean_wham_5_10db', 'ls_test_clean_wham_10_15db', 'ls_test_clean_wham_15_20db', \
                    'ls_test_other_wham_-5_0db', 'ls_test_other_wham_0_5db', 'ls_test_other_wham_5_10db', 'ls_test_other_wham_10_15db', 'ls_test_other_wham_15_20db']

    new_df = pd.DataFrame(columns=['finetune_set', 'enhance_model', 'fusion'] + test_sets)
    wer_dict = {}
    for idx, row in processed.iterrows():
        wer_dict[(row['fusion'], row['enhance_model'], row['finetune_set'], row['test_set'])] = row['wer_min']

    for key, value in tqdm(wer_dict.items()):
        new_row = pd.Series({'finetune_set': key[2], 'enhance_model': key[1], 'fusion': key[0], key[3]: value})
        
        new_df = pd.concat([new_df, pd.DataFrame([new_row])],axis=0,ignore_index=True)

    new_df = new_df.groupby(['finetune_set', 'enhance_model', 'fusion']).agg({k:'min' for k in test_sets}).reset_index()
    # print(new_df)
    new_df.to_csv(os.path.join(os.path.dirname(args.result_tsv), f'{tsv_name}_summary.tsv'), sep="\t", index=False)
    print(f"Saved to {os.path.join(os.path.dirname(args.result_tsv), f'{tsv_name}_summary.tsv')}")