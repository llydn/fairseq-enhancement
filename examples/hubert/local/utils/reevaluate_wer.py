import os
import editdistance
import re
from argparse import ArgumentParser


if __name__ == "__main__":
    # This script is used to filter the sentences where WER>=threshold in infer.log and give a new score.
    #
    # python local/utils/reevaluate_wer.py 
    # --infer_log exp/finetune/base_100h_finetune_ls_clean_100_from_enhfromlibrimix_42950/decode/ls_test_clean/DCUNet/checkpoint_best/viterbi_pad_audio/infer.log
    
    parser = ArgumentParser()
    parser.add_argument("--infer_log", type=str, default="infer.log", help="infer log file")
    parser.add_argument("--new_wer_file_name", type=str, default="wer.filtered", help="new wer file")
    parser.add_argument("--threshold", type=float, default=100.0, help="The threshold of WER above which the sentence should be eliminated.")
    args = parser.parse_args()

    with open(args.infer_log, "r") as f:
        lines = f.readlines()
    

    total_errs, total_length = 0., 0.
    old_wer = None
    i = 0
    while i < len(lines):
        if re.match(r".*- HYPO: ", lines[i]):
            assert re.match(r".*- REF: ", lines[i+1]) # Must be paired.
            hypo_start_idx = re.match(r".*- HYPO: ", lines[i]).span()[1]
            hypo_words = lines[i][hypo_start_idx:].strip().split()
            tgt_start_idx = re.match(r".*- REF: ", lines[i+1]).span()[1]
            tgt_words = lines[i+1][tgt_start_idx:].strip().split()
            errs, length = editdistance.eval(hypo_words, tgt_words), len(tgt_words)
            sentence_wer = errs * 100.0 / length
            if sentence_wer < args.threshold - 1e-5:
                total_errs += errs
                total_length += length
            else: # debug
                print("HYPO:", " ".join(hypo_words))
                print("REF:", " ".join(tgt_words))
                print("Sentence WER:", sentence_wer)
            i += 1
        elif re.match(r".*- Word error rate: ", lines[i]):
            old_wer_start = re.match(r".*- Word error rate: ", lines[i]).span()[1]
            old_wer = lines[i][old_wer_start:].strip()
            assert total_length > 0
            new_wer = total_errs * 100.0 / total_length
        elif re.match(r".*current directory is", lines[i]):                
            total_errs, total_length = 0., 0.
        i += 1

    if old_wer is None:
        print(f"{args.infer_log} not completed.")
    else:
        new_wer_file_path = os.path.join(os.path.dirname(args.infer_log), args.new_wer_file_name)
        with open(new_wer_file_path, "w") as f:
            f.write(
                (
                    f"WER: {new_wer}\n"
                    f"err / num_ref_words = {total_errs} / {total_length}\n\n"
                )
            )

        print(f"Old WER: {old_wer}\tNew WER: {new_wer}\nResult written to file {new_wer_file_path}")
    
            
            
