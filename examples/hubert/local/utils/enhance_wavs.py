import os
import argparse
import torch
import json
import numpy as np
import re
import kaldiio
from tqdm import tqdm
import asteroid
import soundfile as sf



def read_audio(path):
    if re.match(r".*\.ark:\d+", path):
        sample_rate, wav = kaldiio.load_mat(path)
    else:
        wav, sample_rate = sf.read(path)
    return wav, sample_rate



def inference(model, source_scp, output_dir, batch_size):
    with open(source_scp) as f:
        lines = f.readlines()

    batch_len = len(lines) // batch_size if len(lines) % batch_size == 0 else len(lines) // batch_size + 1

    for i in tqdm(range(batch_len)):
        lines_batch = lines[i*batch_size: (i+1)*batch_size]
        inference_batch(model, lines_batch, output_dir)

    print(f"Saved enhanced wav to {output_dir}/data_wav.ark and {output_dir}/wav.scp")

def collater_audio(audios, audio_size):
    collated_audios = audios[0].new_zeros(len(audios), audio_size)
    padding_mask = (
        torch.BoolTensor(collated_audios.shape).fill_(False)
        # if self.pad_audio else None
    )
    audio_starts = [0 for _ in audios]
    for i, audio in enumerate(audios):
        diff = len(audio) - audio_size
        assert diff <= 0, (
            f"Audio {i} is longer ({len(audio)}) than the max size ({audio_size})"
        )
        if diff == 0:
            collated_audios[i] = audio
        elif diff < 0:
            collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0.0)])
            padding_mask[i, diff:] = True
    return collated_audios, padding_mask, audio_starts

def inference_batch(model, lines_batch, output_dir):
    uttids = []
    wavs_noisy = []
    wavs_len = []
    for line_noisy in lines_batch:
        uttid, noisy_path = line_noisy.strip().split(maxsplit=1)
        wav_noisy, sr = read_audio(noisy_path)
        uttids.append(uttid)
        wavs_noisy.append(torch.from_numpy(wav_noisy).to(torch.float32).to(device))
        wavs_len.append(wav_noisy.shape[-1])

    max_wav_len = max(wavs_len)
    wavs_noisy, padding_mask, _ = collater_audio(wavs_noisy, max_wav_len)
    
    with torch.no_grad():
        wavs_enh = model(wavs_noisy).squeeze(1).detach().cpu()
        wavs_enh = wavs_enh / wavs_enh.abs().max(dim=-1, keepdim=True)[0]

    wavs_enh = [wavs_enh[i,padding_mask[i]==0].numpy() for i in range(wavs_enh.shape[0])]
    kaldiio.save_ark(
        os.path.join(output_dir, "data_wav.ark"),
        {uttid: (wav_enh, 16000) for uttid, wav_enh in zip(uttids, wavs_enh)},
        scp=os.path.join(output_dir, "wav.scp"),
        append=True,
        write_function=f"soundfile",
    )
            


if __name__ == "__main__":
    '''
    python local/utils/enhance_wavs.py \
        --noisy_scp data/ls_test_clean_wham_0_5db/test.scp \
        --model_dict models/enh_models.json \
        --models DPTNet DCCRNet DCUNet \
        --output_dir data/ls_test_clean_wham_0_5db/enhanced_wavs \
        --batch_size 1
    '''

    parser = argparse.ArgumentParser(description="Enhance wavs using pretrained model")
    parser.add_argument("--noisy_scp", default="data/ls_test_clean_wham_0_5db/test.scp", help="scp file for noisy wav")
    parser.add_argument("--model_dict", default="models/enh_models.json", help="JSON file containing model names and their paths")
    parser.add_argument("--models", default=["DPTNet", "DCCRNet"], type=str, nargs='+', help="model names")
    parser.add_argument("--output_dir", help='dump dir for data_wav.ark and wav.scp')
    parser.add_argument("--batch_size", '-b', type=int, default=1, help="batch size")

    args = parser.parse_args()

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dict = json.load(open(args.model_dict, 'r')) # dict: model_name -> model_path
    models = {}
    for model_name in args.models:
        assert model_name in model_dict, f"model {model_name} not found in {args.model_dict}"
        model_path = model_dict[model_name]
        model = getattr(asteroid, model_name).from_pretrained(model_path).to(device)
        models[model_name] = model

    os.makedirs(args.output_dir, exist_ok=True)
    assert os.path.isdir(args.output_dir), f"{args.output_dir} is not a directory"

    for model_name, model in models.items():
        print(f"Using {model_name} to enhance wavs...")

        output_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.realpath(output_dir)

        inference(model, args.noisy_scp, output_dir, args.batch_size)

        torch.cuda.empty_cache()