from argparse import ArgumentParser
from tqdm import tqdm
import kaldiio
import soundfile as sf
import scipy.io.wavfile as wavfile

# python local/utils/data/scp2dur.py \
#     --wav_scp data/train/wav.scp \
#     --utt2num_samples data/train/utt2num_samples

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--wav_scp", type=str, help='Path to the wav.scp')
    parser.add_argument("--utt2num_samples", type=str, help='Path to the utt2num_samples')
    args = parser.parse_args()

    num_lines = sum(1 for _ in open(args.wav_scp))
    with open(args.wav_scp, 'r') as ifp, open(args.utt2num_samples, 'w') as ofp:
        for line in tqdm(ifp, total=num_lines):
            uttid, wav_path = line.strip().split()
            sr, wav = kaldiio.load_mat(wav_path)
            # wav, sr = sf.read(wav_path, dtype='float32')
            # sr, wav = wavfile.read(wav_path)
            ofp.write(f"{uttid} {len(wav)}\n")

