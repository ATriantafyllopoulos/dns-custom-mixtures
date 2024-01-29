import argparse
import audiofile
import audtorch
import audtorch.transforms.functional as F
import glob
import librosa
import numpy as np
import os
import pandas as pd
import random
import tqdm

EPS = np.finfo(float).eps

def is_clipped(audio, clipping_threshold=0.99):
    return any(abs(audio) > clipping_threshold)

def normalize_segmental_rms(audio, rms, target_level=-25):
    """Normalize the signal to the target level based on segmental RMS."""
    scalar = 10 ** (target_level / 20) / (rms+EPS)
    audio = audio * scalar
    return audio

def active_rms(clean, noise, fs=16000, energy_thresh=-50):
    """Returns the clean and noise RMS of the noise calculated only in the active portions"""
    window_size = 100 # in ms
    window_samples = int(fs*window_size/1000)
    sample_start = 0
    noise_active_segs = []
    clean_active_segs = []

    while sample_start < len(noise):
        sample_end = min(sample_start + window_samples, len(noise))
        noise_win = noise[sample_start:sample_end]
        clean_win = clean[sample_start:sample_end]
        noise_seg_rms = (noise_win**2).mean()**0.5
        # Considering frames with energy
        if noise_seg_rms > energy_thresh:
            noise_active_segs = np.append(noise_active_segs, noise_win)
            clean_active_segs = np.append(clean_active_segs, clean_win)
        sample_start += window_samples

    if len(noise_active_segs) != 0:
        noise_rms = (noise_active_segs**2).mean()**0.5
    else:
        noise_rms = EPS
        
    if len(clean_active_segs) != 0:
        clean_rms = (clean_active_segs**2).mean()**0.5
    else:
        clean_rms = EPS

    return clean_rms, noise_rms


def segmental_snr_mixer(
        clean, 
        noise, 
        snr, 
        target_level=-25, 
        clipping_threshold=0.99,
        target_level_lower=-35,
        target_level_upper=-15
    ):
    """Function to mix clean speech and noise at various segmental SNR levels"""
    # crop noise to fit speech
    # if longer it pads with zeros
    crop = audtorch.transforms.RandomCrop(clean.shape[0])
    noise = crop(noise)
    
    rmsclean, rmsnoise = active_rms(clean=clean, noise=noise)
    clean = normalize_segmental_rms(clean, rms=rmsclean, target_level=target_level)
    noise = normalize_segmental_rms(noise, rms=rmsnoise, target_level=target_level)
    # Set the noise level for a given SNR
    noisescalar = rmsclean / (10**(snr/20)) / (rmsnoise+EPS)
    noisenewlevel = noise * noisescalar

    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel
    return noisyspeech


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Mix VoiceBank-DEMAND")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--speech", required=True)
    parser.add_argument("--noise", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--sr", default=16000, type=int)
    args = parser.parse_args()

    random.seed(23)
    np.random.seed(23)

    df = pd.read_csv(args.csv)
    os.makedirs(args.dest, exist_ok=True)
    files = df["fname"].values.tolist()
    speech_files = glob.glob(os.path.join(args.speech, "*.wav"))

    data = []
    for index, file in tqdm.tqdm(enumerate(files), total=len(files), desc="Mixing"):
        filename = f"{index:04}.wav"
        speech_file = random.choice(speech_files)
        speech = librosa.load(speech_file, sr=args.sr, mono=True)[0]
        speech = F.normalize(speech)
        os.makedirs(os.path.join(args.dest, "speech"), exist_ok=True)
        audiofile.write(os.path.join(args.dest, "speech", filename), speech, args.sr)
        noise = librosa.load(os.path.join(args.noise, f"{file}.wav"), sr=args.sr, mono=True)[0]
        noise = F.normalize(noise)

        energy = librosa.feature.rms(y=noise)[0]
        rel_energy = energy / energy.max() * 100
        ead = np.where(rel_energy > 20)[0]
        start = ead[0] * 512
        end = ead[-1] * 512
        noise = noise[start:end]

        embedding = noise[:args.sr]
        os.makedirs(os.path.join(args.dest, "embedding-same"), exist_ok=True)
        audiofile.write(os.path.join(args.dest, "embedding-same", filename), embedding, args.sr)

        noise = noise[args.sr:]
        os.makedirs(os.path.join(args.dest, "noise"), exist_ok=True)
        audiofile.write(os.path.join(args.dest, "noise", filename), noise, args.sr)

        snr = np.random.randint(-5, 5)
        mixture = segmental_snr_mixer(speech, noise, snr)
        os.makedirs(os.path.join(args.dest, "mixture"), exist_ok=True)
        audiofile.write(os.path.join(args.dest, "mixture", filename), mixture, args.sr)
        data.append({
            "speech": os.path.splitext(os.path.basename(speech_file))[0],
            "noise": file,
            "snr": snr
        })
    pd.DataFrame(data).to_csv(os.path.join(args.dest, "filelist.csv"), index=False)