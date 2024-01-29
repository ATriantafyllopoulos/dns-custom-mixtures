import argparse
import audiofile
import audtorch
import librosa
import numpy as np
import os
import pandas as pd
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
    parser.add_argument("--logfile", required=True)
    parser.add_argument("--speech", required=True)
    parser.add_argument("--noise", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--sr", default=16000, type=int)
    args = parser.parse_args()

    df = pd.read_csv(args.logfile, sep=" ", header=None)
    df = df.rename(columns={0: "speech", 1: "noise", 2: "snr"})

    os.makedirs(args.dest, exist_ok=True)

    demand_map = {
        "KITCHEN": "DKITCHEN",
        "LIVING": "DLIVING",
        "WASHING": "DWASHING",
        "FIELD": "NFIELD",
        "PARK": "NPARK",
        "RIVER": "NRIVER",
        "HALLWAY": "OHALLWAY",
        "MEETING": "OMEETING",
        "OFFICE": "OOFFICE",
        "CAFE": "PCAFETER",
        "RESTO": "PRESTO",
        "STATION": "PSTATION",
        "PSQUARE": "SPSQUARE",
        "TRAFFIC": "STRAFICT",
        "BUS": "TBUS",
        "CAR": "TCAR",
        "METRO": "TMETRO"
    }

    files = df["speech"].values
    noises = df["noise"].values
    snrs = df["snr"].values
    emb_distance = [0, 10, 20, 30, 60, 90, 120]
    embedding_size = 1
    for file, noise_file, snr in tqdm.tqdm(zip(files, noises, snrs), total=len(files), desc="Mixing"):
        speech = librosa.load(os.path.join(args.speech, file + ".wav"), sr=args.sr, mono=True)[0]
        os.makedirs(os.path.join(args.dest, "speech"), exist_ok=True)
        audiofile.write(os.path.join(args.dest, "speech", f"{file}.wav"), speech, args.sr)
        noise = librosa.load(os.path.join(args.noise, demand_map[noise_file.upper()], "ch01.wav"), sr=args.sr, mono=True)[0]

        for distance in emb_distance:
            start = noise.shape[0] - args.sr * 10 - (distance + embedding_size) * args.sr
            end = start + embedding_size * args.sr
            embedding = noise[start:end]
            os.makedirs(os.path.join(args.dest, f"emb_{distance}"), exist_ok=True)
            audiofile.write(os.path.join(args.dest, f"emb_{distance}", f"{file}.wav"), embedding, args.sr)

        noise = noise[-args.sr*10:]
        os.makedirs(os.path.join(args.dest, "noise"), exist_ok=True)
        audiofile.write(os.path.join(args.dest, "noise", f"{file}.wav"), noise, args.sr)

        mixture = segmental_snr_mixer(speech, noise, snr)
        os.makedirs(os.path.join(args.dest, "mixture"), exist_ok=True)
        audiofile.write(os.path.join(args.dest, "mixture", f"{file}.wav"), mixture, args.sr)
