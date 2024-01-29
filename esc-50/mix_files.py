import argparse
import audiofile as af
import audtorch
import glob
import numpy as np
import os
import pandas as pd
import random
import torchaudio
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
        target_level_upper=-15,
        sr=24000
    ):
    """Function to mix clean speech and noise at various segmental SNR levels"""
    # crop noise to fit speech
    # if longer it pads with zeros
    crop = audtorch.transforms.RandomCrop(clean.shape[0])
    noise = crop(noise)
    
    rmsclean, rmsnoise = active_rms(clean=clean, noise=noise, fs=sr)
    clean = normalize_segmental_rms(clean, rms=rmsclean, target_level=target_level)
    noise = normalize_segmental_rms(noise, rms=rmsnoise, target_level=target_level)
    # Set the noise level for a given SNR
    noisescalar = rmsclean / (10**(snr/20)) / (rmsnoise+EPS)
    noisenewlevel = noise * noisescalar

    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel
    return noisyspeech


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Mix VCTK with ESC50")
    parser.add_argument("speech")
    parser.add_argument("noise")
    parser.add_argument("dest")
    parser.add_argument("--sr", default=24000, type=int)
    args = parser.parse_args()

    random.seed(23)
    np.random.seed(23)

    files = glob.glob(os.path.join(args.speech, "*.wav"))
    
    df = pd.read_csv(os.path.join(args.noise, "meta", "esc50.csv"))
    print(df)
    print(df["category"].unique())
    print(df["take"].unique())
    print(len(files))
    print(len(df.loc[(df["fold"] == 1)]))
    print(df.loc[(df["fold"] == 1), "category"].value_counts().describe())

    files = np.random.choice(files, len(df.loc[(df["fold"] == 1)]), replace=False).tolist()
    ratios = [-15, -5, 0, 5, 15, 25]

    noise_files = sorted(list(df.loc[(df["fold"] == 1), "filename"].values))
    alternative_embeddings = df.loc[(df["fold"] == 2)].sort_values(by=["category", "filename"])

    os.makedirs(args.dest, exist_ok=True)
    for snr in ratios:
        category_counter = {
            key: 0
            for key in df["category"].unique()
        }
        os.makedirs(os.path.join(args.dest, f"SNR_{snr}", "speech"), exist_ok=True)
        os.makedirs(os.path.join(args.dest, f"SNR_{snr}", "noise"), exist_ok=True)
        os.makedirs(os.path.join(args.dest, f"SNR_{snr}", "mixture"), exist_ok=True)
        os.makedirs(os.path.join(args.dest, f"SNR_{snr}", "embedding-same"), exist_ok=True)
        os.makedirs(os.path.join(args.dest, f"SNR_{snr}", "embedding-diff"), exist_ok=True)
        filenames = {
            "noise": [],
            "speech": [],
            "embedding-diff": []
        }
        for index, noise_file in tqdm.tqdm(enumerate(noise_files), total=len(noise_files), desc=f"SNR_{snr}"):
            
            # load matching noise file
            noise_audio, noise_fs = torchaudio.load(os.path.join(args.noise, "audio", noise_file))
            if len(noise_audio.shape) > 1:
                noise_audio = noise_audio.mean(0)
            if noise_fs != args.sr:
                noise_audio = torchaudio.transforms.Resample(noise_fs, args.sr)(noise_audio)
            same_embedding = noise_audio[:args.sr]
            noise_audio = noise_audio[args.sr:]

            category = df.loc[df["filename"] == noise_file, "category"].values[0]
            alternative_noise_file = alternative_embeddings.loc[
                alternative_embeddings["category"] == category
            ].reset_index().loc[category_counter[category], "filename"]
            category_counter[category] += 1
            # print(alternative_noise_file)

            alternative_noise_audio, alternative_noise_fs = torchaudio.load(os.path.join(args.noise, "audio", alternative_noise_file))
            if len(alternative_noise_audio.shape) > 1:
                alternative_noise_audio = alternative_noise_audio.mean(0)
            if alternative_noise_fs != args.sr:
                alternative_noise_audio = torchaudio.transforms.Resample(alternative_noise_fs, args.sr)(alternative_noise_audio)
            alternative_embedding = alternative_noise_audio[:args.sr]

            speech_file = files[index]
            speech_audio, speech_fs = torchaudio.load(speech_file)
            if len(speech_audio.shape) > 1:
                speech_audio = speech_audio.mean(0)
            if speech_fs != args.sr:
                speech_audio = torchaudio.transforms.Resample(speech_fs, args.sr)(speech_audio)

            noisy_audio = segmental_snr_mixer(speech_audio.numpy(), noise_audio.numpy(), snr, sr=args.sr)

            filename = f"{index:03}.wav"
            af.write(
                os.path.join(args.dest, f"SNR_{snr}", "speech", filename),
                speech_audio.numpy(),
                args.sr
            )
            af.write(
                os.path.join(args.dest, f"SNR_{snr}", "noise", filename),
                noise_audio.numpy(),
                args.sr
            )
            af.write(
                os.path.join(args.dest, f"SNR_{snr}", "embedding-same", filename),
                same_embedding.numpy(),
                args.sr
            )
            af.write(
                os.path.join(args.dest, f"SNR_{snr}", "mixture", filename),
                noisy_audio,
                args.sr
            )
            af.write(
                os.path.join(args.dest, f"SNR_{snr}", "embedding-diff", filename),
                alternative_embedding.numpy(),
                args.sr
            )
            filenames["speech"].append(os.path.basename(speech_file))
            filenames["noise"].append(os.path.basename(noise_file))
            filenames["embedding-diff"].append(os.path.basename(alternative_noise_file))

        pd.DataFrame(filenames).to_csv(os.path.join(args.dest, f"SNR_{snr}", "filelist.csv"), index=False)


