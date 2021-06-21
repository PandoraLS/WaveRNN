
# coding: utf-8

# ## Alternative Model (Preprocessing)
# You need to run this before you run notebook 4b.
# 
# The wavs in your dataset will be converted to 9bit linear and 80-band mels.

#import matplotlib.pyplot as plt
import math, pickle, os, glob
import numpy as np
import sys
from utils import *
from utils.dsp import *
import os

bits = 16
notebook_name = 'nb4'

# Point SEG_PATH to a folder containing your training wavs 
# Doesn't matter if it's LJspeech, CMU Arctic etc. it should work fine
SEG_PATH = sys.argv[1]
DATA_PATH = sys.argv[2]
os.makedirs(DATA_PATH, exist_ok=True)

def get_files(path, extension='.wav') :
    filenames = []
    for filename in glob.iglob(f'{path}/**/*{extension}', recursive=True):
        filenames += [filename]
    return filenames

wav_files = get_files(SEG_PATH)

def convert_file(path) :
    wav = load_wav(path, encode=False)
    mel = melspectrogram(wav)
    quant = wav * (2**15 - 0.5) - 0.5 # 量化后的wav
    return mel.astype(np.float32), quant.astype(np.int16)

#m, x = convert_file(wav_files[1])
#plot_spec(m)
#plot(x)
#x = 2 * x / (2**bits - 1) - 1 # 归一化
#librosa.output.write_wav(DATA_PATH + 'test_quant.wav', x, sr=sample_rate)

QUANT_PATH = DATA_PATH + '/quant/'
MEL_PATH = DATA_PATH + '/mel/'
os.makedirs(QUANT_PATH, exist_ok=True)
os.makedirs(MEL_PATH, exist_ok=True)

#wav_files[0].split('/')[-1][:-4]

print("")
# This will take a while depending on size of dataset
dataset_ids = []
for i, path in enumerate(wav_files) :
    id = path.split('/')[-1][:-4]
    dataset_ids += [id]
    m, x = convert_file(path)
    np.save(f'{MEL_PATH}{id}.npy', m)
    np.save(f'{QUANT_PATH}{id}.npy', x)
    print("\r%i/%i", (i + 1, len(wav_files)), end='')

with open(DATA_PATH + '/dataset_ids.pkl', 'wb') as f:
    pickle.dump(dataset_ids, f)
