# -*- coding: utf-8 -*-
# @Time    : 2021/6/21 上午10:28
# @Author  : seenli
# @File    : preprocess16_toys.py

"""
实验preprocess16
"""

import math, pickle, os, glob
import numpy as np
import sys
from utils import *
from utils.dsp import *
from utils.display import *
import os
import librosa
import soundfile as sf

def load_wav(filename, encode=True) :
    x = librosa.load(filename, sr=sample_rate)[0]
    if encode == True : x = encode_16bits(x)
    return x

def convert_file(path) :
    wav = load_wav(path, encode=False)
    mel = melspectrogram(wav)
    quant = wav * (2**15 - 0.5) - 0.5
    return mel.astype(np.float32), quant.astype(np.int16)

if __name__ == '__main__':
    pass
    file_name = "../temp/LJSpeech-1.1/wavs/LJ001-0001.wav"
    m, x = convert_file(file_name)
    plot_spec(m)
    plot(x)
    bits = 16
    x2 = 2 * x / (2 ** bits - 1) - 1  # 归一化
    plot(x2)
    sf.write('test_quant.wav', x2, samplerate=sample_rate)
    print('end...')

