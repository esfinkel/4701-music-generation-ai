import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import io
import pickle

import time
from tqdm import tqdm

import probability_vectors as kern2vec
from rnn_generate_prob import RNN_No_FFNN
import fix_kern


def write_music(formatted):
    """Given well-formatted kern music, write file"""
    dir = 'generated_music_RNN'
    if not os.path.exists(dir):
        os.mkdir(dir)
    with open(f"./{dir}/{round(time.time())}.txt", "w") as file:
        file.write(formatted)

def generate(model, max_lines=150):
    """
    Generate and write random **kern song of length "max_lines",
    using the RNN "model".

    The param `updated_hidden_format` is True.
    """
    print("generating music.....")
    curr_line = torch.from_numpy(kern2vec.ones()).float() 
    hidden = model.init_hidden()
    song_string = ""
    i=0
    while i <= max_lines:
        i+= 1
        try:
            curr_line = curr_line.detach().numpy()
        except:
            pass
        curr_line, hidden = model(curr_line, hidden)
        line_str = kern2vec.convert_line_vec_to_kern(curr_line.detach().numpy())
        if line_str is not None:
            song_string += line_str + "\n"
            curr_line = kern2vec.convert_kern_line_to_vec(line_str)
        else:
            break
    print("Converting music to good **kern....")
    new_music_formatted = fix_kern.convert_to_good_kern(song_string)
    print("Writing music.....")
    write_music(new_music_formatted)


if __name__ == "__main__":
    model = None
    with open('rnn_models/log_prob_vecs&hidden_dim=50&learning_rate=0.25&epoch=40&dist=1.9347562789916992', "rb") as f:
        model = torch.load(f)
    generate(model)


