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

import octave_vecs_v2 as kern2vec
from rnn_sgd_v2 import RNN_No_FFNN
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
    generated = []
    song_string = ""
    while len(generated) <= max_lines:
        try:
            curr_line = curr_line.detach().numpy()
        except:
            pass
        curr_line, hidden = model(curr_line, hidden)
        # song_from_vec_list randomly determines notes for the vector
        # curr_line based on its encoded probability distribution
        line_str = kern2vec.song_from_vec_list([curr_line.detach().numpy()])
        if line_str != "":
            # accumulate result into song_string
            song_string += line_str + "\n"
            # make curr_line a proper one-hot based on the result of 
            # the random determinations made in song_from_vec_list
            curr_line = kern2vec.convert_kern_line_to_vec(line_str)
        # `generated` is only maintained for ease of measuring song length
        generated.append(curr_line)
    print("Converting music to good **kern....")
    new_music_formatted = fix_kern.convert_to_good_kern(song_string)
    print("Writing music.....")
    write_music(new_music_formatted)


if __name__ == "__main__":
    model = None
    with open('rnn_models/Song_GD&octave_vecs&hidden_dim=30&learning_rate=0.5&1618104881&epoch=40&dist=12.511433601379395', "rb") as f:
        model = torch.load(f)
    generate(model)


