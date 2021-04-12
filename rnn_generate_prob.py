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

def generate(model, max_lines=150, updated_hidden_format=False):
    print("generating music.....")
    curr_line = torch.from_numpy(kern2vec.ones()).float() 
    if updated_hidden_format:
        hidden = model.init_hidden()
    generated = []
    song_string = ""
    while len(generated) <= max_lines:
        if updated_hidden_format:
            try:
                curr_line = curr_line.detach().numpy()
            except:
                pass
            curr_line, hidden = model(curr_line, hidden)
            # print(type(curr_line))
            # print(curr_line)
            # added
            line_str = kern2vec.song_from_vec_list([curr_line.detach().numpy()])
            if line_str != "":
                # print(line_str)
                song_string += line_str + "\n"
                curr_line = kern2vec.convert_kern_line_to_vec(line_str)
                # print(type(curr_line))
                # curr_line = torch.Tensor(curr_line)
                
        else:
            curr_line = model(curr_line.detach().numpy())
        generated.append(curr_line)
    print("Converting music to **kern....")
    # song_string = kern2vec.song_from_vec_list(generated)
    new_music_formatted = fix_kern.convert_to_good_kern(song_string)
    print("Writing music.....")
    write_music(new_music_formatted)


if __name__ == "__main__":
    model = None
    with open('rnn_models/Song_GD&octave_vecs&hidden_dim=30&learning_rate=0.5&1618104881&epoch=40&dist=12.511433601379395', "rb") as f:
        model = torch.load(f)
    generate(model, updated_hidden_format=True)


