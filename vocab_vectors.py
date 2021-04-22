import numpy as np
import re
import music_helpers
import math
import random
import torch
import fix_kern
import os

import transpose_pieces

def get_next_beat(lines, i):
    """returns a concatenation of the lines after i that form
    a multiple of 1 quarter note."""
    next_line = ""
    r_dur = 0
    l_dur = 0
    j = i
    for line in lines[i:]:
        j += 1
        if line.strip() == "":
            continue
        if len(list(filter(lambda x : x.strip() != "", line.split("\t")))) < 2:
            continue
        l_notes, r_notes = line.split("\t")[0], line.split("\t")[1]
        r_dur += music_helpers.convert_to_duration(music_helpers.get_duration_of_spine(r_notes))
        l_dur += music_helpers.convert_to_duration(music_helpers.get_duration_of_spine(l_notes))
        next_line += (line.strip()+"\n").replace("[","").replace("]","")
        if r_dur%0.25<=0.01 and l_dur%0.25<=0.01:
            break
        if r_dur > 2 or l_dur > 2:
            return get_next_beat(lines, i+1)
    return next_line, j
    

def gather_vocab():
    """ Gather all beat occurrences in the training data.  """
    vocab = set()
    for filename in os.listdir("./music_in_C_training"):
        if ".DS_Store" in filename:
            continue
        with open(f"./music_in_C_training/{filename}", "r") as f:
            filetext = f.readlines()
            filetext = list(filter(lambda t: t.strip() != "", filetext))
            i = 0
            while i < len(filetext):
                next_line, i = get_next_beat(filetext, i)
                vocab.add(next_line)
    for filename in os.listdir("./music_in_C_test"):
        if ".DS_Store" in filename:
            continue
        with open(f"./music_in_C_test/{filename}", "r") as f:
            filetext = f.readlines()
            filetext = list(filter(lambda t: t.strip() != "", filetext))
            i = 0
            while i < len(filetext):
                next_line, i = get_next_beat(filetext, i)
                vocab.add(next_line)
    return vocab

vocab = gather_vocab()
vocab_to_ind = {token : i for i, token in enumerate(list(vocab))}
ind_to_vocab = {i : token for token, i in vocab_to_ind.items()}


def zeros():
    return np.zeros(len(ind_to_vocab), dtype=np.int16) 

def ones():
    return np.ones(len(ind_to_vocab), dtype=np.int16) 

def to_good_prob_dist(vec):
    ## inverse of softmax is log(S_i) + c
    ## since our vectors are already in log space, we just need
    ## to shift probabilities to 0
    return vec - np.min(vec)

def convert_line_vec_to_kern(line_vec):
    """The inverse of convert_kern_line_to_vec(), possibly with ordering
    changes of the notes. """
    ind = random.choices(np.argsort(-to_good_prob_dist(line_vec))[:50], -np.sort(-to_good_prob_dist(line_vec))[:50])[0]
    return ind_to_vocab[ind]

def convert_kern_line_to_vec(line):
    vec = np.zeros(len(ind_to_vocab))
    vec[vocab_to_ind[line]] = 1
    return vec
  
def vec_list_for_song(lines):
    """Converts a list of kern lines into a list of vectors. """
    vec_list = [ones()]
    lines = list(filter(lambda t: t.strip() != "", lines))
    i = 0
    while i < len(lines):
        next_line, i = get_next_beat(lines, i)
        vec = np.zeros(len(ind_to_vocab))
        vec[vocab_to_ind[next_line]] = 1
        vec_list.append(vec)
    vec_list.append(ones())
    return vec_list

def song_from_vec_list(vecs):
    """Converts a list of vectors into a string representing a kern song."""
    song = ""
    for vec in vecs:
        song_kern = convert_line_vec_to_kern(vec)
        song += song_kern
    return song        

def test():
    with open("./music_in_C_training/Chopin, Frederic___Mazurka in C-sharp Minor, Op. 63, No. 3") as f:
        song = f.readlines()
    vecs = vec_list_for_song(song)
    converted_song = song_from_vec_list(vecs)
    with open("./test_song", "w") as f:
        f.write(converted_song)

if __name__ == "__main__":
    test()
    # print(ind_to_note_map)
