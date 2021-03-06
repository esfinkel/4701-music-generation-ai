import numpy as np
import re
import music_helpers
import math
import random
import torch
import fix_kern

import transpose_pieces

"""
for each hand: 
  [0...12]  a probability distribution over c c# d ... b r
  [13...25] the same
  [26...38] the same
  [39...47] a probability distribution over durations 
            1/16 1/12 1/8 1/4 1/3 3/8 1/2 3/4 1 
  [48...51] a probability distribution over playing 0, 1, 2, or 3 notes 
            in this hand
"""

note = 'c'
note_to_ind_map = {}
for i in range(12):
    note_to_ind_map[note] = i
    note = transpose_pieces.transpose_note_standalone(note, 1)
note_to_ind_map['r'] = 12

ind_to_note_map = {ind : note for note, ind in note_to_ind_map.items()}

common_rhythms = [1/16, 1/12, 1/8, 1/4, 1/3, 3/8, 1/2, 3/4, 1]

def zeros(hands=2):
    return np.zeros(52*hands, dtype=np.int16) 

def ones(hands=2):
    return np.ones(52*hands, dtype=np.int16) 

def get_closest_rhythm_ind(dur):
    for i in range(len(common_rhythms)):
        if dur <= common_rhythms[i]:
            return i
    return len(common_rhythms) - 1

def get_vec_for_hand(hand_notes):
    """Gets bag of notes vector for a single hand. """
    vec = zeros(hands=1)
    if hand_notes == ".":
        vec[48] = 1
        return vec
    notes = sorted(hand_notes.strip().split(" "), key=lambda x: music_helpers.get_gen_note(x))
    for i in range(min(len(notes),3)):
        assert notes[i].strip() != ""
        if notes[i].strip() == ".":
            continue
        dur = music_helpers.convert_to_duration(notes[i])
        note = music_helpers.get_gen_note(notes[i]) 
        note_ind = note_to_ind_map[note] + i*13
        vec[note_ind] = 1 
        vec[get_closest_rhythm_ind(dur) + 39] = 1
    vec[min(3, len(notes)) + 48] = 1 
    return vec

def convert_kern_line_to_vec(line):
    """ Bag of notes, constrained to 1 octave per hand. """
    if line.strip() == "" \
        or len(list(filter(lambda x: x.strip() != "", line.split("\t")))) != 2: 
        return None
    l_notes, r_notes = line.split("\t")
    if l_notes.strip() == "." and r_notes.strip() == ".":
        return None

    return np.concatenate((get_vec_for_hand(l_notes), get_vec_for_hand(r_notes)))

def to_good_prob_dist(vec):
    ## inverse of softmax is log(S_i) + c
    ## since our vectors are already in log space, we just need
    ## to shift probabilities to 0
    return vec - np.min(vec)
    # return math.e**vec

def convert_hand_vec_to_kern(hand_vec, hand):
    """Convert vector for one hand to kern.
    
    Bias the probability distribution for num_notes and choose num notes. 
    Then take probability distribution for notes, and choose them.
    """
    # x = to_good_prob_dist(hand_vec[48:])
    num_notes = random.choices(np.argsort(-to_good_prob_dist(hand_vec[48:]))[:2], -np.sort(-to_good_prob_dist(hand_vec[48:]))[:2])[0]
    # num_notes = 1
    if num_notes == 0:
        return "."
    duration_ind = random.choices(np.argsort(-to_good_prob_dist(hand_vec[39:48]))[:4], -np.sort(-to_good_prob_dist(hand_vec[39:48]))[:4])[0]
    duration = music_helpers.convert_duration_to_notation(common_rhythms[duration_ind])
    # duration = "4"
    notes = []
    for i in range(num_notes):
        note_ind = random.choices(np.argsort(-to_good_prob_dist(hand_vec[i*13:i*13+13]))[:4], -np.sort(-to_good_prob_dist(hand_vec[i*13:i*13+13]))[:4])[0]
        note = ind_to_note_map[note_ind]
        if hand=='L' and note != 'r':
            notes.append(duration+note.upper())
        else:
            notes.append(duration+note)

    notes.sort(key=lambda x: music_helpers.convert_to_duration(x))
    return " ".join(notes)

def convert_line_vec_to_kern(line_vec):
    """The inverse of convert_kern_line_to_vec(), possibly with ordering
    changes of the notes. """
    left_notes = convert_hand_vec_to_kern(line_vec[:len(zeros(hands=1))], 'L')
    right_notes = convert_hand_vec_to_kern(line_vec[len(zeros(hands=1)):], 'R')
    if left_notes == "." and right_notes == ".":
        return None
    return left_notes+"\t"+right_notes
    
def vec_list_for_song(lines):
    """Converts a list of kern lines into a list of vectors. """
    vec_list = [ones()]
    for line in lines:
        vec = convert_kern_line_to_vec(line)
        if vec is not None:
            vec_list.append(vec)
    vec_list.append(ones())
    return vec_list

def song_from_vec_list(vecs):
    """Converts a list of vectors into a string representing a kern song."""
    song = ""
    for vec in vecs:
        left = vec[:len(zeros(hands=1))]
        right = vec[len(zeros(hands=1)):]
        song_kern = convert_line_vec_to_kern(vec)
        if song_kern is not None:
            song += song_kern + "\n"
    return song        

def test():
    with open("./music_in_C/Beethoven, Ludwig van___Piano Sonata no. 10 in G major") as f:
        song = f.readlines()
    vecs = vec_list_for_song(song)
    vec2 = []
    for vec in vecs:
        vec = torch.from_numpy(vec).float()
        vec = torch.cat((
            torch.log_softmax(vec[:13], 0), #pitch 1 L
            torch.log_softmax(vec[13:26], 0), #pitch 2 L
            torch.log_softmax(vec[26:39], 0), #pitch 3 L
            torch.log_softmax(vec[39:48], 0), #duration L
            torch.log_softmax(vec[48:52], 0), #num notes L 
            torch.log_softmax(vec[52:13+52], 0), #pitch 1 R
            torch.log_softmax(vec[13+52:26+52], 0), #pitch 2 R
            torch.log_softmax(vec[26+52:39+52], 0), #pitch 3 R
            torch.log_softmax(vec[39+52:48+52], 0), #duration R
            torch.log_softmax(vec[48+52:52+52], 0) #num notes R
        ))
        vec2.append(vec.detach().numpy())
    converted_song = song_from_vec_list(vec2)
    with open("./test_song", "w") as f:
        f.write(fix_kern.convert_to_good_kern(converted_song))

if __name__ == "__main__":
    test()
    # print(ind_to_note_map)
