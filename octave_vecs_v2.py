import numpy as np
import re
import music_helpers
import math

import transpose_pieces

"""
hand vectors now only contain 12 notes c-b (R) or C-B (L), a rest marker for
each hand, and two cells indicating duration for the whole chord for
each hand. The last SIX cells for each hand represent a one-hot encoding
of the number of notes in the chord. 
"""

# make these global maps just once to save on computation
# AAAA to ccccc - the 88 piano keys
note = 'c'
note_to_ind_map = {}
for i in range(12):
    note_to_ind_map[note] = i
    note = transpose_pieces.transpose_note_standalone(note, 1)
note_to_ind_map['r'] = 12
# print(note_to_ind_map)
## notes [c c# ... b r dur1 dur2 num_notes] (len = 16)

ind_to_note_map = {ind : note for note, ind in note_to_ind_map.items()}

# indices 15, 16, 17, 18, 19, 20 correspond to 0, 1, 2, 3, 4, 5 notes
def zeros(hands=2):
    return np.zeros(21*hands, dtype=np.int16) 

def ones(hands=2):
    return np.ones(21*hands, dtype=np.int16) 

def get_vec_for_hand(hand_notes):
    """Gets bag of notes vector for a single hand. """
    vec = zeros(hands=1)
    if hand_notes == ".":
        return vec
    notes = hand_notes.strip().split(" ")
    for i in range(len(notes)):
        if notes[i].strip() == ".":
            continue
        dur = music_helpers.convert_to_dur_frac(notes[i])
        dur_n, dur_d = dur.numerator, dur.denominator
        note = music_helpers.get_note_note(notes[i]) 
        if note[-1] == "#":
            note = note[0] + note[-1] ##ex: AAA# -> A#
        else:
            note = note[0] ### ex: CCCC -> C
        note_ind = note_to_ind_map[note.lower()]
        vec[note_ind] = 1 
        vec[13] = dur_n 
        vec[14] = dur_d
    vec[min(20, 15 + len(notes))] = 1 # if >5 notes somehow, use 5
    return vec

def convert_kern_line_to_vec(line):
    """ Bag of notes, constrianed to 1 octave per hand. """
    if line.strip() == "" \
        or len(list(filter(lambda x: x.strip() != "", line.split("\t")))) != 2: 
        return None
    l_notes, r_notes = line.split("\t")
    if l_notes.strip() == "." and r_notes.strip() == ".":
        return None

    return np.concatenate((get_vec_for_hand(l_notes), get_vec_for_hand(r_notes)))

def convert_hand_vec_to_kern(hand_vec, hand):
    if np.argmax(hand_vec[-6:]) == 0:
        return "."

    if round(hand_vec[13]) <= 0 or round(hand_vec[14])<= 0:
        hand_dur = "4"
    else:
        hand_dur = music_helpers.convert_duration_to_notation(
                        round(hand_vec[13]) / round(hand_vec[14]))

    notes = []
    num_notes = np.argmax(hand_vec[-6:])
    note_inds = np.argsort(-1*hand_vec[:13])[:num_notes]
    # print(note_inds)
    for i in note_inds:
        note = ind_to_note_map[i] if hand == 'R' else ind_to_note_map[i].upper()
        if 'R' in note:
            note = note.lower()
        notes.append(hand_dur+note)

    notes.sort(key=lambda x: music_helpers.convert_to_duration(x))
    return " ".join(notes)

def convert_line_vec_to_kern(line_vec):
    """The inverse of convert_kern_line_to_vec(), possibly with ordering
    changes of the notes. """
    # print(line_vec)
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
        if np.argmax(left[-6:]) == 0 and np.argmax(right[-6:]) == 0:
            continue
        song_kern = convert_line_vec_to_kern(vec)
        if song_kern is not None:
            song += song_kern + "\n"
    return song        

def test():
    with open("./music_in_C/Beethoven, Ludwig van___Piano Sonata no. 10 in G major") as f:
        song = f.readlines()
    vecs = vec_list_for_song(song)
    converted_song = song_from_vec_list(vecs)
    with open("./test_song", "w") as f:
        f.write(converted_song)

    # vec = convert_kern_line_to_vec("4C 4d\t16r 2.aa 4ddddd")
    # print(vec)
    # print(convert_line_vec_to_kern(vec))
    # res1 = get_vec_for_hand("32r 8ee 16aa") 
    # for i in range(len(res1)):
    #     if res1[i] != 0:
    #         print(i, res1[i])
    # print(convert_hand_vec_to_kern(res1))

if __name__ == "__main__":
    test()
    # print(ind_to_note_map)
