import numpy as np
import re
import music_helpers

import transpose_pieces

"""
NOTE: all of these functions assume the input is CLEANED music, so no
extraneous characters, and no notes that aren't in their cannonical
representation. 
"""

# make these global maps just once to save on computation
# AAAA to ccccc - the 88 piano keys
note = 'AAAA'
note_to_ind_map = {}
duration_to_ind_map = {}
for i in range(88):
    note_to_ind_map[note] = i*3
    note = transpose_pieces.transpose_note_standalone(note, 1)
## 88 * 3 = 264, then 265 and 266 are taken, so rest occurs 267-269
note_to_ind_map['r'] = 267

def zeros(hands=2):
    return np.zeros(270*hands, dtype=np.int16) 

def get_vec_for_hand(hand_notes):
    """Gets bag of notes vector for a single hand."""
    vec = zeros(hands=1)
    notes = hand_notes.strip().split(" ")
    for i in range(len(notes)):
        print(notes[i])
        if notes[i].strip() == ".":
            continue
        dur = music_helpers.convert_to_dur_frac(notes[i])
        dur_n, dur_d = dur.numerator, dur.denominator
        print(dur_n, dur_d)
        note = music_helpers.get_note_note(notes[i])
        print(note)
        if note not in note_to_ind_map:
            # note doesn't exist on piano because of upward transposition; 
            # so transpose down
            note = note[1:]
        note_ind = note_to_ind_map.get(note)
        vec[note_ind] = 1 # if appears twice somehow, is still 1
        vec[note_ind + 1] = dur_n 
        vec[note_ind + 2] = dur_d
    return vec

def test():
    # With current logic, the RNN will be able to differentiate between the
    # respective durations of the different 3 notes in the hand. idk if
    # that's necessary
    res1 = get_vec_for_hand("4r 8aa 16ee") 
    for i in range(len(res1)):
        if res1[i] != 0:
            print(i, res1[i])
    # I'm too lazy to work out what this actually should be, but 
    # looks plausible - 2 notes found
# test()

def convert_kern_line_to_vec(line):
    """ Bag of notes, where each note position is 1 if the note is present
    and 0 otherwise. to the right of each note will be two entries,
    one each for the numerator and denominator of the duration of the note.
    We ignore [ and ]. If a note is somehow repeated in a hand in a line, 
    break duration ties arbitrarily. 
    
    Each kern line is essentially two concatenated bag of notes vectors, one
    for the left hand and one for the right. If a hand has a `.` instead of
    notes, all the values for the hand will be 0. Lines that somehow have 
    two `.` should be skipped. Lines that have notes in one hand but a space
    in the other hand should be skipped. Lines that are just white space 
    should be skipped. 

    Notes include only valid piano notes. Notes that are too high will be 
    transposed down.
    
    Every song starts and ends with the zero vector.  """
    if line.strip() == "" \
        or len(list(filter(lambda x: x.strip() != "", line.split("\t")))) != 2: 
        return None
    l_notes, r_notes = line.split("\t")
    if l_notes.strip() == "." and r_notes.strip() == ".":
        return None

    return np.concatenate((get_vec_for_hand(l_notes), get_vec_for_hand(r_notes)))

def convert_vec_to_kern(line_vec):
    """The inverse of convert_kern_line_to_vec()"""
    pass


def vec_list_for_song(lines):
    # should add zero vec to start/end of song, as mentioned in
    # `convert_kern_line_to_vec`
    vec_list = [zeros()]
    for line in lines:
        # TODO maybe normalize before appending
        vec_list.append(convert_kern_line_to_vec(line))
    vec_list.append(zeros())
    return vec_list
