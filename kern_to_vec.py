import numpy as np
import re

import transpose_pieces

### 12 notes per octave, 
### the standard 88 keys per hand (+1 for rest)
### might to combine octaves to deal with sparsity

# and maybe a few cells per note for duration? We could just round all
# durations to be either 1/1, 1/2, 1/4, 1/8, 1/16. I don't mind the loss
# in rhythmic complexity.

### 94*3 = <300 cells per hand 
### 600 cells total


# make these global maps just once to save on computation
# AAAA to ccccc - the 88 piano keys
note = 'AAAA'
note_to_ind_map = {}
duration_to_ind_map = {}
for i in range(88):
    note_to_ind_map[note] = i
    note = transpose_pieces.transpose_note_standalone(note, 1)
for i in range(6):
    duration_to_ind_map[str(2**i)] = i + 88


def get_vec_for_hand(hand_notes):
    """Gets bag of notes vector for a single hand."""
    vec = np.zeros(300)
    note_pattern = r'(\d+)\.*([A-Ga-g]+#?)' # treats dotted quarters as quarters, etc
    notes = re.findall(note_pattern, hand_notes) # this is currently ignoring rests! TODO fix
    # print(f'found {len(notes)} matches')
    for i in range(len(notes)):
        duration, note = notes[i]
        note_ind = note_to_ind_map.get(note) + i*100
        duration_ind = duration_to_ind_map.get(duration, duration_to_ind_map["4"]) + i*100
                                                    # default to quarter note I guess
        if note_ind is not None:
            vec[note_ind] = 1 # if appears twice somehow, is still 1
            vec[duration_ind] = 1
    return vec

def test():
    # With current logic, the RNN will be able to differentiate between the
    # respective durations of the different 3 notes in the hand. idk if
    # that's necessary
    res1 = get_vec_for_hand("4ryy	8aa^	16eeLL") 
    for i in range(len(res1)):
        if res1[i] != 0:
            print(i)
    # I'm too lazy to work out what this actually should be, but 
    # looks plausible - 2 notes found
# test()

def convert_kern_line_to_vec(line):
    # @caroline I'm currently trying my own thing as far as notating duration,
    # but let me know what you think
    """ Bag of notes, where each note position is 1 if the note is present
    and 0 otherwise. to the right of each note will be two vector entries,
    one each for the numerator and denominator of the duration of the note.
    We ignore [ and ]. If a note is somehow repeated in a hand in a line, 
    break duration ties arbitrarily. 
    
    Each kern line is essentially two concatenated bag of notes vectors, one
    for the left hand and one for the right. If a hand has a `.` instead of
    notes, all the values for the hand will be 0. Lines that somehow have 
    two `.` should be skipped. Lines that have notes in one hand but a space
    in the other hand should be skipped. Lines that are just white space 
    should be skipped. 

    Notes range from AAAA to ccccc. Some notes might be higher because of the
    transposing up; those will be ignored.
    
    Every song starts and ends with the zero vector.  """
    if line.strip() == "" \
        or len(list(filter(lambda x: x.strip() != "", line.split("\t")))) != 2: 
        return None
    l_notes, r_notes = line.split("\t")
    if l_notes.strip() == "." and r_notes.strip() == ".":
        return None

    return get_vec_for_hand(l_notes).extend(get_vec_for_hand(r_notes))


def convert_vec_to_kern(line_vec):
    """The inverse of convert_kern_line_to_vec()"""
    pass


def vec_list_for_song(lines):
    # should add zero vec to start/end of song, as mentioned in
    # `convert_kern_line_to_vec`
    pass