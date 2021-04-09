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
for i in range(88):
    note_to_ind_map[note] = i*3
    note = transpose_pieces.transpose_note_standalone(note, 1)
## 88 * 3 = 264
note_to_ind_map['r'] = 264

ind_to_note_map = {ind : note for note, ind in note_to_ind_map.items()}

def zeros(hands=2):
    return np.zeros(270*hands, dtype=np.int16) 

def get_vec_for_hand(hand_notes):
    """Gets bag of notes vector for a single hand."""
    vec = zeros(hands=1)
    notes = hand_notes.strip().split(" ")
    for i in range(len(notes)):
        if notes[i].strip() == ".":
            continue
        dur = music_helpers.convert_to_dur_frac(notes[i])
        dur_n, dur_d = dur.numerator, dur.denominator
        note = music_helpers.get_note_note(notes[i])
        if note not in note_to_ind_map:
            # note doesn't exist on piano because of upward transposition; 
            # so transpose down
            note = note[1:]
        note_ind = note_to_ind_map[note]
        vec[note_ind] = 1 # if appears twice somehow, is still 1
        vec[note_ind + 1] = dur_n 
        vec[note_ind + 2] = dur_d
    return vec

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

def convert_hand_vec_to_kern(hand_vec):
    if np.sum(hand_vec) == 0:
        return "."
    hand_notes = []
    for i in range(0,len(hand_vec),3):
        if hand_vec[i] != 0:
            note = ind_to_note_map[i]
            duration = float(hand_vec[i+1]) /  hand_vec[i+2]
            duration_notated = music_helpers.convert_duration_to_notation(duration)
            hand_notes.append(duration_notated + note)
    hand_notes.sort(key=lambda x: music_helpers.convert_to_duration(x))
    return " ".join(hand_notes)

def convert_line_vec_to_kern(line_vec):
    """The inverse of convert_kern_line_to_vec(), possibly with ordering
    changes of the notes. """
    left_notes = convert_hand_vec_to_kern(line_vec[:len(zeros(hands=1))])
    right_notes = convert_hand_vec_to_kern(line_vec[len(zeros(hands=1)):])
    return left_notes+"\t"+right_notes
    
def vec_list_for_song(lines):
    """Converts a list of kern lines into a list of vectors. """
    vec_list = [zeros()]
    for line in lines:
        vec = convert_kern_line_to_vec(line)
        if vec is not None:
            vec_list.append(vec)
    vec_list.append(zeros())
    return vec_list

def song_from_vec_list(vecs):
    """Converts a list of vectors into a string representing a kern song."""
    song = ""
    for vec in vecs:
        if np.sum(vec) == 0:
            continue
        song += convert_line_vec_to_kern(vec) + "\n"
    return song        

def test():
    with open("./music_in_C/Beethoven, Ludwig van___Piano Sonata no. 10 in G major") as f:
        song = f.readlines()
    vecs = vec_list_for_song(song)
    converted_song = song_from_vec_list(vecs)
    with open("./test_song", "w") as f:
        f.write(converted_song)
    # vec = convert_kern_line_to_vec("4C 4d\t16r 2.aa 4ddddd")
    # print(convert_line_vec_to_kern(vec))
    # res1 = get_vec_for_hand("32r 8ee 16aa") 
    # for i in range(len(res1)):
    #     if res1[i] != 0:
    #         print(i, res1[i])
    # print(convert_hand_vec_to_kern(res1))

if __name__ == "__main__":
    test()
    # print(ind_to_note_map)
