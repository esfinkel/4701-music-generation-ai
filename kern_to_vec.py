

### 12 notes per octave, 
### CCC CCC CC C c cc cc = 7*12 + 1 = 85 notes per hand (+1 for rest)
### 84*3 = 255 cells per hand 
### 510 cells total

def get_vec_for_hand(hand_notes):
  """Gets bag of notes vector for a single hand."""
  pass

def convert_kern_line_to_vec(line):
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

  Notes range from CCCC to bbb. Higher and lower notes will be clipped???
  
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
  pass