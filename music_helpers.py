import math
from fractions import Fraction

## standardized note representations 
CANNONICAL_NOTES = ["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", 
                    "a", "a#", "b"]

def frac_add(f1, f2):
  n1,d1 = f1.numerator, f1.denominator
  n2,d2 = f2.numerator, f2.denominator
  return Fraction(n1*d2+n2*d1, d1*d2)

def frac_sub(f1, f2):
  n1,d1 = f1.numerator, f1.denominator
  n2,d2 = f2.numerator, f2.denominator
  return Fraction(n1*d2-n2*d1, d1*d2)

def get_note_register(note):
  """ Returns the register of a note, where register 0 is 
  c...b, register -1 is C...B, register 1 is cc...bb, etc."""
  ## indicates whether we should climb or drop octaves upon 
  ## encountering repeats
  pitch_up = (note[0] == note[0].lower())
  register = 0 if pitch_up else -1 
  base_note = note[0]
  for i in range(1,len(note)):
    if note[i] != base_note:
      break
    register += 1 if pitch_up else -1
  return register

def get_cannonical_note_idx(note):
  """Returns the index of the cannonical note to `note`,
  in register 0. Also returns the true register of `note`. """
  if "#" not in note and "-" not in note:
    return CANNONICAL_NOTES.index(note.lower()[0]), len(note) - 1 if not note.isupper() else len(note) * -1

  register = get_note_register(note)
  idx = CANNONICAL_NOTES.index(note[0].lower())
  start = min(note.index("#") if "#" in note else len(note), 
              note.index("-") if "-" in note else len(note))
  for i in range(start, len(note)):
    assert note[i] == "#" or note[i] == "-", note[i]
    if note[i] == "#":
      if idx + 1 >= len(CANNONICAL_NOTES):
        register += 1
      idx = (idx + 1) % len(CANNONICAL_NOTES)
    else:
      if idx - 1 < 0:
        register -= 1
      idx = (idx - 1) % len(CANNONICAL_NOTES)
  return idx, register 

def to_cannonical_note(note):
  """Takes as input a string of the form [a-gA-G] followed by any number 
  of "#" or "-" symbols, and returns the cannonical representation
  of this note. Note that the first letter may be repeated any number
  of times. 
  Example: g-   =>  f#
  Example: b#   =>  cc
  Example: AA#- =>  AA
  Example: c-   =>  B  
  """
  if "#" not in note and "-" not in note:
    return note
  idx, register = get_cannonical_note_idx(note)
  cannonical_base_note = CANNONICAL_NOTES[idx][0]
  if register > 0:
    ## base note c in register 2  => ccc
    cannonical_base_note = cannonical_base_note * (register+1)
  elif register < 0:
    ## base note c in register -2 => CC 
    cannonical_base_note = cannonical_base_note.upper() * (register * -1)
  return cannonical_base_note + CANNONICAL_NOTES[idx][1:]

def get_duration_of_spine(spine):
  """Given **kern spine line, returns just the duration portion. 
  If there are multiple notes in the line, returns just the duration
  portion of the first note. """
  number = ""
  for c in spine:
    if c.isdigit():
      number += c
    if c == ".":
      number += c
    if c == " ":
      return number
  return number

def get_left_note(spine):
  """Given kern spine line, returns the note value of the leftmost note.
  Example: 4AA 2c  => AA"""
  if spine == ".":
    return "."
  left_note = spine.split(" ")[0]
  note = ""
  for c in left_note:
    if (not c.isdigit()) and c not in {".", "]", "[", "#", "-"}:
      note += c
  return note

def get_note_pitch(note):
  """Given a kern note, returns only the pitch portion."""
  pitch = ""
  for c in note:
    if (not c.isdigit()) and c not in {".", "]", "[", "#", "-"}:
      pitch += c
  return pitch

def get_note_note(note):
  """Given a kern note, returns only the [a-zA-Z][-#]* portion"""
  pitch = ""
  for c in note:
    if (not c.isdigit()) and c not in {".", "]", "["}:
      pitch += c
  return pitch

def get_index_of_pitch(line, pitch, start_char=None, end_char=None):
  """Given a spine line, returns the index at which the pitch 
  occurs, or -1 if the pitch doesn't occur. If start_char is not None,
  then the note containing the pitch must start with start_char. 
  Similarly with end_char. """
  notes = line.strip().split(" ")
  for i in range(len(notes)):
    note_pitch = get_note_pitch(notes[i])
    if pitch == note_pitch:
      if (start_char is None or notes[i][0] == start_char) \
          and (end_char is None or notes[i][-1] == end_char):
        return i
  return -1

def note_in_spine(pitch, spine_line):
  """Given a spine line, returns true if this pitch occurs in a note
  in the line."""
  notes = spine_line.strip().split(" ")
  for i in range(len(notes)):
    note_pitch = get_note_pitch(notes[i])
    if note_pitch == pitch:
      return True
  return False

def convert_to_duration(note):
  """Given a **kern note, returns the duration of this note. 
  Example: 4r -> 1/4
  Example: 2..AA -> (1/2) + (1/4) + (1/8) -> 0.875
  `.` has duration 0."""
  number=""
  dots = 0
  for c in note:
    if c.isdigit():
      number += c
    if c == ".":
      dots += 1
  if number == "":
    return 0
  number = int(number)
  duration = 1 / float(number)
  added_duration = float(duration)
  for i in range(dots):
    added_duration /= 2
    duration += added_duration
  return duration

def convert_to_dur_frac(note):
  """Given a **kern note, returns the duration of this note, as a fraction. \n
  Example: [4r -> 1/4 \n
  Example: 2..AA -> (1/2) + (1/4) + (1/8) -> 7/8 \n
  `.` has duration 0."""
  number=""
  dots = 0
  for c in note:
    if c.isdigit():
      number += c
    if c == ".":
      dots += 1
  if number == "":
    return 0
  number = int(number)
  duration = Fraction(1, number)
  added_duration = Fraction(duration)
  for i in range(dots):
    added_duration = Fraction(added_duration.numerator, 
                              added_duration.denominator * 2)
    duration = frac_add(added_duration, duration)
  return duration

def convert_duration_to_notation(duration):
  """Converts a float duration to a number followed by some number
  of `.`, such that 
  convert_to_duration(convert_duration_to_notation(duration)) = duration"""
  if duration == 0:
    return "."
  notation = ""
  ## ex: for duration 0.75, pow_2 = -1
  pow_2 = math.floor(math.log(duration, 2))
  ## ex: value is 1/2
  value = 2 ** pow_2
  ## ex: appearance is 2
  appearance = 1 / value
  if int(appearance) == appearance:
    notation += str(int(appearance))
  else:
    notation += str(appearance)
  ## ex: remainder is 1/4 
  remainder = duration - value
  while remainder > 0:
    value /= 2
    notation += "." ##ex: we add one . then remainder -= 1/4 
    remainder -= value
  return notation
  
