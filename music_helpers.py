## standardized note representations 
CANNONICAL_NOTES = ["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", 
                    "a", "a#", "b"]


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
