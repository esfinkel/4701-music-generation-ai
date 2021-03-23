import os

## order of sharps: f c g d a e b
## ex: a key with 4 sharps is the key sharp_keys[4] = e
sharp_keys = ["c", "g", "d", "a", "e", "b", "f#", "c#"]

## order of flats: b e a d g c f
## ex: a key with 4 flats is the key flat_keys[4] = g#,
##     or more correctly, Ab, but I want to represent notes consistently 
flat_keys = ["c", "f", "a#", "d#", "g#", "c#", "f#", "b"]

## standardized note representations 
cannonical_notes = ["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", 
                    "a", "a#", "b"]


def get_note_register(note):
  """ Returns the register of a note, where register 0 is 
  c...b, register -1 is C...B, register 1 is cc...bb, etc. 
  Also returns the index at which we start seeing # or
  - symbols in the note instead of pitch indicators. """
  ## indicates whether we should climb or drop octaves upon 
  ## encountering repeats
  pitch_up = (note[0] == note[0].lower())
  ## register 0 is the range of notes c...b, -1 is C...B
  register = 0 if pitch_up else -1 
  base_note = note[0]
  pitch_end = 0
  for i in range(1,len(note)):
    if note[i] != base_note:
      pitch_end = i 
      break
    register += 1 if pitch_up else -1
  return register, pitch_end

def get_cannonical_note_idx(note):
  """Returns the index of the cannonical note to `note`,
  in register 0. Also returns the true register of `note`. """
  register, pitch_end = get_note_register(note)
  idx = cannonical_notes.index(note[0].lower())
  for i in range(pitch_end, len(note)):
    assert note[i] == "#" or note[i] == "-", note[i]
    if note[i] == "#":
      if idx + 1 >= len(cannonical_notes):
        register += 1
      idx = (idx + 1) % len(cannonical_notes)
    else:
      if idx - 1 < 0:
        register -= 1
      idx = (idx - 1) % len(cannonical_notes)
  return idx, register 

def to_cannonical_note(note):
  """Takes as input a string of the form [a-gA-G] followed by any number 
  of "#" or "-" symbols, and returns the cannonical representation
  of this note. Note that the first letter may be repeated any number
  of times. 
  Example: g-   =>  f#
  Example: b#   =>  cc
  Example: AA#- =>  AA
  Example: c-   =>  B  """
  if "#" not in note and "-" not in note:
    return note
  idx, register = get_cannonical_note_idx(note)
  cannonical_base_note = cannonical_notes[idx][0]
  if register > 0:
    ## base note c in register 2  => ccc
    cannonical_base_note = cannonical_base_note * (register+1)
  elif register < 0:
    ## base note c in register -2 => CC 
    cannonical_base_note = cannonical_base_note.upper() * (register * -1)
  return cannonical_base_note + cannonical_notes[idx][1:]



## loop thorugh the files in raw1/ and find the key each pieces is in. 
## then load the cleaned up version of that file from processed_music/
## and create a version of the piece that's transposed into the key of C
## (initially; the piece may change keys throughout). 
## store the transposed piece in music_in_C/
# if __name__ == "__main__":
#   for filename in os.listdir("./raw1/"):
#     try:
#       file_string = ""
#       with open(f"./raw1/{filename}", "r") as f:
#         file_string = clean_file(f)
#       with open(f"./processed_music/{filename}", "w") as f:
#         f.write(file_string)
#     except AssertionError as err:
#       print(f"Assertion error in reading {filename}: {err}")
#     except:
#       print(f"Unexpected error in reading {filename}")