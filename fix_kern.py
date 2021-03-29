import os
import re
from music_helpers import convert_to_duration, convert_duration_to_notation, get_duration_of_spine, get_left_note


def fix_rhythm(kern_string):
  """Given a string representing a **kern file, scans through all the lines
  and ensures that rhythmic values are legal throughout. For example, a
  spine cannot have a new note or rest if the duration from its previous
  note has not run out. Or, a spine MUST have a new note or rest if the
  duration from its previous note has run out. 
  
  Returns a kern_string with potentially modified notes and rhythms to make
  the sequence of notes legal."""
  l_duration = 0.0
  r_duration = 0.0
  rhythm_fix = ""
  for line in kern_string.split("\n"):
    if line.strip() == "":
      continue
    ## INVARIANT: one of l_duration or r_duration is zero
    l_note, r_note = line.split("\t")
    l_note, r_note = l_note.strip(), r_note.strip()
    add_line = ""
    if l_note == "." and r_note == ".":
      continue
    if l_duration == 0 and r_duration == 0:
      if l_note == ".":
        r_dur_this_line = get_duration_of_spine(r_note)
        add_line = [f"{r_dur_this_line}r",r_note]
      elif r_note == ".":
        l_dur_this_line = get_duration_of_spine(l_note)
        add_line = [l_note,f"{l_dur_this_line}r"]
    elif l_duration > 0 and l_note != ".":
      if r_note != ".":
        add_line = [".",r_note]
      else:
        n,d = l_duration.as_integer_ratio()
        rhythm_fix += n_lines_of(n, ".", f"{d}r")
        add_line = [l_note, f"{get_duration_of_spine(l_note)}r"]
    elif r_duration > 0 and r_note != ".":
      if l_note != ".":
        add_line = [l_note, "."]
      else:
        n,d = r_duration.as_integer_ratio()
        rhythm_fix += n_lines_of(n, f"{d}r", ".")
        add_line = [f"{get_duration_of_spine(r_note)}r", r_note]
    else:
      add_line = [l_note, r_note]
    l_dur_this_line = convert_to_duration(get_duration_of_spine(add_line[0]))
    r_dur_this_line = convert_to_duration(get_duration_of_spine(add_line[1]))
    time_step = min(l_dur_this_line, r_dur_this_line)
    l_duration += l_dur_this_line - time_step
    r_duration += r_dur_this_line - time_step 
    if l_duration < 0:
      n,d = (l_duration - (l_dur_this_line - time_step)).as_integer_ratio()
      r_note_leftmost = get_left_note(add_line[1])
      r_line_remainder = " ".join(add_line[1].split(" ")[1:])
      if n == 1:
        rhythm_fix += f".\t{d}{r_note_leftmost} {r_line_remainder}"
      elif n >= 2:
        rhythm_fix += f".\t[{d}{r_note_leftmost} {r_line_remainder}"
        rhythm_fix += n_lines_of(n-2, ".", f"[{d}{r_note_leftmost}]")
        rhythm_fix += f".\t{d}{r_note_leftmost}]"



      

def convert_to_good_kern(kern_string):
  pass


if __name__ == "__main__":
  pass
  # for filename in os.listdir("./raw1/"):
  #   try:
  #     file_string = ""
  #     with open(f"./raw1/{filename}", "r") as f:
  #       file_string = clean_file(f)
  #     with open(f"./processed_music/{filename}", "w") as f:
  #       f.write(file_string)
  #   except AssertionError as err:
  #     print(f"Assertion error in reading {filename}: {err}")
  #   except:
  #     print(f"Unexpected error in reading {filename}")