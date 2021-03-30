import os
import re
from music_helpers import convert_to_duration, convert_duration_to_notation, get_duration_of_spine, get_left_note

def n_lines_of(n, l_note, r_note):
  lines = ""
  for i in range(n):
    lines += f"{l_note}\t{r_note}\n"
  return lines

def fix_negative_duration(l_duration, r_duration, add_line, 
                          rhythm_fix, time_step):
  """Modifies rhythm_fix to fix the previous action that caused the 
  remaining duration to go negative in one hand. This negative duration
  can be caused by a note in the right hand exceeding the duration
  of the remaining duration in the left hand. We fix this by shortening
  the note in the right hand to be exactly equal to the remining 
  left hand duration."""
  l_dur_this_line = convert_to_duration(get_duration_of_spine(add_line[0]))
  r_dur_this_line = convert_to_duration(get_duration_of_spine(add_line[1]))
  if l_duration < 0:
    n,d = (l_duration - (l_dur_this_line - time_step)).as_integer_ratio()
    r_note_leftmost = get_left_note(add_line[1])
    r_line_remainder = " ".join(add_line[1].split(" ")[1:])
    if n == 1:
      rhythm_fix += f".\t{d}{r_note_leftmost} {r_line_remainder}\n"
    elif n >= 2:
      rhythm_fix += f".\t[{d}{r_note_leftmost} {r_line_remainder}\n"
      rhythm_fix += n_lines_of(n-2, ".", f"[{d}{r_note_leftmost}]")
      rhythm_fix += f".\t{d}{r_note_leftmost}]\n"
  elif r_duration < 0:
    n,d = (r_duration - (r_dur_this_line - time_step)).as_integer_ratio()
    l_note_leftmost = get_left_note(add_line[0])
    l_line_remainder = " ".join(add_line[1].split(" ")[1:])
    if n == 1:
      rhythm_fix += f"{d}{l_note_leftmost} {l_line_remainder}\t.\n"
    elif n >= 2:
      rhythm_fix += f"[{d}{l_note_leftmost} {l_line_remainder}\t.\n"
      rhythm_fix += n_lines_of(n-2,f"[{d}{l_note_leftmost}]",".")
      rhythm_fix += f"{d}{l_note_leftmost}]\t.\n"
  return rhythm_fix

def get_next_line(l_duration, l_note, r_duration, r_note, rhythm_fix):
  """Given the remaining duration for the left hand and right hand, along
  with the notes that are written for this time slice, returns a line that is 
  similar to 'l_note<tab>r_note', but that satisfies the constraints imposed by 
  l_duration and r_duration. 
  
  It's possible that these constraints will be fulfilled in part by a 
  modification to rhythm_fix. 
  
  Assumes l_duration == 0 OR r_duration == 0"""
  add_line = []
  if l_duration == 0 and r_duration == 0:
    if l_note == ".":
      ## the left hand MUST play a note if it has no remaining duration
      r_dur_this_line = get_duration_of_spine(r_note)
      add_line = [f"{r_dur_this_line}r",r_note] # add a rest in left ahnd
    elif r_note == ".":
      ## the right hand MUST play a note if it has no remaining duration
      l_dur_this_line = get_duration_of_spine(l_note)
      add_line = [l_note,f"{l_dur_this_line}r"] # add a rest in right hand
    else:
      ## this line is legal
      add_line = [l_note,r_note]
  elif l_duration > 0 and l_note != ".":
    ## the left hand is palying a note but it has duration remaining
    if r_note != ".":
      add_line = [".",r_note] # delete the note 
    else:
      ## deleting the note would delete the line 
      ## also, the right hand isn't playing a note, but it should be! 
      n,d = l_duration.as_integer_ratio()
      rhythm_fix += n_lines_of(n, ".", f"{d}r") # add rests in right hand to 
                                                # deplete left hand duration
      l_duration = 0
      add_line = [l_note, f"{get_duration_of_spine(l_note)}r"]
  elif r_duration > 0 and r_note != ".":
    ## right hand is playing note but it has remaining duration
    if l_note != ".":
      add_line = [l_note, "."] # delete extra note 
    else:
      ## deleting note would delete line, also, left hand is not playing
      ## a note, but it should be. 
      n,d = r_duration.as_integer_ratio()
      rhythm_fix += n_lines_of(n, f"{d}r", ".")
      r_duration = 0
      add_line = [f"{get_duration_of_spine(r_note)}r", r_note]
  else:
    ## this line is legal
    add_line = [l_note, r_note]
  return add_line, rhythm_fix, l_duration, r_duration
  

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

    l_note, r_note = line.split("\t")
    l_note, r_note = l_note.strip(), r_note.strip()
    if l_note == "." and r_note == ".":
      continue

    ## determine a legal next line
    add_line, rhythm_fix, l_duration, r_duration = get_next_line(l_duration, 
        l_note, r_duration, r_note, rhythm_fix) 
    ## advance one time step, given the duration of the line that was 
    ## just added 
    l_dur_this_line = convert_to_duration(get_duration_of_spine(add_line[0]))
    r_dur_this_line = convert_to_duration(get_duration_of_spine(add_line[1]))
    time_step = r_dur_this_line if l_dur_this_line == 0 \
                else (l_dur_this_line if r_dur_this_line == 0 \
                  else min(l_dur_this_line, r_dur_this_line))
    l_duration += l_dur_this_line - time_step
    r_duration += r_dur_this_line - time_step 

    if l_duration < 0 or r_duration < 0:
      rhythm_fix = fix_negative_duration(l_duration, r_duration, add_line, 
                                          rhythm_fix, time_step)
      l_duration = 0
      r_duration = 0
      x = rhythm_fix.split("\n")[-2]
    else:
      rhythm_fix += f"{add_line[0]}\t{add_line[1]}\n"
  ## clean up end-of-piece rhythm
  if l_duration > 0:
    n,d = l_duration.as_integer_ratio()
    rhythm_fix += n_lines_of(n, ".", f"{d}r")
  elif r_duration > 0:
    n,d = r_duration.as_integer_ratio()
    rhythm_fix += n_lines_of(n, f"{d}r", ".")
  return rhythm_fix

def add_barlines(kern_string):
  """Insert a barline every 4 quarter notes in the right hand"""
  out = ""
  r_dur = 0
  for line in kern_string.split("\n"):
    if line.strip() == "":
      continue
    r_notes = line.split("\t")[1]
    r_dur += convert_to_duration(get_duration_of_spine(r_notes))
    out += line+"\n"
    if r_dur >= 1:
      out += "=\t=\n"
      r_dur = 0
  return out

def convert_to_good_kern(kern_string):
  pass


if __name__ == "__main__":
  # kern = """
  # 1A\t4d
  # .\t1a
  # 4a\t.
  # """
  # print(fix_rhythm(kern))

  with open("./music_in_C/Beethoven, Ludwig van___Piano Sonata no. 2 in A major") as f:
    fixed = fix_rhythm(f.read())
    lines = add_barlines(fixed)
    print(lines)