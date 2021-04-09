import os
import re
from music_helpers import convert_to_duration, get_duration_of_spine, get_left_note, get_note_pitch, get_index_of_pitch, note_in_spine, convert_to_dur_frac, frac_add, frac_sub
from fractions import Fraction

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
  l_dur_this_line = convert_to_dur_frac(get_duration_of_spine(add_line[0]))
  r_dur_this_line = convert_to_dur_frac(get_duration_of_spine(add_line[1]))
  if l_duration.numerator < 0:
    fixed_l_dur = frac_sub(l_duration, frac_sub(l_dur_this_line, time_step))
    n,d = fixed_l_dur.numerator, fixed_l_dur.denominator
    r_note_leftmost = get_left_note(add_line[1])
    r_line_remainder = " ".join(add_line[1].split(" ")[1:])
    if n == 1:
      rhythm_fix += f".\t{d}{r_note_leftmost} {r_line_remainder}\n"
    elif n >= 2:
      rhythm_fix += f".\t[{d}{r_note_leftmost} {r_line_remainder}\n"
      rhythm_fix += n_lines_of(n-2, ".", f"[{d}{r_note_leftmost}]")
      rhythm_fix += f".\t{d}{r_note_leftmost}]\n"
  elif r_duration.numerator < 0:
    fixed_r_dur = frac_sub(r_duration, frac_sub(r_dur_this_line, time_step))
    n,d = fixed_r_dur.numerator, fixed_r_dur.denominator
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
  if l_duration.numerator == 0 and r_duration.numerator == 0:
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
  elif l_duration.numerator > 0 and l_note != ".":
    ## the left hand is palying a note but it has duration remaining
    if r_note != ".":
      add_line = [".",r_note] # delete the note 
    else:
      ## deleting the note would delete the line 
      ## also, the right hand isn't playing a note, but it should be! 
      n,d = l_duration.numerator, l_duration.denominator
      rhythm_fix += n_lines_of(n, ".", f"{d}r") # add rests in right hand to 
                                                # deplete left hand duration
      l_duration = Fraction(0)
      add_line = [l_note, f"{get_duration_of_spine(l_note)}r"]
  elif r_duration.numerator > 0 and r_note != ".":
    ## right hand is playing note but it has remaining duration
    if l_note != ".":
      add_line = [l_note, "."] # delete extra note 
    else:
      ## deleting note would delete line, also, left hand is not playing
      ## a note, but it should be. 
      n,d = r_duration.numerator, r_duration.denominator
      rhythm_fix += n_lines_of(n, f"{d}r", ".")
      r_duration = Fraction(0)
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
  l_duration = Fraction(0)
  r_duration = Fraction(0)
  rhythm_fix = ""
  for line in kern_string.split("\n"):
    if line.strip() == "":
      continue

    try:
      l_note, r_note = line.split("\t")
    except:
      # l_note, r_note = line, "."
      continue
      # print("error with decoding: " + line)
      # return
    l_note, r_note = l_note.strip(), r_note.strip()
    if l_note == "." and r_note == ".":
      continue
    if l_note.strip() == "" or r_note.strip() == "":
      continue

    ## determine a legal next line
    add_line, rhythm_fix, l_duration, r_duration = get_next_line(l_duration, 
        l_note, r_duration, r_note, rhythm_fix) 
    ## advance one time step, given the duration of the line that was 
    ## just added 
    l_dur_this_line = convert_to_dur_frac(get_duration_of_spine(add_line[0]))
    r_dur_this_line = convert_to_dur_frac(get_duration_of_spine(add_line[1]))
    time_step = r_dur_this_line if l_dur_this_line == 0 \
                else (l_dur_this_line if r_dur_this_line == 0 \
                  else min(l_dur_this_line, r_dur_this_line))
    l_duration = frac_add(l_duration, frac_sub(l_dur_this_line, time_step))
    r_duration = frac_add(r_duration, frac_sub(r_dur_this_line,time_step))

    if l_duration.numerator < 0 or r_duration.numerator < 0:
      rhythm_fix = fix_negative_duration(l_duration, r_duration, add_line, 
                                          rhythm_fix, time_step)
      l_duration = Fraction(0)
      r_duration = Fraction(0)
    else:
      rhythm_fix += f"{add_line[0]}\t{add_line[1]}\n"
  ## clean up end-of-piece rhythm
  if l_duration.numerator > 0:
    n,d = l_duration.numerator, l_duration.denominator
    rhythm_fix += n_lines_of(n, ".", f"{d}r")
  elif r_duration > 0:
    n,d = r_duration.numerator, r_duration.denominator
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

def crawl_forward(i, note_pitch, spine, lines):
  """Assumes we have found a note with '['. Crawls forward through lines 
  looking for a string of pitch matches to note_pitch, and adding []
  and ] until the string ends. """
  j = i
  prev_note = j
  while True:
    if j + 1 < len(lines) \
      and (len(lines[j+1].split("\t")) < 2
      or note_in_spine(note_pitch, lines[j+1].split("\t")[spine])
      or lines[j+1].split("\t")[spine].strip() == "."):
      if len(lines[j+1].split("\t")) < 2\
        or lines[j+1].split("\t")[spine].strip() == ".":
        j += 1
        continue
      prev_note = j+1
      next_line = lines[j+1].split("\t")[spine].strip()
      pitch_ind = get_index_of_pitch(next_line, note_pitch)
      next_line_notes = next_line.split(" ")
      if next_line_notes[pitch_ind][-1:] != "]":
        next_line_notes[pitch_ind] += "]"
      if next_line_notes[pitch_ind][0] != "[":
        next_line_notes[pitch_ind] = "[" + next_line_notes[pitch_ind]
      lines[j+1] = lines[j+1].split("\t")[0] +"\t"+ " ".join(next_line_notes) \
                    if spine == 1 else \
                      " ".join(next_line_notes) +"\t"+ lines[j+1].split("\t")[1]
    else:
      j = prev_note
      curr_line = lines[j].split("\t")[spine].strip()
      pitch_ind = get_index_of_pitch(curr_line, note_pitch, start_char="[")
      curr_line_notes = curr_line.split(" ")
      curr_line_notes[pitch_ind] = curr_line_notes[pitch_ind][1:]
      lines[j] = lines[j].split("\t")[0] +"\t"+ " ".join(curr_line_notes) \
                    if spine == 1 else \
                      " ".join(curr_line_notes) +"\t"+ lines[j].split("\t")[1]
      break
    j += 1
  return lines

def crawl_backward(i, note_pitch, spine, lines):
  """Assumes we have found a note with ']'. Crawls backwards through
  lines looking for pitch matches of note_pitch in the current spine. 
  Adds [] and [ to extend the tie backwards to matching notes, until
  the string of pitch matches is broken."""
  j = i
  prev_note = j
  while True:
    if j-1 >= 0 and (len(lines[j-1].split("\t")) < 2
    or note_in_spine(note_pitch, lines[j-1].split("\t")[spine])
    or lines[j-1].split("\t")[spine].strip() == "."):
      if len(lines[j-1].split("\t")) < 2\
        or lines[j-1].split("\t")[spine].strip() == ".":
        j -= 1
        continue
      prev_note = j-1
      prev_line = lines[j-1].split("\t")[spine].strip()
      pitch_ind = get_index_of_pitch(prev_line, note_pitch)
      prev_line_notes = prev_line.split(" ")
      if prev_line_notes[pitch_ind][-1:] != "]":
        prev_line_notes[pitch_ind] += "]"
      if prev_line_notes[pitch_ind][0] != "[":
        prev_line_notes[pitch_ind] = "[" + prev_line_notes[pitch_ind]
      lines[j-1] = lines[j-1].split("\t")[0] +"\t"+ " ".join(prev_line_notes) \
                    if spine == 1 else \
                      " ".join(prev_line_notes) +"\t"+ lines[j-1].split("\t")[1]
    else:
      j = prev_note
      curr_line = lines[j].split("\t")[spine].strip()
      pitch_ind = get_index_of_pitch(curr_line, note_pitch, end_char="]")
      curr_line_notes = curr_line.split(" ")
      curr_line_notes[pitch_ind] = curr_line_notes[pitch_ind][:-1]
      lines[j] = lines[j].split("\t")[0] +"\t"+ " ".join(curr_line_notes) \
                    if spine == 1 else \
                      " ".join(curr_line_notes) +"\t"+ lines[j].split("\t")[1]
      break
    j -= 1
  return lines


def fix_ties_for_spine(spine, lines, i):
  """Given a spine number and the index of a line in lines, 
  crawl forward or backwards in the lines, greedily extending
  ties as far as possible. This may overwrite sequences like
  [..][..]..., and I'm okay with that. 
  Spine must be 1 or 0.
  """
  line = lines[i]
  if line.split("\t")[spine].strip() == "[2cc [2a [2f":
    test = 1
  for note in line.split("\t")[spine].split(" "):
    if "[" in note and "]" not in note:
      note_pitch = get_note_pitch(note)
      lines = crawl_forward(i, note_pitch, spine, lines)
    elif "]" in note and "[" not in note:
      note_pitch = get_note_pitch(note)
      lines = crawl_backward(i, note_pitch, spine, lines)
  return lines

def fix_ties(kern_string):
  """Makes sure that all ties in the piece are well-formed. 
  
    Example: [4a [4b 4c    4d 4c 4b       [4a [4b 4c     4d 4c [4b 
             4a 4c 4b     4b 4c      =>   [4a] 4c [4b]   [4b] 4
             4b 4a        4b]             [4b] 4a]       4b]
             4b           4a              4b]            4a
  """
  lines = kern_string.split("\n")
  for i in range(len(lines)):
    line = lines[i]
    if line.strip() == "":
      continue
    if "[" not in line and "]" not in line:
      continue
    if "[" in line.split("\t")[0] or "]" in line.split("\t")[0]:
      lines = fix_ties_for_spine(0, lines, i)
    if "[" in line.split("\t")[1] or "]" in line.split("\t")[1]:
      lines = fix_ties_for_spine(1, lines, i)
  return "\n".join(lines) + "\n"


def convert_to_good_kern(kern_string):
  kern_string = kern_string.strip()
  rhythm_fixed = fix_rhythm(kern_string)
  ties_fixed = fix_ties(rhythm_fixed)
  with_barlines = add_barlines(ties_fixed)
  return (  "**kern" +"\t"  +"**kern\n" 
          + "*staff2"+"\t"  +"*staff1\n"
          + "*Ipiano"+"\t"  +"*Ipiano\n"
          + "*clefF4"+"\t"  +"*clefG2\n"
          +  with_barlines
          + "*-"+"\t"+"*-\n")


if __name__ == "__main__":

  # kern_string = """
  # 4a\t2b
  # 4r\t4r
  # .\t4a
  # .\t4a]
  # """
  # print(convert_to_good_kern(kern_string))

  with open("./music_in_C/Beethoven, Ludwig van___Piano Sonata no. 16 in G major") as f:
    good_kern = convert_to_good_kern(f.read())
  with open("./test.txt", "w") as fw:
    fw.write(good_kern)