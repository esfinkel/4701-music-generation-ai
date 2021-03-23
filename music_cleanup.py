import os
import re
from music_helpers import convert_to_duration, to_cannonical_note

def should_remove_line(line):
  """Returns true if this is a line that should not 
  appear at all in the cleaned up data. Lines with *^ and 
  *v won't appear in the outputted cleaned file, 
  but we need them for intermediate steps and thus won't
  immediately remove them."""
  if line[0] in {'!', '=', '*'} and '*^' not in line and '*v' not in line:
    return True
  if 'q' in line:
    return True
  return False 

def find_columns(line):
  """Returns a list of column indices in the **kern code that correspond
  to columns representing the left hand notes and the right hand notes.
  Input must be the line in the document that specifies **kern
  vs non-**kern columns."""
  column_headers = line.split("\t")
  hand_columns = []
  for i in range(len(column_headers)):
    if column_headers[i].strip() == "**kern":
      hand_columns.append([i])
  assert len(hand_columns) == 2, \
    "# **kern columns != 2: " + " | ".join(column_headers)
  return hand_columns

def insert_col(cols_list, i):
  """Return updated cols_list, reflecting that a column has been
  inserted at position i. If i occurs within cols_list,
  update reflects that this new column is in cols_list."""
  for j in range(len(cols_list)):
    if cols_list[j] > i:
      cols_list[j] += 1
  if i in cols_list:
    cols_list.append(i+1)
    cols_list.sort()
  return cols_list

def merge_cols(cols_list, columns, i):
  """Returns updated cols_list where the merge group ending at 
  i has been merged. Also returns the position of the left-most 
  column in the merge group.
  Throws an error if the merge group was only partially in this column
  list (a merge group should be either not at all overlapping with
  this column list, or compeltely overlapping)."""
  k = i - 1
  while columns[k] == "*v":
    if i in cols_list:
      assert k in cols_list, "merge group not contained in one column list"
    k -= 1
  num_cols_merged = i-k 
  original_col_values = cols_list.copy()
  for j in range(len(cols_list)):
    if cols_list[j] > i:
      cols_list[j] -= num_cols_merged - 1
  for col in original_col_values:
    if col > k + 1 and col <= i:
      cols_list.remove(col)
  i = k + 1
  return cols_list, i

def update_hand_columns(line, hand_columns):
  """Outputs a list of two lists, where the list at position 0
  corresponds to the columns that are assigned to the left hand, 
  and the list at position 1 corresponds to the columns that 
  are assigned to the right hand, given the merges and splits
  indicated in `line`. 
  Input must be a line containing only tab-separated *, *^, and *v"""
  columns = line.split("\t")
  left_cols = hand_columns[0]
  right_cols = hand_columns[1]
  i = len(columns) - 1
  while i >= 0:
    if columns[i] == "*^":
      left_cols = insert_col(left_cols, i)
      right_cols = insert_col(right_cols, i)
    elif columns[i] == "*v":
      left_cols, _ = merge_cols(left_cols, columns, i)
      right_cols, i = merge_cols(right_cols, columns, i)
    i -= 1
  return [left_cols, right_cols]

def combine_notes(notes):
  """return a list of notes such that the list is either: a single
  rest of the shortest duration in the list (if the list was made up 
  entirely of rests), a single `.` (if the list was entirely `.`s), or
  the list with all rests and standalone `.`s removed, with no
  duplicate notes, ordered such that the note with the shortest 
  duration comes first. 
  Notes with the same pitch but different durations are not 
  considered to be duplicates."""
  if len(notes) == 0:
    return ["."]
  notes = sorted(list(set(notes)), key=lambda x: convert_to_duration(x))
  all_dots = True
  all_rests = True 
  for note in notes:
    if 'r' not in note:
      all_rests = False 
    if note != '.':
      all_dots = False 
  if all_rests:
    return [notes[0]]
  if all_dots:
    return ["."]
  return list(filter(lambda x: 'r' not in x and x != ".", notes))

def convert_notes_to_cannonical_form(notes):
  """Returned notes, but with every note converted to its
  'cannonical form'. Rests and `.`s are unchanged. 
  Example: ["2.f##]", "4c-"]  => ["2.g]", "4B"]
  Example: ["."] => ["."]
  Example: ["4r"] => ["4r"] 
  Requires: If notes contains a `.` or rest, then notes has length 1. """
  if len(notes) == 1:
    if "." == notes[0] or "r" in notes[0]:
      return notes
  cannonical_notes = []
  for note in notes:
    pitch = re.sub("[^A-Ga-g#\-]", "", note)
    cannonical_pitch = to_cannonical_note(pitch)
    cannonical_notes.append(re.sub("[A-Ga-g]+[#\-]*", cannonical_pitch, note))
  return cannonical_notes

def clean_line(line, hand_columns):
  """Outputs `line` with extraneous characters and columns removed. 
  Condenses all left hand columns and right hand columns into a single column
  each. """
  left_notes = []
  right_notes = []
  columns = line.split("\t")
  for i in range(len(columns)):
    columns[i] = re.sub("[^A-Ga-g0-9\ \[\]r#\-\.]", "", columns[i])
    if i in hand_columns[0]:
      left_notes.extend(columns[i].split(" "))
    if i in hand_columns[1]:
      right_notes.extend(columns[i].split(" "))
  left_notes = convert_notes_to_cannonical_form(combine_notes(left_notes))
  right_notes = convert_notes_to_cannonical_form(combine_notes(right_notes))
  return " ".join(left_notes) + "\t" + " ".join(right_notes)

def clean_file(f):
  """Outputs a string that is equivalent to the data found in f, 
  but with all extraneous data removed, and all staffs consolidated
  to one column when split."""
  out_string = ""
  hand_columns = []
  for line in f:
    if len(line.strip()) == 0:
      continue
    if "**kern" in line and line[0] == "*":
      hand_columns = find_columns(line)
    if should_remove_line(line):
      continue
    cleaned_line = clean_line(line, hand_columns) 
    if '*^' in line or '*v' in line:
      hand_columns = update_hand_columns(line, hand_columns)
    else:
      out_string += cleaned_line + "\n"
  return out_string

## loop thorugh the files in raw1/, and create new, processed files in 
## processed_music/ with the same filename 
if __name__ == "__main__":
  for filename in os.listdir("./raw1/"):
    try:
      file_string = ""
      with open(f"./raw1/{filename}", "r") as f:
        file_string = clean_file(f)
      with open(f"./processed_music/{filename}", "w") as f:
        f.write(file_string)
    except AssertionError as err:
      print(f"Assertion error in reading {filename}: {err}")
    except:
      print(f"Unexpected error in reading {filename}")
