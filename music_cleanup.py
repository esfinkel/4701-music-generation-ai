import os
import re

def should_remove_line(line):
  """Returns true if this is a line of **kern that should not 
  appear at all in the cleaned up data."""
  if line[0] in {'!', '=', '*'} and '*^' not in line and '*v' not in line:
    return True
  if 'q' in line:
    return True
  return False 

def find_dynamics_columns(line):
  """Returns a list of column indices in the **kern code that correspond
  to columns we don't care about.
  
  These columns may move around depending on the presence of *^
  and *v lines."""
  column_headers = line.split("\t")
  dynam_columns = []
  for i in range(len(column_headers)):
    if column_headers[i] != "**kern":
      dynam_columns.append(i)
  return dynam_columns

def update_dynam_col(line, dynam_columns):
  """Outputs the new indices of the dynamics columns,
  after any number of columns have been split or merged
  with the *^ or *v syntax."""
  columns = line.split("\t")
  i = len(columns) - 1
  while i >= 0:
    if columns[i] == "*^":
      for j in range(len(dynam_columns)):
        if dynam_columns[j] > i:
          dynam_columns[j] += 1
      if i in dynam_columns:
        dynam_columns.append(i+1)
        dynam_columns.sort()
    elif columns[i] == "*v":
      k = i - 1
      while columns[k] == "*v":
        k -= 1
      num_cols_merged = i-k 
      for j in range(len(dynam_columns)):
        if dynam_columns[j] > i:
          dynam_columns[j] -= num_cols_merged - 1
      i = k + 1
    i -= 1
  return dynam_columns

def remove_extraneous_information(f):
  """Outputs a string that is equivalent to the data found in f, 
  but with all extraneous data removes (bar lines, kern column
  markers, grace notes, comments, formatting and dynamics, ...). Lines
  denoting column merges or splits are left in, for checks done later. """
  out_string = ""
  dynam_columns = []
  for line in f:
    if "**kern" in line:
      dynam_columns = find_dynamics_columns(line)
    if '*^' in line or '*v' in line:
      dynam_columns = update_dynam_col(line, dynam_columns)
      continue
    if should_remove_line(line):
      continue
    columns = line.split("\t")
    for i in range(len(columns)):
      if i not in dynam_columns:
        cleaned_note = re.sub("[^A-Ga-g0-9\[\]r#\-\.]", "", columns[i])
        out_string += cleaned_note + "\t"
    out_string += "\n"



## loop thorugh the files in raw1/, and create new, processed files in 
## processed_music/ with the same filename 
for filename in os.listdir("./raw1/"):
  with open(f"./raw1/{filename}") as f:
    file_string = remove_extraneous_information(f)