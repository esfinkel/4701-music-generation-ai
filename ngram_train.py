import os
from collections import defaultdict

def gather_counts(directory):
  counts_un = defaultdict(int)
  counts_bi = defaultdict(int)
  counts_tri = defaultdict(int)
  prev_prev = "<s>"
  prev = "<s>"
  for filename in os.listdir(f"./{directory}"):
    with open(f"./{directory}/{filename}", "r") as f:
      for line in f:
        line = line.strip()
        if len(line) == 0:
          continue
        counts_un[line+"\n"] += 1
        counts_bi[prev+"\n"+line+"\n"] += 1
        counts_tri[prev_prev+"\n"+prev+"\n"+line+"\n"] += 1
        prev_prev = prev
        prev = line
      counts_un["</s>"] += 2
      counts_bi["</s>\n</s>\n"] += 1
      counts_bi[prev+"\n"+"</s>\n"] += 1
      counts_tri[prev_prev+"\n"+prev+"\n" + "</s>\n"] += 1
      counts_tri[prev+"\n</s>\n</s>\n"] += 1
      

        