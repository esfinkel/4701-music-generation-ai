import os
from collections import defaultdict

def gather_counts(directory):
  """Finds unigram, bigram, and trigram counts for all **kern files in 
  `directory`. prepends two <s> lines to the beginning of every file
  and appends two </s> lines to every file. """
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
  return counts_un, counts_bi, counts_tri

# def ln_probs(counts, l1, l2, l3):
#   """Creates dictionary of log probabilities, where:
#   -  the probability of line1 line2 line3 is lambda1 * P(line3 | line1 line2) + 
#   lambda2 * P(line3 | line2) + lambda3 * P(line3)
#   - the probability of line1 line2 is lambda"""

if __name__ == "__main__":
  counts_un, counts_bi, counts_tri = gather_counts("music_in_C")
  un_counts = [(line, count) for line, count in counts_un.items()]
  un_counts.sort(key=lambda x: x[1], reverse=True)
  for line, count in un_counts[:10]:
    print(line)
    print(count)
      

        