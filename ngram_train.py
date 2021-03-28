import os
from collections import defaultdict
import math 

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

def get_count(count_dict, line):
  if line in count_dict:
    return float(count_dict[line])
  return 0.0

def invalid_start_stop_token(line1,line2,line3):
  """Returns true if line1 line2 line3 forma  sequence of the form:
  - <s> is in the sequence but is not line1 
  - <s> is line3 
  - </s> is in the sequence but is not line3 
  - </s> is line1 """
  seq = [line1.strip(), line2.strip(), line3.strip()]
  if "<s>" in seq and "<s>" != seq[0]:
    return True 
  if "<s>" == seq[2]:
    return True 
  if "</s>" in seq and "</s>" != seq[2]:
    return True
  if "</s>" == seq[0]:
    return True 
  return False

def ln_probs(counts_un, counts_bi, counts_tri, l1, l2, l3, k):
  """ Creates dictionary mapping line1 line2 line3 to log probability: 
        lambda1 * P(line3 | line1 line2) + 
        lambda2 * P(line3 | line2) + 
        lambda3 * P(line3)
      For every combination of line1, line2, line3 in the vocabulary. Also
      includes probabilities for any number of unknown lines in the trigram.
      
      For example, line1 _ line3 is a case where we see a trigram where we know
      line1 and line3, but we have never seen the second line. We essentially
      treat _ as a line with count 0. 

      Uses add-k smoothing, except in the unigram case since we won't see
      any unknowns. 
  """
  vocab = set(counts_un.keys()) | {"_\n"}
  # print(len(vocab))
  probs = dict()
  for line1 in vocab:
    for line2 in vocab:
      for line3 in vocab:
        if invalid_start_stop_token(line1,line2,line3):
          continue
        tri_prob = l1 * ((get_count(counts_tri,line1+line2+line3) + k) / 
                        (get_count(counts_bi, line1+line2) + len(vocab)*k))
        bi_prob = l2 * ((get_count(counts_bi,line2+line3) + k) / 
                        (get_count(counts_un, line1) + len(vocab)*k))
        un_prob = l3 * ((get_count(counts_un,line3)+k) / (len(vocab) + k*len(vocab)))
        probs[line1+line2+line3] = math.log(tri_prob + bi_prob + un_prob)
        # if "16g" in line1:
        #   print(line1+line2+line3)
        if line1+line2+line3 == """16g\t.\n16e\t.\n16g\t.\n""":
          print(tri_prob)
          print(bi_prob)
          print(un_prob)
          print("""16g	.\n16e	.\n16g	.\n""")
  return probs

if __name__ == "__main__":
  counts_un, counts_bi, counts_tri = gather_counts("music_in_C")
  # un_counts = [(line, count) for line, count in counts_tri.items()]
  # un_counts.sort(key=lambda x: x[1], reverse=True)
  # for line, count in un_counts[:10]:
  #   print(line)
  #   print(count)
  probs = ln_probs(counts_un, counts_bi, counts_tri, 0.7, 0.2, 0.1, 1)
  # probs["""16g	.\n16e	.\n16g	.\n"""]
      

        