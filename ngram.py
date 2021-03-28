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
      counts_un["</s>\n"] += 2
      counts_bi["</s>\n</s>\n"] += 1
      counts_bi[prev+"\n"+"</s>\n"] += 1
      counts_tri[prev_prev+"\n"+prev+"\n" + "</s>\n"] += 1
      counts_tri[prev+"\n</s>\n</s>\n"] += 1
  return counts_un, counts_bi, counts_tri

def get_count(count_dict, line):
  if line in count_dict:
    return float(count_dict[line])
  return 0.0

def get_trigram_prob(line1, line2, line3, counts_un, counts_bi, counts_tri, l1, l2, l3, k):
  tri_prob = l1 * ((get_count(counts_tri,line1+line2+line3) + k) / 
                  (get_count(counts_bi, line1+line2) + len(counts_un)*k))
  bi_prob = l2 * ((get_count(counts_bi,line2+line3) + k) / 
                  (get_count(counts_un, line1) + len(counts_un)*k))
  un_prob = l3 * ((get_count(counts_un,line3)+k) / 
                  (len(counts_un) + k*len(counts_un)))
  return tri_prob + bi_prob + un_prob

def prob_dist(line1, line2, counts_un, counts_bi, counts_tri, l1, l2, l3, k):
  """Returns a dictionary mapping each possible completion of the trigram
  beginning with line1+line2 to the probability of that trigram, using
  linear interpolation with add-k smoothing. """
  vocab = set(counts_un.keys())
  probs = dict()
  print("<s>" in vocab)
  for line3 in vocab:
    probs[line3] = get_trigram_prob(line1, line2, line3, counts_un, counts_bi, counts_tri, l1, l2, l3, k)
  return probs


def log_prob_of_file(filepath, counts_un, counts_bi, counts_tri, l1, l2, l3, k):
  """Outputs the log probability of the music in file. """
  vocab = set(counts_un.keys())
  tot = 0
  prev_prev = "<s>\n"
  prev = "<s>\n"
  with open(filepath) as f:
    for line in f:
      line = line.strip()+"\n"
      tri_prob = get_trigram_prob(prev_prev, prev, line, counts_un, counts_bi, counts_tri, l1, l2, l3, k)
      tot += math.log(tri_prob)
      prev_prev = prev
      prev = line 
  for line in ["</s>\n", "</s>\n"]:
    tri_prob = get_trigram_prob(prev_prev, prev, line, counts_un, counts_bi, counts_tri, l1, l2, l3, k)
    tot += math.log(tri_prob)
    prev_prev = prev
    prev = line 
  return tot

if __name__ == "__main__":
  counts_un, counts_bi, counts_tri = gather_counts("music_in_C")
  probs = prob_dist("16g	.\n", "16e	.\n", counts_un, counts_bi, counts_tri, 0.7, 0.2, 0.1, 1)
  probs = [(line, prob) for line, prob in probs.items()]
  probs.sort(key=lambda x: x[1], reverse=True)
  ## top ten most probable lines to follow 16g  .\n16e  .\n
  for line, prob in probs[:10]:
    print(line)
    print(prob)
  ## then, we can easily generate music by using pythons random.choices
  ## function, which lets you input a probability distribution
  ## over some choices, and selects a value based on that 
  ## random_number = random.choices(a_list, distribution)
  ## distribution weights don't have to add up to 1. 



###################################################################
######################## USELESS ##################################
###################################################################

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
  """ TAKES TOO LONG TO RUN. Apparently 36k^3 is a big number. 
  
      Creates dictionary mapping line1 line2 line3 to log probability: 
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
  print(len(vocab))
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
  return probs

      

        