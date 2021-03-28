import os
from collections import defaultdict
import math 

class LMModel():
  def __init__(self, counts_un, counts_bi, counts_tri, l1, l2, l3, k):
    self.un = counts_un
    self.bi = counts_bi
    self.tri = counts_tri
    self.l1 = l1
    self.l2 = l2 
    self.l3 = l3 
    self.k = k
    self.vocab = set(counts_un.keys())

  def get_count(self, count_dict, line):
    if line in count_dict:
      return float(count_dict[line])
    return 0.0

  def get_trigram_prob(self, line1, line2, line3):
    """Returns the probability of trigram line1=line2+line3,
    using linear interpolation with add-k smoothing """
    tri_prob = (self.l1 * 
                  ((self.get_count(self.tri,line1+line2+line3) + self.k) / 
                      (self.get_count(self.bi, line1+line2) 
                          + len(self.vocab)*self.k)) )
    bi_prob = self.l2 * ((self.get_count(self.bi,line2+line3) + self.k) / 
                    (self.get_count(self.un, line1) + len(self.vocab)*self.k))
    un_prob = self.l3 * ((self.get_count(self.un,line3)+self.k) / 
                    (len(self.vocab) + self.k*len(self.vocab)))
    return tri_prob + bi_prob + un_prob

def prob_dist(line1, line2, model):
  """Returns a dictionary mapping each possible completion of the trigram
  beginning with line1+line2 to the probability of that trigram, using
  linear interpolation with add-k smoothing. """
  vocab = set(counts_un.keys())
  probs = dict()
  for line3 in vocab:
    probs[line3] = model.get_trigram_prob(line1, line2, line3)
  return probs

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


def log_prob_of_file(filepath, model):
  """Outputs the log probability of the music in file. Also outputs the 
  number of tokens in the file. """
  vocab = set(counts_un.keys())
  tot = 0
  count = 0
  prev_prev = "<s>\n"
  prev = "<s>\n"
  with open(filepath) as f:
    for line in f:
      count += 2
      line = line.strip()+"\n"
      tri_prob = model.get_trigram_prob(prev_prev, prev, line)
      tot += math.log(tri_prob)
      prev_prev = prev
      prev = line 
  for line in ["</s>\n", "</s>\n"]:
    tri_prob = model.get_trigram_prob(prev_prev, prev, line)
    tot += math.log(tri_prob)
    prev_prev = prev
    prev = line 
  return tot, count

def perplexity(filepath, model):
  """returns the perplexity of the file. """
  log_prob, count = log_prob_of_file(filepath, model)
  perplexity = math.exp((-1.0/count) * log_prob)
  return perplexity

if __name__ == "__main__":
  counts_un, counts_bi, counts_tri = gather_counts("music_in_C")
  lm = LMModel(counts_un, counts_bi, counts_tri, 0.7, 0.2, 0.1, 1)
  print(perplexity("./music_in_C/Beethoven, Ludwig van___Piano Sonata No. 19 in G minor", lm))

  probs = prob_dist("16g	.\n", "16e	.\n", lm)
  probs = [(line, prob) for line, prob in probs.items()]
  probs.sort(key=lambda x: x[1], reverse=True)
  ## top five most probable lines to follow 16g  .\n16e  .\n
  for line, prob in probs[:5]:
    print(line)
    print(prob)
  # then, we can easily generate music by using pythons random.choices
  # function, which lets you input a probability distribution
  # over some choices, and selects a value based on that 
  # random_number = random.choices(a_list, distribution)
  # distribution weights don't have to add up to 1. 


      

        