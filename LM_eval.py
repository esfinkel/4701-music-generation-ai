import sys, os
import numpy as np
import re
import time
from tqdm import tqdm
from ngram import LMModel
from collections import defaultdict
import math

def is_rest(s):
    rest_re = r"[0-9]+.*r"
    return s=="." or (re.match(rest_re, s) is not None)

def normalize_line(s):
    """Return normalized form of the **kern line `s`: all notes (of either
    hand), sorted alphabetically and then joined by spaces. Will be
    the empty string if `s` only contains rests.
    
    Return None if line does not have any notes."""
    s = s.replace("[", "").replace("]", "")
    if s.strip() == "":
        return None
    if "*" in s or "=" in s:
        return None
    partial = list(set(s.strip().split()))
    partial = list(filter(lambda s: not is_rest(s), partial))
    if len(partial) == 0:
      return None
    return " ".join(sorted(partial))

def normalize_song(lines):
    """Given a list of **kern lines, return a normalized list. All non-music
    lines have been removed; all music lines have been normalized as
    specified in `normalize_line`. """
    lines = list(filter((lambda l: l.strip() != ""), lines))
    song = []
    for line in lines:
        n = normalize_line(line)
        if n is not None:
            song.append(n)
    return song

def gather_counts(directory):
    """Finds unigram, bigram, and trigram counts for all **kern files in 
    `directory`. prepends two <s> lines to the beginning of every file
    and appends two </s> lines to every file. """
    counts_un = defaultdict(int)
    counts_bi = defaultdict(int)
    counts_tri = defaultdict(int)
    for filename in os.listdir(f"./{directory}"):
        if ".DS_Store" in filename:
            continue
        with open(f"./{directory}/{filename}", "r") as f:
            filetext = f.readlines()
        filetext = ["<s>"]*2 + normalize_song(filetext) + ["</s>"]*2
        filetext = list(filter(lambda t: t.strip() != "", filetext))
        for i in range(len(filetext)-2):
            a, b, c = filetext[i].strip()+"\n", filetext[i+1].strip()+"\n", filetext[i+2].strip()+"\n"
            counts_un[c] += 1
            counts_bi[b+c] += 1
            counts_tri[a+b+c] += 1
        counts_un["</s>\n"] += 2
        counts_bi["</s>\n</s>\n"] += 1
    return counts_un, counts_bi, counts_tri

def create_LM():
  counts_un, counts_bi, counts_tri = gather_counts("./music_in_C")
  LM = LMModel(counts_un, counts_bi, counts_tri, 0.2, 0.3, 0.5, 0.001)
  return LM

def log_prob_of_file(filepath, model):
    """Outputs the log probability of the music in file. Also outputs the 
    number of tokens in the file. """
    # vocab = set(counts_un.keys())
    tot = 0
    count = 4
    prev_prev = "<s>\n"
    prev = "<s>\n"
    with open(filepath) as f:
        for line in f:
            count += 1
            line = normalize_line(line)
            if line is None:
              continue
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


def check_generated(filepath):
    print("training model...")
    model = create_LM()
    print("calculating perplexity...")
    print(perplexity(filepath, model))

if __name__ == '__main__':
    if len(sys.argv)<2:
        print('please specify generated file to evaluate')
    else:
        check_generated(sys.argv[1])
