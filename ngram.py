import os
from collections import defaultdict
import math 
import random
import time

import fix_kern

class LMModel():
    def __init__(self, counts_un, counts_bi, counts_tri, l1, l2, l3, k):
        self.un = counts_un
        self.num_tokens = sum(counts_un.values())
        self.bi = counts_bi
        self.tri = counts_tri
        self.l1 = l1
        self.l2 = l2 
        self.l3 = l3 
        self.k = k
        self.vocab = set(counts_un.keys())

    def get_count(self, count_dict, line):
        return float(count_dict.get(line, 0))

    def get_trigram_prob(self, line1, line2, line3):
        """Returns the probability of trigram line1=line2+line3,
        using linear interpolation with add-k smoothing """
        tri_prob = (self.l3 * 
                      ((self.get_count(self.tri,line1+line2+line3) + self.k) / 
                          (self.get_count(self.bi, line1+line2) 
                              + len(self.vocab)*self.k)) )
        bi_prob = self.l2 * ((self.get_count(self.bi,line2+line3) + self.k) / 
                        (self.get_count(self.un, line2) + len(self.vocab)*self.k))
        un_prob = self.l1 * ((self.get_count(self.un,line3)+self.k) / 
                        (self.num_tokens + len(self.vocab)*self.k))
        return tri_prob + bi_prob + un_prob

def prob_dist(line1, line2, model):
    """Returns a dictionary mapping each possible completion of the trigram
    beginning with line1+line2 to the probability of that trigram, using
    linear interpolation with add-k smoothing. """
    vocab = model.vocab
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
    for filename in os.listdir(f"./{directory}"):
        if ".DS_Store" in filename:
            continue
        with open(f"./{directory}/{filename}", "r") as f:
            filetext = f.readlines()
        filetext = ["<s>"]*2 + filetext + ["</s>"]*2
        filetext = list(filter(lambda t: t.strip() != "", filetext))
        for i in range(len(filetext)-2):
            a, b, c = filetext[i].strip()+"\n", filetext[i+1].strip()+"\n", filetext[i+2].strip()+"\n"
            counts_un[c] += 1
            counts_bi[b+c] += 1
            counts_tri[a+b+c] += 1
        counts_un["</s>\n"] += 2
        counts_bi["</s>\n</s>\n"] += 1
    return counts_un, counts_bi, counts_tri


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


def avg_perplexity(dir_path, model):
    """returns the avg perplexity over all the pieces (not normalized)"""
    ps = []
    for filename in os.listdir(f"./{dir_path}"):
        path = f"./{dir_path}/{filename}"
        if ".DS" in path:
            continue
        with open(path) as f:
            if f.read().strip() == "":
                continue
        # try:
        ps.append(perplexity(path, model))
        # except:
        #     print("error w perplex of "+path)
    
    return sum(ps)/len(ps)


def generate_music(start, model, has_max=True, max_notes=200):
    """generates and returns new music starting with `start` (which
    does not contain any start token). If `start` is empty, then it's
    ignored. """
    music = ["<s>\n", "<s>\n"]
    if start is not None and start != "":
        music.extend(start.split("\n"))
    while True:#music[-1] != "</s>\n":
        prob_dictionary = prob_dist(music[-2], music[-1], model)
        sorted_dict = sorted([(prob, tok) for tok, prob in prob_dictionary.items()], reverse=True)
        toks, probs = [], []
        for prob, tok in sorted_dict:
            toks.append(tok)
            probs.append(prob)
            # probs.append(2**prob)
        next = random.choices(toks[:100], weights=probs[:100])[0]
        # next = random.choices(toks, weights=probs)[0]
        # print(next.strip(), 'prob was', prob_dictionary[next], 'max probs were', sorted(probs, reverse=True)[:3], 'num options', len(probs))
        if next.strip() != "</s>":
            music.append(next.strip()+"\n")
        if has_max and len(music) > max_notes:
            print(f"terminated after {max_notes} notes")
            break
    if music[-1] == "</s>\n":
        print("stop symbol seen")
    else:
        music.append("</s>\n")
    return "\n".join(music[2:-1]) + "\n"


def generate_random(model):
    """generates and returns new music"""
    return generate_music("", model)

# def tests():
#     counts_un, counts_bi, counts_tri = gather_counts("music_in_C_training")
#     lm = LMModel(counts_un, counts_bi, counts_tri, 0.7, 0.2, 0.1, 1)
#     print(perplexity("./music_in_C/Beethoven, Ludwig van___Piano Sonata No. 19 in G minor", lm))

#     probs = prob_dist("16g	.\n", "16e	.\n", lm)
#     probs = [(line, prob) for line, prob in probs.items()]
#     probs.sort(key=lambda x: x[1], reverse=True)
#     ## top five most probable lines to follow 16g  .\n16e  .\n
#     for line, prob in probs[:5]:
#         print(line)
#         print(prob)

def write_music(formatted):
    """Given well-formatted kern music, write file"""
    dir = 'generated_music'
    if not os.path.exists(dir):
        os.mkdir(dir)
    with open(f"./{dir}/{round(time.time())}.txt", "w") as file:
        file.write(formatted)


# def perplexity_expt(dir, un, bi, tri):
#     results = []
#     for l1 in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#         for l2 in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#             l3 = 1 - l1 - l2
#             if l3 < 0 or l3 > 1:
#                 continue
#             lm = LMModel(counts_un, counts_bi, counts_tri, l1, l2, l3, k=0.1)
#             results.append((avg_perplexity(dir, lm), l1, l2, l3))
#     return sorted(results, reverse=False)

if __name__ == "__main__":
    counts_un, counts_bi, counts_tri = gather_counts("music_in_C_training")
    lm = LMModel(counts_un, counts_bi, counts_tri, 0, 0, 1, k=0.002)
    # while True:
    #     # print("l1: ",end="")
    #     # l1 = float(input())
    #     l1 = 0.0
    #     # print("l2: ",end="")
    #     # l2 = float(input())
    #     l2 = 0.0
    #     # print("l3: ",end="")
    #     # l3 = float(input())
    #     l3 = 1.0
    #     print("k: ",end="")
    #     k = float(input())
    #     lm = LMModel(counts_un, counts_bi, counts_tri, l1=l1, l2=l2, l3=l3, k=k)
    #     print(avg_perplexity("music_in_C_test", lm))
    # print(perplexity_expt("music_in_C_test", counts_un, counts_bi, counts_tri))
    # generate music
    new_music = generate_random(lm)
    # format and write
    new_music_formatted = fix_kern.convert_to_good_kern(new_music)
    write_music(new_music_formatted)
    # view online at http://verovio.humdrum.org/