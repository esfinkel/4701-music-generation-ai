import os
from collections import defaultdict
import math 
import random
import time
from music_helpers import convert_to_duration, get_duration_of_spine

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
                        (self.num_tokens * (1 + self.k)))
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

def get_next_beat(lines, i):
    """returns a concatenation of the lines after i that form
    1 quarter note."""
    next_line = ""
    r_dur = 0
    l_dur = 0
    j = i
    for line in lines[i:]:
        j += 1
        if line.strip() == "":
            continue
        l_notes, r_notes = line.split("\t")[0], line.split("\t")[1]
        r_dur += convert_to_duration(get_duration_of_spine(r_notes))
        l_dur += convert_to_duration(get_duration_of_spine(l_notes))
        next_line += line.strip()+"\n"
        # if r_dur >= 0.25 or l_dur >= 0.25:
        if r_dur%0.25<=0.01 and l_dur%0.25<=0.01:
            break
    return next_line, j

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
        filetext = list(filter(lambda t: t.strip() != "", filetext))
        i = 0
        music = ["<s>\n", "<s>\n"]
        while i < len(filetext):
            next_line, i = get_next_beat(filetext, i)
            music.append(next_line)
        # music += ["</s>\n"]*2
        for i in range(len(music)-2):
            a, b, c = music[i], music[i+1], music[i+2]
            counts_un[c] += 1
            counts_bi[b+c] += 1
            counts_tri[a+b+c] += 1
        # counts_un["</s>\n"] += 2
        # counts_bi["</s>\n</s>\n"] += 1
    return counts_un, counts_bi, counts_tri



def generate_music(model, has_max=True, max_beats=100):
    """generates and returns new music starting with `start` (which
    does not contain any start token). If `start` is empty, then it's
    ignored. """
    music = ["<s>\n", "<s>\n"]
    while music[-1] != "</s>\n":
        prob_dictionary = prob_dist(music[-2], music[-1], model)
        sorted_dict = sorted([(prob, tok) for tok, prob in prob_dictionary.items()], reverse=True)
        toks, probs = [], []
        for prob, tok in sorted_dict:
            toks.append(tok)
            probs.append(prob)
        # print([round(i, 6) for i in probs[:10]])
        # print(len(probs))
        # next = random.choices(toks[:100], weights=probs[:100])[0]
        next = random.choices(toks, weights=probs)[0]
        # print(next, probs[toks.index(next)])
        music.append(next.strip()+"\n")
        # print(next)
        if has_max and len(music) > max_beats:
            print(f"terminated after {max_beats} beats")
            break
    if music[-1] == "</s>\n":
        print("stop symbol seen")
    else:
        music.append("</s>\n")
    return "\n".join(music[2:-1]) + "\n"

def write_music(formatted):
    """Given well-formatted kern music, write file"""
    dir = 'generated_music_LM_beats'
    if not os.path.exists(dir):
        os.mkdir(dir)
    with open(f"./{dir}/{round(time.time())}.txt", "w") as file:
        file.write(formatted)


if __name__ == "__main__":
    counts_un, counts_bi, counts_tri = gather_counts("music_in_C_training")

    lm = LMModel(counts_un, counts_bi, counts_tri, l1=0.2, l2=0.2, l3=0.6, k=0.1)

    new_music = generate_music(lm)
    new_music_formatted = fix_kern.convert_to_good_kern(new_music)
    write_music(new_music_formatted)
    # view online at http://verovio.humdrum.org/
