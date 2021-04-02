import os
from collections import defaultdict
import math 
import random
import time

import fix_kern

class LMModel():
    def __init__(self, counts_un, counts_bi, counts_tri, counts_quad, counts_quin, l1, l2, l3, l4, l5, k):
        self.un = counts_un
        self.num_tokens = sum(counts_un.values())
        self.bi = counts_bi
        self.tri = counts_tri
        self.quad = counts_quad
        self.quin = counts_quin
        self.l1 = l1
        self.l2 = l2 
        self.l3 = l3 
        self.l4 = l4
        self.l5 = l5
        self.k = k
        self.vocab = set(counts_un.keys())

    def get_count(self, count_dict, line):
        return float(count_dict.get(line, 0))

    def get_pentagram_prob(self, line1, line2, line3, line4, line5):
        """Returns the probability of quingram line1=line2+line3+line4+line5,
        using linear interpolation with add-k smoothing """
        qui_prob = (self.l5 * 
                      ((self.get_count(self.quin,line1+line2+line3+line4+line5) + self.k) / 
                          (self.get_count(self.quad, line1+line2+line3+line4) 
                              + len(self.vocab)*self.k)) )
        qua_prob = (self.l4 * 
                      ((self.get_count(self.quad,line1+line2+line3+line4) + self.k) / 
                          (self.get_count(self.tri, line1+line2+line3) 
                              + len(self.vocab)*self.k)) )
        tri_prob = (self.l3 * 
                      ((self.get_count(self.tri,line1+line2+line3) + self.k) / 
                          (self.get_count(self.bi, line1+line2) 
                              + len(self.vocab)*self.k)) )
        bi_prob = self.l2 * ((self.get_count(self.bi,line2+line3) + self.k) / 
                        (self.get_count(self.un, line1) + len(self.vocab)*self.k))
        un_prob = self.l1 * ((self.get_count(self.un,line3)+self.k) / 
                        (self.num_tokens * (1 + self.k)))
        return qui_prob + qua_prob + tri_prob + bi_prob + un_prob

def prob_dist(line1, line2, line3, line4, model):
    """Returns a dictionary mapping each possible completion of the trigram
    beginning with line1+line2 to the probability of that trigram, using
    linear interpolation with add-k smoothing. """
    vocab = set(counts_un.keys())
    probs = dict()
    for line5 in vocab:
        probs[line5] = model.get_pentagram_prob(line1, line2, line3, line4, line5)
    return probs

def gather_counts(directory):
    """Finds unigram, bigram, and trigram counts for all **kern files in 
    `directory`. prepends two <s> lines to the beginning of every file
    and appends two </s> lines to every file. """

    counts_un = defaultdict(int)
    counts_bi = defaultdict(int)
    counts_tri = defaultdict(int)
    counts_quad = defaultdict(int)
    counts_qui = defaultdict(int)
    for filename in os.listdir(f"./{directory}"):
        if ".DS_Store" in filename:
            continue
        with open(f"./{directory}/{filename}", "r") as f:
            filetext = f.readlines()
        filetext = ["<s>\n"]*4 + filetext + ["</s>\n"]*4
        filetext = list(filter(lambda t: t.strip() != "", filetext))
        for i in range(len(filetext)-4):
            a, b, c = filetext[i].strip()+"\n", filetext[i+1].strip()+"\n", filetext[i+2].strip()+"\n"
            d, e = filetext[i+3].strip()+"\n", filetext[i+4].strip()+"\n"
            counts_un[a] += 1
            counts_bi[a+b] += 1
            counts_tri[a+b+c] += 1
            counts_quad[a+b+c+d] += 1
            counts_qui[a+b+c+d+e] += 1
        counts_un["</s>\n"] += 4
        counts_bi["</s>\n</s>\n"] += 3
        counts_un["</s>\n</s>\n</s>\n"] += 2
        counts_bi["</s>\n</s>\n</s>\n</s>\n"] += 1
    return counts_un, counts_bi, counts_tri, counts_quad, counts_qui


def log_prob_of_file(filepath, model):
    """Outputs the log probability of the music in file. Also outputs the 
    number of tokens in the file. """
    # vocab = set(counts_un.keys())
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

def generate_music(start, model, has_max=True, max_notes=200):
    """generates and returns new music starting with `start` (which
    does not contain any start token) """
    music = ["<s>", "<s>", "<s>", "<s>"]
    music.extend(start.split("\n"))
    while music[-1] != "</s>":
        prob_dictionary = prob_dist(music[-4]+"\n", music[-3]+"\n", music[-2]+"\n", music[-1]+"\n", model)
        sorted_dict = sorted([(prob, tok) for tok, prob in prob_dictionary.items()], reverse=True)
        toks, probs = [], []
        for prob, tok in sorted_dict:
            toks.append(tok)
            probs.append(prob)
            # probs.append(prob**2)
        next = random.choices(toks[:5000], weights=probs[:5000])[0]
        # next = random.choices(toks, weights=probs)[0]
        # print(next.strip(), 'prob was', prob_dictionary[next], 'max probs were', sorted(probs, reverse=True)[:3], 'num options', len(probs))
        music.append(next.strip())
        if has_max and len(music) > max_notes:
            print(f"terminated after {max_notes} notes")
            break
    if music[-1] == "</s>":
        print("stop symbol seen")
    return "\n".join(music[2:-1]) + "\n"


def generate_random(model):
    """generates and returns new music"""
    return generate_music("", model)


def write_music(formatted):
    """Given well-formatted kern music, write file"""
    dir = 'generated_music'
    if not os.path.exists(dir):
        os.mkdir(dir)
    with open(f"./{dir}/{round(time.time())}.txt", "w") as file:
        file.write(new_music_formatted)



def gather_penta_counts(directory):
    counts = defaultdict(int)
    for filename in os.listdir(f"./{directory}"):
        if ".DS_Store" in filename:
            continue
        with open(f"./{directory}/{filename}", "r") as f:
            text = f.readlines()
        for i in range(len(text)-4):
            vals = [text[i], text[i+1], text[i+2], text[i+3], text[i+4]]
            vals = [v.strip() for v in vals]
            counts["\n".join(vals)] += 1
    return counts


def test_penta():
    c = gather_penta_counts("music_in_C_training")
    cc = [(c, t) for t, c in c.items()]
    m = ""
    for c, t in sorted(cc, reverse=True)[:20]:
        m += t + "\n"
    print(fix_kern.convert_to_good_kern(m))


if __name__ == "__main__":
    counts_un, counts_bi, counts_tri, counts_quad, counts_qui = gather_counts("music_in_C_training")
    lm = LMModel(counts_un, counts_bi, counts_tri, counts_quad, counts_qui, 0.1, 0.2, 0.4, 0.2, 0.1, k=1)
    
    # generate music
    new_music = generate_random(lm)
    # format and write
    new_music_formatted = fix_kern.convert_to_good_kern(new_music)
    write_music(new_music_formatted)
    # view online at http://verovio.humdrum.org/