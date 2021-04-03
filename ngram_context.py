import os
from collections import defaultdict
import math 
import random
import time

import fix_kern

class LMModel():
    def __init__(self, vocab, counts_bi, counts_tri, counts_4, counts_5, l3, l4, l5, k):
        self.tri = counts_tri
        self.four = counts_4
        self.five = counts_5
        self.bi = counts_bi
        self.l5 = l5
        self.l4 = l4
        self.l3 = l3 
        self.k = k
        self.vocab = vocab

    def get_count(self, count_dict, line):
        return float(count_dict.get(line, 0))

    def get_5gram_prob(self, line1, line2, line3, line4, line5):
        """Returns the probability of trigram line1=line2+line3,
        using linear interpolation with add-k smoothing """
        five_prob = (self.l5 * 
                      ((self.get_count(self.five,line1+line2+line3+line4+line5) 
                        + self.k) / 
                          (self.get_count(self.four, line1+line2+line3+line4) 
                              + len(self.vocab)*self.k)) )
        four_prob = self.l4 * ((self.get_count(self.four,
                                line2+line3+line4+line5) + self.k) / 
                        (self.get_count(self.tri, line2+line3+line4) + len(self.vocab)*self.k))
        tri_prob = self.l3 * ((self.get_count(self.tri,line3+line4+line5)
                        +self.k) / 
                        (self.get_count(self.bi, line3+line4) + len(self.vocab)*self.k))
        return tri_prob + four_prob + five_prob

def prob_dist(line1, line2, line3, line4, model):
    """Returns a dictionary mapping each possible completion of the trigram
    beginning with line1+line2 to the probability of that trigram, using
    linear interpolation with add-k smoothing. """
    vocab = model.vocab
    probs = dict()
    for line5 in vocab:
        probs[line5] = model.get_5gram_prob(line1, line2, line3, line4, line5)
    return probs

def gather_counts(directory):
    """Finds unigram, bigram, and trigram counts for all **kern files in 
    `directory`. prepends two <s> lines to the beginning of every file
    and appends two </s> lines to every file. """
    vocab = set()
    counts_bi = defaultdict(int)
    counts_tri = defaultdict(int)
    counts_4 = defaultdict(int)
    counts_5 = defaultdict(int)
    for filename in os.listdir(f"./{directory}"):
        if ".DS_Store" in filename:
            continue
        with open(f"./{directory}/{filename}", "r") as f:
            filetext = f.readlines()
        filetext = ["<s>\n"]*4 + filetext + ["</s>\n"]*4
        filetext = list(filter(lambda t: t.strip() != "", filetext))
        for i in range(len(filetext)-4):
            a, b, c, d, e = (filetext[i].strip()+"\n", 
                            filetext[i+1].strip()+"\n", 
                            filetext[i+2].strip()+"\n", 
                            filetext[i+3].strip()+"\n", 
                            filetext[i+4].strip()+"\n")
            counts_5[a+b+c+d+e] += 1
            counts_4[b+c+d+e] += 1
            counts_tri[c+d+e] += 1
            counts_bi[d+e] += 1
            vocab.add(e)
        counts_bi["</s>\n</s>\n"] += 1
        counts_tri["</s>\n</s>\n</s>"] += 1
        counts_4["</s>\n</s>\n</s>\n</s>\n"] += 1
    return vocab, counts_bi, counts_tri, counts_4, counts_5


def log_prob_of_file(filepath, model):
    """Outputs the log probability of the music in file. Also outputs the 
    number of tokens in the file. """
    # vocab = set(counts_un.keys())
    tot = 0
    count = 8
    prev_4 = "<s>\n"
    prev_3 = "<s>\n"
    prev_2 = "<s>\n"
    prev = "<s>\n"
    with open(filepath) as f:
        for line in f:
            count += 1
            line = line.strip()+"\n"
            five_prob = model.get_5gram_prob(prev_4, prev_3, prev_2, prev, line)
            tot += math.log(five_prob)
            prev_4 = prev_3
            prev_3 = prev_2
            prev_2 = prev
            prev = line
    for line in ["</s>\n", "</s>\n", "</s>\n", "</s>\n"]:
        five_prob = model.get_5gram_prob(prev_4, prev_3, prev_2, prev, line)
        tot += math.log(five_prob)
        prev_4 = prev_3
        prev_3 = prev_2
        prev_2 = prev
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
    music = ["<s>", "<s>", "<s>", "<s>"]
    if start is not None and start != "":
        music.extend(start.split("\n"))
    while music[-1] != "</s>":
        prob_dictionary = prob_dist(music[-4]+"\n", music[-3]+"\n", music[-2]+"\n", music[-1]+"\n", model)
        sorted_dict = sorted([(prob, tok) for tok, prob in prob_dictionary.items()], reverse=True)
        toks, probs = [], []
        for prob, tok in sorted_dict:
            toks.append(tok)
            probs.append(prob)
            # probs.append(2**prob)
        next = random.choices(toks[:5000], weights=probs[:5000])[0]
        # next = random.choices(toks, weights=probs)[0]
        # print(next.strip(), 'prob was', prob_dictionary[next], 'max probs were', sorted(probs, reverse=True)[:3], 'num options', len(probs))
        music.append(next.strip())
        print(next.strip())
        if has_max and len(music) > max_notes:
            print(f"terminated after {max_notes} notes")
            break
    if music[-1] == "</s>":
        print("stop symbol seen")
    else:
        music.append("</s>")
    return "\n".join(music[2:-1]) + "\n"


def generate_random(model):
    """generates and returns new music"""
    return generate_music("", model)

def tests():
    counts_un, counts_bi, counts_tri = gather_counts("music_in_C_training")
    lm = LMModel(counts_un, counts_bi, counts_tri, 0.7, 0.2, 0.1, 1)
    print(perplexity("./music_in_C/Beethoven, Ludwig van___Piano Sonata No. 19 in G minor", lm))

    probs = prob_dist("16g	.\n", "16e	.\n", lm)
    probs = [(line, prob) for line, prob in probs.items()]
    probs.sort(key=lambda x: x[1], reverse=True)
    ## top five most probable lines to follow 16g  .\n16e  .\n
    for line, prob in probs[:5]:
        print(line)
        print(prob)

def write_music(formatted):
    """Given well-formatted kern music, write file"""
    dir = 'generated_music'
    if not os.path.exists(dir):
        os.mkdir(dir)
    with open(f"./{dir}/{round(time.time())}.txt", "w") as file:
        file.write(formatted)



if __name__ == "__main__":
    vocab, counts_bi, counts_tri, counts_4, counts_5 = gather_counts("Mozart")
    # lm = LMModel(counts_un, counts_bi, counts_tri, 0.1, 0.2, 0.7, k=0.01)
    lm = LMModel(vocab, counts_bi, counts_tri, counts_4, counts_5, 0.1, 0.2, 0.7, 0.01)
    # print(perplexity_expt("music_in_C_test", counts_un, counts_bi, counts_tri))
    # generate music
    new_music = generate_random(lm)
    # format and write
    new_music_formatted = fix_kern.convert_to_good_kern(new_music)
    write_music(new_music_formatted)
    # view online at http://verovio.humdrum.org/