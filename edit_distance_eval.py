import sys, os
import collections
import numpy as np
import re
import time
from tqdm import tqdm

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
    return "\n".join(song)

def insertion_cost(training, j):
    return 1

def deletion_cost(generated, i):
    return 1

def substitution_cost(generated, training, i, j):
    if generated[i-1] == training[j-1]:
        return 0
    else:
        return 2

def normalized_edit_distance(generated, training):
    m = len(generated) + 1
    n = len(training) + 1
    
    matrix = np.zeros((m, n))
    for i in range(1, m):
        matrix[i, 0] = matrix[i-1, 0] + deletion_cost(generated, i)
    
    for j in range(1, n):
        matrix[0, j] = matrix[0, j-1] + insertion_cost(training, j)
    
    for i in range(1, m):
        for j in range(1, n):
            matrix[i, j] = min(
                matrix[i-1, j] + deletion_cost(generated, i), 
                matrix[i, j-1] + insertion_cost(training, j), 
                matrix[i-1, j-1] + substitution_cost(generated, training, i, j)
            )
    
    return matrix[-1][-1] / abs(len(training) - len(generated))

def get_song_strings():
    """Visit every song in the training directory. Return a dictionary
    that maps each song title to the contents, normalized as per
    `normalize_song`.
    """
    print("getting normalized training set...")
    songs = {}
    directory = 'music_in_C_test'
    for filename in os.listdir(f"./{directory}"):
        if ".DS_Store" in filename:
            continue
        with open(f"./{directory}/{filename}", "r") as f:
            filetext = f.readlines()
        songs[filename] = normalize_song(filetext)
    return songs


def check_generated(filepath):
    song_texts = get_song_strings()
    with open(filepath) as f:
        generated = f.readlines()
    generated = normalize_song(generated)
    print("checking edit distance...")
    start_time = time.time()
    tot_distance = 0
    for training_song in tqdm(song_texts):
        tot_distance += normalized_edit_distance(generated, training_song)
    print(f"Average normalized edit distance from real music: {tot_distance / len(song_texts)}")
    print(f"computation time: {time.time() - start_time}")


if __name__ == '__main__':
    if len(sys.argv)<2:
        print('please specify generated file to evaluate')
    else:
        check_generated(sys.argv[1])
