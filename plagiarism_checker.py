import sys, os
import collections
import numpy as np
import re

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
        # include this if rests should be included:
        elif n=="":
            song.append(".")
    return song


def get_song_texts():
    """Visit every song in the training directory. Return a dictionary
    that maps each song title to the contents, normalized as per
    `normalize_song`.
    """
    songs = {}
    directory = 'music_in_C_training'
    for filename in os.listdir(f"./{directory}"):
        if ".DS_Store" in filename:
            continue
        with open(f"./{directory}/{filename}", "r") as f:
            filetext = f.readlines() #.read()
        songs[filename] = normalize_song(filetext)
    return songs


def matching_phrases(generated, original, k, print_phrases=False):
    """Given normalized songs `generated` and `original`, find how many
    length-k passages in `generated` were also present in `original`.

    Returns tuple (matches, count, was_pla)
    * `matches` is the number of passages from `generated` that appeared
        in `original`
    * `count` is the number of times a passage in `generated` appeared
        in `original`
    * `was_pla` is a vector with one index for each passage; the passage's
        value is 1 iff that passage appeared in generated.
    """
    matches = 0
    count = 0
    was_pla = np.zeros(len(generated)-k+1)
    # print()
    original_text = "\n".join(original)
    for i in range(len(generated)-k):
        passage = generated[i:i+k]
        if passage == ["."]*k:
            continue
        # print(len(passage))
        passage_text = "\n".join(passage)
        if passage_text in original_text:
            matches += 1
            was_pla[i] = 1
        if print_phrases and (passage_text in original_text):
            print(i, repr(passage_text))
        count += original_text.count(passage_text)
    return matches, count, was_pla


def check_generated_k(generated, song_texts, k):
    """Given song `generated` and list of training songs `song_texts`
    (all normalized as per `normalize_song`), print out the degree of 
    plagiarism detected, using passages of length `k`.
    
    The values printed are as follows:
    * How many files contain a passage from `generated`
    
    * How many times a passage from `generated` was found in the training data
    * What percent of passages from `generated` were found in the training data. """
    files = 0
    num_passages = len(generated)-k+1
    was_pla_accum = np.zeros(num_passages)
    file_matches = collections.defaultdict(int)
    file_counts = collections.defaultdict(int)
    for song in song_texts:
        file_matches[song], file_counts[song], was_pla = \
                matching_phrases(generated, song_texts[song], k=k)
        if file_matches[song] > 0:
            files += 1
        was_pla_accum += was_pla
    print(f"k={k}:")
    # Sum of "number of passages in `generated` that were found in t_song" for each training song t_song:
    # sum(file_matches.values())
    if files > 0:
        print(
            f"Potential plagiarism from {files} files. {sum(was_pla_accum > 0.1)} passages were found in training data, for a total of {sum(file_counts.values())} times."
        )
    if k==1:
        for i in range(num_passages):
            if was_pla_accum[i]==0:
                print(generated[i])
    plagiarism_threshold = 0.1 # 1>x>0 to account for float error
    filtered_array_val = sum(was_pla_accum > plagiarism_threshold)
    print(f"Passages potentially plagiarized: {round(filtered_array_val*100/num_passages, 1)}%\n")



def check_generated(filepath):
    song_texts = get_song_texts()
    with open(filepath) as f:
        generated = f.readlines()
    generated = normalize_song(generated)
    print('\nk is the length of a "passage"\n')
    for k in [10, 15, 20, 25, 30, 40]:
        check_generated_k(generated, song_texts, k)


if __name__ == '__main__':
    if len(sys.argv)<2:
        print('please specify generated file to evaluate')
    else:
        check_generated(sys.argv[1])
        # with open('./generated_music_RNN/1618031260.txt') as f:
        #     print(normalize_song(f.readlines()))
