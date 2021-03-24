import os
import re

## NOTE: There are 12 pieces not explicitly in C Major or A Minor that 
##       don't list an explicit key. They have not been transposed or
##       added to the key-of-C training data. 

sharp_order = ["f#","c#","g#", "d#", "a#", "e#", "b#"]
## ex: a key with 4 sharps is the key sharp_keys[4] = e
sharp_keys = ["c", "g", "d", "a", "e", "b", "f#", "c#"]

flat_order = ["b-", "e-", "a-", "d-", "g-", "c-", "f-"]
## ex: a key with 4 flats is the key flat_keys[4] = g#,
##     or more correctly, Ab, but I want to represent notes consistently 
flat_keys = ["c", "f", "a#", "d#", "g#", "c#", "f#", "b"]

def get_key(f):
    """Given an unprocessed **kern file, returns the key of the piece, 
    according to the *k[...] line. If no such line exists, throws 
    ValueError. If the key is empty and the piece is not explicitly
    listed as being in C major or A minor, throws AssertionError. If
    the listed key does not follow standard key conventions (e.g.
    bad combination of sharps, wrong ordering of sharps, etc.) throws
    AssertionError. """
    for line in f:
        if line[:3] == "*k[":
            key = line.strip().split("\t")[0][3:-1]
            if len(key.strip()) == 0:
                if ("c major" in filename.lower() or "a minor" in filename.lower() 
                        or "c-major" in filename.lower()):
                    return "c"
                assert len(key.strip()), f"empty key"

            if "#" in key:
                assert key == "".join(sharp_order[:len(key.split("#")) - 1]), \
                        f"key doesn't match standard sharp odering: {key}"
                return sharp_keys[len(key.split("#")) - 1]
            else:
                assert key == "".join(flat_order[:len(key.split("-")) - 1]), \
                        f"key doesn't match standard flat odering: {key}"
                return flat_keys[len(key.split("-")) - 1]
    raise ValueError("No **kern line specifying key.")

def transpose_piece_by_octaves(text: str, octaves: int):
    """Given the `text` of a **kern files, transpose the piece by `octaves`
    octaves (up or down depending on sign). Return transposed result.
    This function exists because the function `transpose_piece` transposes
    everything up, so some pieces will be transposed up 11 half-steps, and
    we may want to transpose them back down one octave. """
    if octaves==0:
        return text

    ordering = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
    ordering_bass = [c.upper() for c in ordering]

    def transpose_note(match, octaves):
        """Given the match object `match`, transpose the note it contains
        by `octaves` octaves. Return transposed note.
        Precondition: `octaves != 0` """
        old_note = match.group(1)
        new_note = old_note
        while octaves < 0: # transpose down
            if new_note in ordering:
                new_note = ordering_bass[ordering.index(new_note)]
            elif new_note[0] in ordering:
                new_note = new_note[1:]
            else:
                new_note = new_note[0] + new_note
            octaves += 1
        while octaves > 0: # transpose up
            if new_note in ordering_bass:
                new_note = ordering[ordering_bass.index(new_note)]
            elif new_note[0] in ordering_bass:
                new_note = new_note[1:]
            else:
                new_note = new_note[0] + new_note
            octaves -= 1
        return match.group(0).replace(old_note, new_note, 1)

    note_pattern = r'\d+\.*([A-Ga-g]+#?)'
    new_text, _ = re.subn(note_pattern, (lambda x: transpose_note(x, octaves)), text)
    return new_text



def transpose_piece(text: str, orig_key: str):
    """Given the `text` of a **kern file in key `orig_key`, transpose 
    the piece up to 'c'. """
    if orig_key == 'c':
        return text, 0
    ordering = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
    ordering_bass = [c.upper() for c in ordering]
    tr_by = (12 - ordering.index(orig_key)) % 12

    def transpose_note(match):
        """Given a match object `match` containing a note, transpose the
        note up by `tr_by` half-steps and return resulting string. The
        variable `tr_by` is defined in the enclosing function. """
        old_note = match.group(1)
        # if note is bass and would pass middle c, switch to treble
        if old_note in ordering_bass and ordering_bass.index(old_note)+tr_by >= 12:
            new_note = ordering[ordering_bass.index(old_note)+tr_by-12]
            return match.group(0).replace(old_note, new_note, 1)
        # process original note
        was_sharp = '#' in old_note
        was_bass = old_note[0] in ordering_bass
        old_octave = len(old_note) - was_sharp # number of letters in note
        old_note_type = old_note[0].lower()
        if was_sharp:
            old_note_type += '#'
        # calculate new note
        new_note_type_ind = (ordering.index(old_note_type.lower())+tr_by) % 12
        new_note_type = (ordering_bass if was_bass else ordering)[new_note_type_ind]
        new_note = new_note_type[0] * old_octave
        is_sharp = '#' in new_note_type
        if is_sharp:
            new_note += '#'
        if (ordering.index(old_note_type.lower())+tr_by) >= 12:
            # note jumps to next octave; add or remove repeat letter accordingly
            if was_bass:
                assert len(new_note)-is_sharp >= 2
                new_note = new_note[1:]
            else:
                new_note = new_note[0] + new_note
        # return original text with note replaced
        return match.group(0).replace(old_note, new_note, 1)

    note_pattern = r'\d+\.*([A-Ga-g]+#?)'
    new_text, num_notes = re.subn(note_pattern, transpose_note, text)
    return new_text, num_notes


def edit_key_sig(piece, orig_key):
    """Looks in the **kern file and replaces the original key signature
    with the C major key signature. """
    print('edit_key_sig has not been implemented')
    return piece


def test_transpose():
    # test that formatting is preserved
    test_piece = '\n16..gg\n4GG8A-----'
    test_piece_transposed, num_notes = transpose_piece(test_piece, 'g')
    assert test_piece_transposed == '\n16..ccc\n4C8d-----', repr(test_piece_transposed)
    assert num_notes == 3
    assert test_piece == transpose_piece_by_octaves(transpose_piece_by_octaves(test_piece, -2), 2)
    assert test_piece == transpose_piece_by_octaves(transpose_piece_by_octaves(test_piece, 2), -2)
    
    # now some chromatic test_pieces
    # transpose bass clef up a half-step
    test_piece = '\n4AA4AA#4BB4C4C#4D4D#4E4F4F#4G4G#4A4A#4B4c4c#4d'
    test_piece_transposed, _ = transpose_piece(test_piece, 'b')
    test_piece_tr_manually = '\n4AA#4BB4C4C#4D4D#4E4F4F#4G4G#4A4A#4B4c4c#4d4d#'
    assert test_piece_transposed == test_piece_tr_manually

    # transpose bass clef up 11 half-steps
    test_piece = '\n4AA4AA#4BB4C4C#4D4D#4E4F4F#4G4G#4A4A#4B4c4c#4d'
    test_piece_transposed, _ = transpose_piece(test_piece, 'c#')
    test_piece_tr_manually = '\n4G#4A4A#4B4c4c#4d4d#4e4f4f#4g4g#4a4a#4b4cc4cc#'
    assert test_piece_transposed == test_piece_tr_manually
    assert transpose_piece_by_octaves(test_piece_transposed, -1) == \
            '\n4GG#4AA4AA#4BB4C4C#4D4D#4E4F4F#4G4G#4A4A#4B4c4c#'

    # transpose treble clef up a half-step
    test_piece = '4d4d#4e4f4f#4g4g#4a4a#4b4cc4cc#4dd4dd#4ee'
    test_piece_transposed, _ = transpose_piece(test_piece, 'b')
    test_piece_tr_manually = '4d#4e4f4f#4g4g#4a4a#4b4cc4cc#4dd4dd#4ee4ff'
    assert test_piece_transposed == test_piece_tr_manually

    # transpose treble clef up 11 half-steps
    test_piece = '4d4d#4e4f4f#4g4g#4a4a#4b4cc4cc#4dd4dd#4ee'
    test_piece_transposed, _ = transpose_piece(test_piece, 'c#')
    test_piece_tr_manually = '4cc#4dd4dd#4ee4ff4ff#4gg4gg#4aa4aa#4bb4ccc4ccc#4ddd4ddd#'
    assert test_piece_transposed == test_piece_tr_manually
    assert transpose_piece_by_octaves(test_piece_transposed, -1) == \
            '4c#4d4d#4e4f4f#4g4g#4a4a#4b4cc4cc#4dd4dd#'


## loop thorugh the files in raw1/ and find the key each pieces is in. 
## then load the cleaned up version of that file from processed_music/
## and create a version of the piece that's transposed into the key of C
## (initially; the piece may change keys throughout). 
## store the transposed piece in music_in_C/
if __name__ == "__main__":
    test_transpose()

    for filename in os.listdir("./raw1/"):
        if '.DS_Store' in filename:
            continue
        try:
            with open(f"./raw1/{filename}", "r") as f:
                key = get_key(f)

            with open(f"./processed_music/{filename}", "r") as rf:
                dir = 'music_in_C'
                if not os.path.exists(dir):
                    os.mkdir(dir)
                with open(f"./{dir}/{filename}", "w") as wf:
                    try:
                        # should maybe log specific errors
                        transposed, _ = transpose_piece(rf.read(), key)
                        # transposed = edit_key_sig(transposed, key)
                        wf.write(transposed)
                    except Exception as e:
                        print(e)

        except AssertionError as err:
            print(f"{filename}: {err}")
        except ValueError as err:
            print(f"{filename}: {err}")
        except:
            print(f"Unexpected error in {filename}")
        # break
