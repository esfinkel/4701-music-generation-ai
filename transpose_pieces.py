import os

## order of sharps: f c g d a e b
## ex: a key with 4 sharps is the key sharp_keys[4] = e
sharp_keys = ["c", "g", "d", "a", "e", "b", "f#", "c#"]

## order of flats: b e a d g c f
## ex: a key with 4 flats is the key flat_keys[4] = g#,
##     or more correctly, Ab, but I want to represent notes consistently 
flat_keys = ["c", "f", "a#", "d#", "g#", "c#", "f#", "b"]


## loop thorugh the files in raw1/ and find the key each pieces is in. 
## then load the cleaned up version of that file from processed_music/
## and create a version of the piece that's transposed into the key of C
## (initially; the piece may change keys throughout). 
## store the transposed piece in music_in_C/
# if __name__ == "__main__":
#   for filename in os.listdir("./raw1/"):
#     with open(f"./raw1/{filename}", "r") as f:
#       ??
#     with open(f"./processed_music/{filename}", "w") as f:
#       f.write(file_string)