import os

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
        return sharp_keys[len(key.split("#"))]
      else:
        assert key == "".join(flat_order[:len(key.split("-")) - 1]), \
              f"key doesn't match standard flat odering: {key}"
        return flat_keys[len(key.split("-"))]
        
## loop thorugh the files in raw1/ and find the key each pieces is in. 
## then load the cleaned up version of that file from processed_music/
## and create a version of the piece that's transposed into the key of C
## (initially; the piece may change keys throughout). 
## store the transposed piece in music_in_C/
if __name__ == "__main__":
  for filename in os.listdir("./raw1/"):
    try:
      with open(f"./raw1/{filename}", "r") as f:
        key = get_key(f)
      # with open(f"./music_in_C/{filename}", "w") as f:
      #   pass
    except AssertionError as err:
      print(f"{filename}: {err}")
    except:
      print(f"Unexpected error in {filename}")