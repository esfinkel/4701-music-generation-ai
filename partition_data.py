import os
import random
import shutil

for filename in os.listdir("./music_in_C/"):
    if random.random() > 0.9:
        shutil.copy(f"./music_in_C/{filename}", f"./music_in_C_test/{filename}")
    else:
        shutil.copy(f"./music_in_C/{filename}", f"./music_in_C_training/{filename}")
    