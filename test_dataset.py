import pathlib
import os
import glob
path = pathlib.Path().resolve()
target_path = os.path.join(path, 'Dataset/**/*.txt')
print(target_path) 
for file in glob.iglob(target_path, recursive=True):
    print(file)
    with open(file) as f:
        lines = f.readlines()
        print(len(lines))