# run `python validator/validate.py --level 2 --coref --lang en *FILE*`
# for all files in a specific folder

import os
import subprocess
import sys

# get folder source from sysargv
folder = sys.argv[1]

# get all files in folder
files = os.listdir(folder)
files = [f for f in files if f.endswith(".conllu")]

# run validate.py for each file
for f in sorted(files):
    lang_code = f.split("_")[0]
    print("Validating file: " + f)
    file_path = os.path.join(folder, f)
    subprocess.run(["python", "src/validator/validate.py", "--level", "2", "--coref", "--lang", lang_code, file_path])
