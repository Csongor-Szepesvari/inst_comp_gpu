# This files job is to 
#   1. delete everything in the not_started, started, finished folders
#   2. run "python parameter_generator.py" in the command line
#   3. Execute 'git add .', 'git commit -m "reseting"', 'git push'

import os
import shutil
import subprocess
from git import Repo

# Define folder paths
BASE_DIR = os.getcwd()
NOT_STARTED_FOLDER = os.path.join(BASE_DIR, "not_started")
STARTED_FOLDER = os.path.join(BASE_DIR, "started")
FINISHED_FOLDER = os.path.join(BASE_DIR, "finished")

# 1. Delete everything in the folders
for folder in [NOT_STARTED_FOLDER, STARTED_FOLDER, FINISHED_FOLDER]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

# 2. Run parameter_generator.py
subprocess.run(["python", "parameter_generator.py"])

# 3. Execute git commands
repo = Repo(BASE_DIR)
repo.git.add('--all')
repo.index.commit("reseting")
origin = repo.remote(name='origin')
origin.push()

