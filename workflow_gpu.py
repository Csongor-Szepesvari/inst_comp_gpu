import os
import shutil
import pandas as pd
from git import Repo
import cupy as cp
import time as t
import main_gpu
from multiprocessing import Pool, cpu_count

# Define folders
BASE_DIR = os.getcwd()
NOT_STARTED_FOLDER = os.path.join(BASE_DIR, "not_started")
STARTED_FOLDER = os.path.join(BASE_DIR, "started")
FINISHED_FOLDER = os.path.join(BASE_DIR, "finished")
REPO_DIR = BASE_DIR  # Assumes this script is run from the root of the GitHub repo

# Ensure required folders exist
for folder in [NOT_STARTED_FOLDER, STARTED_FOLDER, FINISHED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Initialize the Git repo
repo = Repo(REPO_DIR)
if repo.bare:
    raise Exception("Not a valid Git repository")

# Ensure 'master' is the active branch
if "master" in repo.heads:
    master_branch = repo.heads["master"]
    master_branch.checkout()
else:
    raise Exception("'master' branch does not exist in the repository.")



def update_git(files):
    """Commit and push changes to Git."""
    origin.pull()
    for src, dst in files:
        repo.index.add([dst])
        repo.index.remove([src])
    repo.index.commit(f"Moved {len(files)} files in batch.")
    
    # Fetch updates from the remote
    origin.fetch()

    # Push changes to the remote
    origin.push(refspec=f"master:master")  # Push local 'master' to remote 'master'

def move_files_in_batch(files):
    """Move a batch of files from their source to their destination."""
    for src, dst in files:
        shutil.move(src, dst)

    update_git(files)

def process_file(file_name):
    """Process a single file: move, process, and finalize."""
    file_path = os.path.join(STARTED_FOLDER, file_name)

    if not file_name.endswith(".csv"):
        print(f"Skipped {file_name}: Not a CSV.")
        return
    print("Currently processing", file_name)
    # Read and process the file
    df = pd.read_csv(file_path)

    start = t.time()

    # Process each row and expand the results into separate columns
    # Apply process_row to each row in the DataFrame
    histogram_dfs = df.apply(main_gpu.process_row, axis=1)
    # Concatenate the resulting histogram DataFrames with the original DataFrame
    df = pd.concat([df, pd.concat(histogram_dfs.tolist(), ignore_index=True)], axis=1)

    # Save the processed file
    finished_path = os.path.join(FINISHED_FOLDER, file_name)
    df.to_csv(finished_path, index=False)

    # now remove this file from the started path
    os.remove(file_path)

    end = t.time()
    print(f"Processed and moved {file_name} to 'finished'. Took {end-start} seconds to finish a file")
    return (file_path, finished_path)


    

if __name__ == "__main__":
    
    while True:
        # Pull the latest changes from the remote repository
        origin = repo.remote(name="origin")
        origin.pull()

        # Get the list of files in the 'not_started' folder (up to 100 files)
        files = os.listdir(NOT_STARTED_FOLDER)[:min(100, len(os.listdir(NOT_STARTED_FOLDER)))]

        if not files:
            print("No more files to process in 'not_started'. Exiting.")
            break

        # Move all selected files to 'started' first in a batch
        batch_moves = []
        for file_name in files:
            src = os.path.join(NOT_STARTED_FOLDER, file_name)
            dst = os.path.join(STARTED_FOLDER, file_name)
            batch_moves.append((src, dst))

        move_files_in_batch(batch_moves)

        # Use multiprocessing Pool to process files concurrently
        num_cores = cpu_count()
        with Pool(num_cores-2) as pool:
            files_batch = pool.map(process_file, files)

        # push the changes to github
        update_git(files_batch)
        