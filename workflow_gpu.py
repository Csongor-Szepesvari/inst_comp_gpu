import os
import shutil
import pandas as pd
from git import Repo
import cupy as cp
import datetime
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

def update_git(files):
    """Commit and push changes to Git."""
    for src, dst in files:
        repo.index.add([dst])
        repo.index.remove([src])
    repo.index.commit(f"Moved {len(files)} files in batch.")
    origin = repo.remote(name="origin")
    origin.pull()
    origin.push()

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

    # Process each row and expand the results into separate columns
    results = df.apply(main_gpu.process_row, axis=1, result_type='expand')
    poly_degree = len(results.iloc[0][0])  # Get the number of polynomial coefficients
    df = pd.concat([
        df,
        results[0].apply(pd.Series).rename(columns=lambda x: f'coef_{x}'),
        results[1].rename('mean'),
        results[2].rename('std')
    ], axis=1)

    # Save the processed file
    finished_path = os.path.join(FINISHED_FOLDER, file_name)
    df.to_csv(finished_path, index=False)

    # now remove this file from the started path
    os.remove(file_path)

    # update tracking in git
    repo.index.add([dst])
    repo.index.remove([src])


    print(f"Processed and moved {file_name} to 'finished'.")

if __name__ == "__main__":
    # Detect GPU availability
    if not cp.cuda.runtime.getDeviceCount():
        raise RuntimeError("No compatible GPU detected. Ensure that CUDA is properly installed and configured.")
    device = cp.cuda.Device()
    print(f"Using GPU: {device.id} - {device.compute_capability}")

    # Get thread information
    attrs = device.attributes
    print(f"Max threads per block: {attrs['MaxThreadsPerBlock']}")
    print(f"Max threads per multiprocessor: {attrs['MaxThreadsPerMultiProcessor']}")
    print(f"Max grid dimensions: {attrs['MaxGridDimX']}, {attrs['MaxGridDimY']}, {attrs['MaxGridDimZ']}")
    #print(attrs.keys())



    while True:
        # Pull the latest changes from the remote repository
        origin = repo.remote(name="origin")
        origin.pull()

        # Get the list of files in the 'not_started' folder (up to 100 files)
        files = os.listdir(NOT_STARTED_FOLDER)[:min(10, len(os.listdir(NOT_STARTED_FOLDER)))]

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
            pool.map(process_file, files)

        # push the changes to github
        repo.index.commit(f"Moved {len(files)} files in batch.")
        origin = repo.remote(name="origin")
        origin.pull()
        origin.push()
        