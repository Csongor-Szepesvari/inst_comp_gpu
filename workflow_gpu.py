import os
import shutil
import pandas as pd
from git import Repo
import cupy as cp
import datetime
from main_gpu import process_row

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

def process_file_on_individual_cuda_cores(file_name, poly_degree=20):
    """Process a single file using independent CUDA cores for each task."""
    file_path = os.path.join(STARTED_FOLDER, file_name)

    if not file_name.endswith(".csv"):
        print(f"Skipped {file_name}: Not a CSV.")
        return

    # Read the file into a DataFrame
    df = pd.read_csv(file_path)

    # Move data to GPU
    print("Processing file on individual CUDA cores: ", file_name)
    df_gpu = cp.array(df.to_numpy(), dtype=object)
    num_rows = df_gpu.shape[0]

    # Prepare CUDA streams for independent calculations
    streams = [cp.cuda.Stream() for _ in range(num_rows)]
    results = cp.zeros((num_rows, poly_degree+3), dtype=cp.float32)  # For storing the polynomial coefficients + "underdog_mean" and "underdog_variance"

    # Launch tasks on different CUDA streams
    for i in range(num_rows):
        with streams[i]:
            row = df_gpu[i]
            result = process_row(pd.Series(row.tolist()), poly_degree=poly_degree)  # Use the existing row processing logic
            for j in range(poly_degree+1):
                results[i, j] = result[0][j]
            
            results[i, poly_degree+1] = result[1]  # underdog_mean
            results[i, poly_degree+2] = result[2]  # underdog_std

    # Synchronize all streams
    cp.cuda.Stream.null.synchronize()

    # Extract results back to CPU and store in DataFrame
    for i in range(len(poly_degree+1)):
        df[str(i)] = cp.asnumpy(results[:])
    df["underdog_mean"] = cp.asnumpy(results[:, 1])
    df["underdog_variance"] = cp.asnumpy(results[:, 1])

    # Save the processed file
    finished_path = os.path.join(FINISHED_FOLDER, file_name)
    df.to_csv(finished_path, index=False)
    print(f"Processed {file_name} and saved to 'finished'.")
    os.remove(file_path)
    return file_path, finished_path

if __name__ == "__main__":
    # Detect GPU availability
    if not cp.cuda.runtime.getDeviceCount():
        raise RuntimeError("No compatible GPU detected. Ensure that CUDA is properly installed and configured.")
    print(f"Using GPU: {cp.cuda.Device().name}")

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

        # Process files using independent CUDA cores
        now = datetime.datetime.now()
        print("Starting processing batch at:", now.time())
        for file_name in files:
            process_file_on_individual_cuda_cores(file_name)

        print("Finished processing batch of files. Updating Git.")
