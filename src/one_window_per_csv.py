"""
Script to generate just one window per csv file.
The data generated is saved in the folder data/one_window_per_csv, and uses data/windows as input.

It's supposed to be run from the root folder of the project, so the path to the data is correct.

python src/one_window_per_csv.py
"""
import os
import pandas as pd
from tqdm import tqdm

# Path to the folder where the data is stored
windows_path = "data/windows"

# Path to the folder where the data will be saved
one_window_per_csv_path = "data/one_window_per_csv"

# Create the folder if it doesn't exist
if not os.path.exists(one_window_per_csv_path):
    os.makedirs(one_window_per_csv_path)

# Get the list of files in the folder
win_files = os.listdir(windows_path)

ictal_files = [f for f in win_files if f.split("_")[1] == "ictal.csv"]
interictal_files = [f for f in win_files if f.split("_")[1] == "interictal.csv"]
preictal_files = [f for f in win_files if f.split("_")[1] == "preictal.csv"]
prepreictal_files = [f for f in win_files if f.split("_")[1] == "prepreictal.csv"]


def make_one_window(files: list, name: str) -> None:
    """Makes one csv file per window, for the given list of files."""
    for file in files:
        print(f"Processing {file}")
        df = pd.read_csv(os.path.join(windows_path, file))
        edf_files = df["edf_file"].unique()
        for edf_file in tqdm(edf_files):
            file_name = edf_file[:-4] + "_" + name + ".csv"
            df[df["edf_file"] == edf_file].drop("edf_file", axis=1).to_csv(
                os.path.join(one_window_per_csv_path, file_name), index=False
            )
        print()


make_one_window(ictal_files, "ictal")
make_one_window(interictal_files, "interictal")
make_one_window(preictal_files, "preictal")
make_one_window(prepreictal_files, "prepreictal")
