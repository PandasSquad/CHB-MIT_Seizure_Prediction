"""
Dataset utils. Reading CSV files, labels mapping, etc.
"""
import os
import pandas as pd
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Custom dataset class."""

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.label_mapping = {
            "preictal": 0,
            "ictal": 1,
            "prepreictal": 2,
            "interictal": 3,
        }
        self.data = []
        self.labels = []
        self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def _load_data(self):
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(self.folder_path, filename)
                # Get label from filename (preictal, ictal, etc.)
                label = filename.split("_")[2]
                if label in self.label_mapping:
                    # Load CSV file
                    data = pd.read_csv(file_path)
                    # Check dimensions
                    if data.shape == (15360, 22):
                        self.data.append(data.values)
                        self.labels.append(self.label_mapping[label])
                    else:
                        print(
                            f"Ignoring file {file_path} with unexpected dimensions {data.shape}"
                        )
