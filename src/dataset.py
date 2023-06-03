import os
import pandas as pd
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Custom dataset class."""

    def __init__(self, folder_path: str):
        """
        Initialize the dataset.

        :param folder_path: Path to the folder with the data files
        """
        self.folder_path = folder_path
        self.label_mapping = {
            "preictal": 0,
            "ictal": 1,
            "prepreictal": 2,
            "interictal": 3,
        }
        self.file_paths = []
        self.labels = []
        self._load_file_paths()

    def __len__(self) -> int:
        """
        Get the total number of samples in the dataset.

        :return: Total number of samples
        """
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a sample by index.

        :param idx: Index
        :return: Sample and label
        """
        file_path = self.file_paths[idx]
        data = pd.read_csv(file_path)
        return data.values, self.labels[idx]

    def _load_file_paths(self) -> None:
        """
        Load all the file paths and their labels.

        :return: None
        """
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(self.folder_path, filename)
                label = filename.split("_")[2]
                if label in self.label_mapping:
                    self.file_paths.append(file_path)
                    self.labels.append(self.label_mapping[label])
                else:
                    print(f"Ignoring file {file_path} with unexpected label {label}")
