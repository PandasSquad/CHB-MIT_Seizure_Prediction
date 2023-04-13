# CHB-MIT Seizure Prediction - Data preparation 🧠 📊

## Description

This repository contains the code for preparing the CHB-MIT Seizure Prediction dataset for a comparative study of different modern Deep Learning techniques to predict the pre-ictal period using EEG data. The study is part of the final project for the Biomedical Engineering degree by Matías N. Sosa and Cristian E. Morilla.

The code in this repository transforms the raw data into a format that can be used for training and testing various Deep Learning models for seizure prediction.

Data preparation includes analysis, feature extraction, feature selection and transformation.

## Installation

1. Clone the repository
2. Install the required packages (see requirements.txt)
3. Edit the data_path.txt file with the local data folder path on your system. You have to download

## Create `data_path.txt` file

To avoid conflicts and ensure that each team member has access to their own data safely, each member can have their own data_path.txt file in their local project folder, which specifies the path to the data folder on their system. The data_path.txt file is added to the .gitignore file, so each team member's file is not uploaded to the repository.

To read the data folder path from the data_path.txt file, each team member edits their own data_path.txt file with the local data folder path on their system. Then, when running the Python code, it reads the data folder path from the local data_path.txt file. This approach allows each team member to specify their own data folder path without affecting the project's code, but it may be a bit more complicated to set up compared to using environment variables.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

🤓 Matías Nicolás Sosa

🤓 Cristian Ezequiel Morilla
