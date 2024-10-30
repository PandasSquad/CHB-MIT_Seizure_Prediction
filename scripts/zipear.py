import shutil
import os

folders_to_zip = ["spectrograms_all", "normalized_spectrograms"]

for folder in folders_to_zip:
    zip_filename = f"{folder}.zip"
    
    shutil.make_archive(folder, 'zip', folder)
    print(f"{zip_filename} creado con éxito.")
