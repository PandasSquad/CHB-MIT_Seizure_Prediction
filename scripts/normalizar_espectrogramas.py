from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import glob

def normalize_and_save(npz_file):
    data = np.load(npz_file)
    t = data['t']
    f = data['f']
    Sxx_log = data['Sxx_log']

    mean_Sxx_log = np.mean(Sxx_log, axis=1, keepdims=True)
    Sxx_log_normalized = Sxx_log - mean_Sxx_log

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t, f, Sxx_log_normalized, shading='gouraud')
    plt.axis('off')
    plt.tight_layout(pad=0)

    output_filename = os.path.join(output_dir, os.path.basename(npz_file).replace('.npz', '_normalized.png'))
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    plt.close()

input_dir = "spectrograms_all"
output_dir = "normalized_spectrograms"
os.makedirs(output_dir, exist_ok=True)

npz_files = glob.glob(os.path.join(input_dir, "*.npz"))

if __name__ == "__main__":
    with Pool(processes=os.cpu_count()) as pool:
        list(tqdm(pool.imap(normalize_and_save, npz_files), total=len(npz_files)))
