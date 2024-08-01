"""Armo espectrogramas a partir de ventanas de 30 segundos en formato parquet."""

import os
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt, iirnotch, spectrogram
import matplotlib.pyplot as plt
import glob


import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt, iirnotch, spectrogram
import matplotlib.pyplot as plt
import glob


a_30 = os.listdir("windows_per_parquet_30")


sampling_rate = 256  # Hz
highpass_cutoff = 0.1  # Hz
notch_freq1 = 60.0  # Frecuencia central del notch para 57-63 Hz
notch_bandwidth1 = 6.0  # Ancho de banda del notch (Hz)
notch_freq2 = 120.0  # Frecuencia central del notch para 117-123 Hz
notch_bandwidth2 = 6.0  # Ancho de banda del notch (Hz)

def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def notch_filter(freq, fs, bandwidth):
    Q = freq / bandwidth
    b, a = iirnotch(freq, Q, fs)
    return b, a

# Crear filtros
b_high, a_high = butter_highpass(highpass_cutoff, sampling_rate, order=5)
b_notch1, a_notch1 = notch_filter(notch_freq1, sampling_rate, notch_bandwidth1)
b_notch2, a_notch2 = notch_filter(notch_freq2, sampling_rate, notch_bandwidth2)

input_dir = "windows_per_parquet_30"
output_dir = "spectrograms_all"
os.makedirs(output_dir, exist_ok=True)

def process_file(file_path):
    # Filtrado de la señal
    df = pd.read_parquet(file_path)
    df_filtered = df.apply(lambda x: filtfilt(b_high, a_high, x))
    df_filtered = df_filtered.apply(lambda x: filtfilt(b_notch1, a_notch1, x))
    df_filtered = df_filtered.apply(lambda x: filtfilt(b_notch2, a_notch2, x))

    for column in df_filtered.columns:
        f, t, Sxx = spectrogram(df_filtered[column], fs=sampling_rate, nperseg=256*8, noverlap=256, window='hamming')
        
        # Guardar el espectrograma como imagen
        plt.figure(figsize=(10, 5))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        plt.axis('off')
        plt.tight_layout(pad=0)
        image_filename = os.path.join(output_dir, f"{os.path.basename(file_path).replace('.parquet', '')}_{column}.png")
        plt.savefig(image_filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Guardar los datos numéricos
        data_filename = os.path.join(output_dir, f"{os.path.basename(file_path).replace('.parquet', '')}_{column}.npz")
        np.savez(data_filename, t=t, f=f, Sxx_log=10 * np.log10(Sxx))


input_dir = "windows_per_parquet_30"
output_dir = "spectrograms_all"
os.makedirs(output_dir, exist_ok=True)

# Definiciones de los filtros
sampling_rate = 256
highpass_cutoff = 0.1
notch_freq1 = 60.0
notch_bandwidth1 = 6.0
notch_freq2 = 120.0
notch_bandwidth2 = 6.0

b_high, a_high = butter_highpass(highpass_cutoff, sampling_rate, order=5)
b_notch1, a_notch1 = notch_filter(notch_freq1, sampling_rate, notch_bandwidth1)
b_notch2, a_notch2 = notch_filter(notch_freq2, sampling_rate, notch_bandwidth2)

file_paths = glob.glob(os.path.join(input_dir, "*.parquet"))

print("Procesos:", os.cpu_count())

with Pool(processes=os.cpu_count()) as pool:
    list(tqdm(pool.imap_unordered(process_file, file_paths), total=len(file_paths)))
