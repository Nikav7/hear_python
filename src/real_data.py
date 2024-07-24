#import sys
import os
import glob
import scipy.io as sio
from scipy.signal import lfilter, filtfilt, savgol_filter, triang
import numpy as np
import mne
#from mne.viz import plot_topomap
from mne import create_info
import matplotlib.pyplot as plt
import pandas as pd
from hear_ import HEAR
import pickle
import seaborn as sns


#### READ ALL RUNS

# Specify the directory
folder_path = 'subj_9/selected/'

# Pattern to match files named S009R*.edf
edf_pattern = os.path.join(folder_path, 'S009R*.edf')

# List all matching .edf files
edf_files = glob.glob(edf_pattern)
edf_files.sort()  # Ensure the files are sorted in the correct order

# Read and concatenate the Raw objects
raw_list = []

for file in edf_files:
    print(f"Reading {file}")
    raw = mne.io.read_raw_edf(file, preload=True)
    raw_list.append(raw)

# Concatenate all Raw objects
raw_combined = mne.concatenate_raws(raw_list)
print(raw_combined.info['chs'])

# PREPARE TEST DATA

test_data = raw_combined.get_data()
print(test_data.shape)

# Save the combined Raw object to a new .fif file (optional)
#raw_combined.save('combined_raw.fif', overwrite=True)


## PREPARE TRAIN DATA

raw_1 = mne.io.read_raw_edf(f'subj_9/train/S009R01.edf', preload=True)
raw_2 = mne.io.read_raw_edf(f'subj_9/train/S009R02.edf', preload=True)

raw_list_= [raw_1, raw_2]

print(raw_list_)

# Concatenate all Raw objects
train = mne.concatenate_raws(raw_list_)

## SET MONTAGE TO GET ELECTRODES LOCATIONS

std_montage = mne.channels.make_standard_montage("standard_1020")
print(std_montage)
new_names = {
    'Fc5.': 'FC5',
    'Fc3.': 'FC3',
    'Fc1.': 'FC1',
    'Fcz.': 'FCz',
    'Fc2.': 'FC2',
    'Fc4.': 'FC4',
    'Fc6.': 'FC6',
    'C5..': 'C5',
    'C3..': 'C3',
    'C1..': 'C1',
    'Cz..': 'Cz',
    'C2..': 'C2',
    'C4..': 'C4',
    'C6..': 'C6',
    'Cp5.': 'CP5',
    'Cp3.': 'CP3',
    'Cp1.': 'CP1',
    'Cpz.': 'CPz',
    'Cp2.': 'CP2',
    'Cp4.': 'CP4',
    'Cp6.': 'CP6',
    'Fp1.': 'Fp1',
    'Fpz.': 'Fpz',
    'Fp2.': 'Fp2',
    'Af7.': 'AF7',
    'Af3.': 'AF3',
    'Afz.': 'AFz',
    'Af4.': 'AF4',
    'Af8.': 'AF8',
    'F7..': 'F7',
    'F5..': 'F5',
    'F3..': 'F3',
    'F1..': 'F1',
    'Fz..': 'Fz',
    'F2..': 'F2',
    'F4..': 'F4',
    'F6..': 'F6',
    'F8..': 'F8',
    'Ft7.': 'FT7',
    'Ft8.': 'FT8',
    'T7..': 'T7',
    'T8..': 'T8',
    'T9..': 'T9',
    'T10.': 'T10',
    'Tp7.': 'TP7',
    'Tp8.': 'TP8',
    'P7..': 'P7',
    'P5..': 'P5',
    'P3..': 'P3',
    'P1..': 'P1',
    'Pz..': 'Pz',
    'P2..': 'P2',
    'P4..': 'P4',
    'P6..': 'P6',
    'P8..': 'P8',
    'Po7.': 'PO7',
    'Po3.': 'PO3',
    'Poz.': 'POz',
    'Po4.': 'PO4',
    'Po8.': 'PO8',
    'O1..': 'O1',
    'Oz..': 'Oz',
    'O2..': 'O2',
    'Iz..': 'Iz'
}

mne.rename_channels(train.info, new_names, allow_duplicates=False)

train.set_montage(std_montage, on_missing='ignore')

## COMPUTE INTERPOLATION MATRIX OF RELATIVE DISTANCES D

def calculate_interpolation_matrix(chanlocs, k):
  ch_names = [chan['ch_name'] for chan in chanlocs]
  n_channels = len(ch_names)
  allchans = np.arange(n_channels)
  D = np.zeros((n_channels, n_channels))

  # Compute D
  for i in range(n_channels):
      for j in range(i + 1, n_channels):
          # Compute the Euclidean distances between all electrode locations
          loc_i = np.array(chanlocs[i]['loc'][:3])
          loc_j = np.array(chanlocs[j]['loc'][:3])
          dist = np.linalg.norm(loc_i - loc_j)
          #D[i, i] = np.inf
          #D[j, j] = np.inf
          #dist_idxs = np.argsort(D[i, :])
          #neighbor_chan_idxs = dist_idxs[:k]

          #D[i, np.setdiff1d(allchans, neighbor_chan_idxs)] = np.inf

          # Keep only the closest channels (k neighbors)
          if dist < k:
              # convert absolute distances to relative distances
            invdist = 1.0 / dist
            D[i, j] = invdist
            D[j, i] = invdist

  # Normalize the rows of D
  row_sums = D.sum(axis=1)
  D = D / row_sums[:, np.newaxis]

  return D

#print(train.info['chs'])

D = calculate_interpolation_matrix(raw_1.info['chs'], 4)
np.set_printoptions(precision=3, floatmode="fixed")
print(D)

## DEFINE AND TRAIN HEAR

train_data = train.get_data()
print(train_data.shape)

fs = 160
hear_mdl = HEAR(fs, D)
hear_mdl.train(data=train_data)

with open('hear_model_r.pkl', 'wb') as f:
     pickle.dump(hear_mdl, f)


print('HEAR successfully trained.')

## APPLY HEAR

with open('hear_model_r.pkl', 'rb') as f:
    hear_mdl = pickle.load(f)
    output = hear_mdl.apply(test_data)
    print(output)

p_art = output[0]
p_confidence = output[1]
corrected_data = output[2]
#print(output[0]), len(output[1]), len(output[2]))

print(corrected_data.shape)
print(corrected_data)

## SNR

def calculate_snr(clean_signal, noisy_signal):
    # get power of the clean signal
    signal_power = np.mean(clean_signal ** 2)
    # Compute the noise (difference between clean_signal and noisy_signal)
    noise = noisy_signal - clean_signal
    # get power of the noise
    noise_power = np.mean(noise ** 2)
    # Calculate the SNR in linear scale
    snr = signal_power / noise_power
    # Convert the SNR to decibels (dB)
    snr_db = 10 * np.log10(snr)

    return snr, snr_db

snr, snr_db = calculate_snr(corrected_data, test_data)
print(f"SNR: {snr} SNR(dB): {snr_db}")


## FILTER AND EPOCH THE DATA

# SELECTED DATA (imagine opening and closing left or right fist), RUNS 4,8,12

fmin = 0  # Minimum frequency (Hz)
fmax = 40  # Maximum frequency (Hz)

# Apply the bandpass filter
raw_combined.filter(l_freq=fmin, h_freq=fmax)

# Print the combined Raw object info
print(raw_combined)
print(raw_combined.ch_names)
event_dict = {
    "T0": 1,
    "T1": 2,
    "T2": 3
}
eventz = mne.events_from_annotations(raw_combined)

print(eventz)

array_eventz = eventz[0]

# Check the shape to confirm it's (n_events, 3)
print(array_eventz.shape)

#mne.find_events(raw_combined, stim_channel=None,)

mne.viz.plot_events(array_eventz, sfreq=160, first_samp=0, color=None, event_id=event_dict, axes=None, equal_spacing=True, show=True, on_missing='raise', verbose=None)


epochs = mne.Epochs(
    raw_combined,
    array_eventz,
    event_id=event_dict,
    tmin=-0.2,
    tmax=2.0,
    preload=True,
)

print(epochs)

left_right_fist = mne.pick_events(array_eventz, include=[2, 3])

## !!! convert epochs to df to plot

df = epochs.to_data_frame()
print(df)

#epochs.plot(
#    events=left_right_fist,
#    event_id=event_dict,
    #event_color=dict(left="red", right="blue"),
#)

#epochs["T1"].compute_psd().plot(picks="eeg", exclude="bads", amplitude=False)
#epochs["T2"].compute_psd().plot(picks="eeg", exclude="bads", amplitude=False)

#epochs['T1'].plot_psd(picks='eeg')

#epochs["T1"].plot_image(picks=["C1..", "Cz..", "C2.."])


long_df = epochs.to_data_frame(time_format=None, index="condition", long_format=True)
print(long_df)

plt.figure()
channels = ["C1..", "Cz.."]
data = long_df.loc["T1"].query("channel in @channels")
# convert channel column (CategoryDtype → string; for a nicer-looking legend)
data["channel"] = data["channel"].astype(str)
data.reset_index(drop=True, inplace=True)  # speeds things up
sns.lineplot(x="time", y="value", hue="channel", data=data)
plt.show()
