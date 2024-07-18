import mne
import os
import numpy as np
from scipy.signal import lfilter, filtfilt, triang
from scipy.stats import norm
from hear_ import HEAR
import pickle
import matplotlib.pyplot as plt

## LOAD AND PREPROCESS DATA (in EEG lab format)

def load_and_process_eeg(num_subjs_train, num_subjs_test):
    rest_data = []
    reach_data = []

    for i in range(1, num_subjs_train + 1):
        rest_f = os.path.join(f'all_subs/simrest{i}.set')
        rest_epoch = mne.io.read_epochs_eeglab(rest_f)
        rest_data.append(rest_epoch.get_data())
    
    for j in range(1, num_subjs_test + 1):
        reach_f = os.path.join(f'all_subs/simreach{j}.set')
        reach_epoch = mne.io.read_epochs_eeglab(reach_f)
        reach_data.append(reach_epoch.get_data())

    train_data = concatenate_subjects_data(rest_data)
    test_data = concatenate_subjects_data(reach_data)
    
    return train_data, test_data

def concatenate_subjects_data(subjects_data):
    concatenated_data = []

    for data in subjects_data:
        concatenated_data.append(np.concatenate((data[0,:,:], data[1,:,:], data[2,:,:], data[3,:,:],
                                                 data[4,:,:], data[5,:,:], data[6,:,:], data[7,:,:],
                                                 data[8,:,:], data[9,:,:], data[10,:,:], data[11,:,:]
                                                 ), axis=1))

    return np.concatenate(concatenated_data, axis=1)

## FIRST 7 SIMULATED SUBJECTS
rest1 = mne.io.read_epochs_eeglab('all_subs/simrest1.set')
rest_1 = rest1.get_data()
reach1 = mne.io.read_epochs_eeglab('all_subs/simreach1.set')
reach_1 = reach1.get_data()

#train_data = np.concatenate((rest_1[0,:,:], rest_1[1,:,:], rest_1[2,:,:], rest_1[3,:,:],
#                             rest_1[4,:,:], rest_1[5,:,:], rest_1[6,:,:], rest_1[7,:,:]
#                             ), axis=1)

#test_data = np.concatenate((reach_1[0,:,:], reach_1[1,:,:], reach_1[2,:,:], reach_1[3,:,:],
#                             reach_1[4,:,:], reach_1[5,:,:], reach_1[6,:,:], reach_1[7,:,:],
#                             reach_1[8,:,:], reach_1[9,:,:], reach_1[10,:,:]), axis=1)

num_subjs_train = 15
num_subjs_test = 15
train_data, test_data = load_and_process_eeg(num_subjs_train, num_subjs_test)

print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

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

D = calculate_interpolation_matrix(rest1.info['chs'], 4)
np.set_printoptions(precision=3, floatmode="fixed")
print(D)

## DEFINE AND TRAIN HEAR

fs = 200
hear_mdl = HEAR(fs, D)
hear_mdl.train(data=train_data)

with open('hear_model_15s_8t_ca.pkl', 'wb') as f:
     pickle.dump(hear_mdl, f)


print('HEAR successfully trained.')

## APPLY HEAR

with open('hear_model_15s_8t_bi.pkl', 'rb') as f:
    hear_mdl = pickle.load(f)
    output = hear_mdl.apply(test_data)
    print(output)

p_art = output[0]
p_confidence = output[1]
corrected_data = output[2]
print(len(output[0]), len(output[1]), len(output[2]))

print(corrected_data.shape)
print(corrected_data)

with open('hear_model_15s_8t_ca.pkl', 'rb') as f:
    hear_mdl_c = pickle.load(f)
    output_ = hear_mdl_c.apply(test_data)
    print(output_)

p_art_ = output_[0]
p_confidence_ = output_[1]
corrected_data_c = output_[2]
print(len(output_[0]), len(output_[1]), len(output_[2]))

print(corrected_data_c.shape)

## CALCULATE SNR BETWEEN SIGNALS

"""### Calculate SNR to compare the signals

SNR is defined as the power of the signal to the power of the noise in the same band of frequencies, and measured in the same time interval
"""

import numpy as np

def calculate_snr(clean_signal, noisy_signal):
    # get power of the clean signal
    signal_power = np.mean(clean_signal ** 2)
    # Compute the noise (difference between clean_signal and noisy_signal)
    noise = clean_signal - noisy_signal
    # get power of the noise
    noise_power = np.mean(noise ** 2)
    # Calculate the SNR in linear scale
    snr = signal_power / noise_power
    # Convert the SNR to decibels (dB)
    snr_db = 10 * np.log10(snr)

    return snr, snr_db

snr, snr_db = calculate_snr(corrected_data, test_data)
print(f"HEAR(B) SNR: {snr} SNR(dB): {snr_db}")

snr_c, snr_db_c = calculate_snr(corrected_data_c, test_data)
print(f"HEAR(O) SNR: {snr_c} SNR(dB): {snr_db_c}")

## PLOT SMOOTHED

split_arrays = np.split(corrected_data, 15, axis=1)
split_arrays_c = np.split(corrected_data_c, 15, axis=1)
split_arrays_or = np.split(test_data, 15, axis=1)

# Verify the shapes of the resulting arrays
for i, arr in enumerate(split_arrays):
    print(f"Array {i+1} shape: {arr.shape}")

# take 1 subject to plot
split_7b = np.split(split_arrays[6], 8, axis=1)
split_7c = np.split(split_arrays_c[6], 8, axis=1)
split_7or = np.split(split_arrays_or[6], 8, axis=1)

# Define the frequency band for filtering the signal (optional)
fmin = 0  # Minimum frequency (Hz)
fmax = 40  # Maximum frequency (Hz)

# Apply the bandpass filter
#corrected_data.filter(l_freq=fmin, h_freq=fmax)

# Extract data for channel C1
channel_name = 'C1'
channel_index = reach1.ch_names.index(channel_name)

# Create a time vector in milliseconds
times = reach1.times

# Define the time range for plotting
start_time = 5.5  # seconds
end_time = 9  # seconds

# Find the indices corresponding to the defined time range
start_idx = int(start_time * 200)
end_idx = int(end_time * 200)

c1_data = []
c1_data_c = []
c1_data_or = []

# Slice the data and time vector within the defined time range
for i, arr in enumerate(split_7b):
    channel_data = arr[channel_index, start_idx:end_idx]
    c1_data.append(channel_data)

for i, arr in enumerate(split_7c):
    channel_data_c = arr[channel_index, start_idx:end_idx]
    c1_data_c.append(channel_data_c)    

for i, arr in enumerate(split_7or):
    channel_data_or = arr[channel_index, start_idx:end_idx]
    c1_data_or.append(channel_data_or) 
#print(c1_data)
times = times[start_idx:end_idx]

c1_data = np.array(c1_data)
c1_data_c = np.array(c1_data_c)
c1_data_or = np.array(c1_data_or)

print(c1_data.shape)

# Compute the grand average across all epochs for the sliced data
grand_average = c1_data.mean(axis=0)
grand_average_c = c1_data_c.mean(axis=0)
grand_average_or = c1_data_or.mean(axis=0)

# Smooth the signal using Savitzky-Golay filter
#smoothed_signal = savgol_filter(grand_average, window_length=101, polyorder=3)
#smoothed_signal_b = savgol_filter(grand_average_b, window_length=101, polyorder=3)

# Plot the original and smoothed signals for channel C1 within the defined time range
#plt.figure()
#plt.plot(times, grand_average, label='HEAR(0)')
#plt.plot(times, smoothed_signal_b, label='HEAR(B)')
#plt.title(f"Averaged Signal for Channel {channel_name} after rejection")
#plt.xlabel("Time (ms)")
#plt.ylabel("Amplitude (µV)")
#plt.legend()
#plt.grid(True)
#plt.show()

## TRIANGULAR WINDOW ZERO-PHASE FILTER

# Define the triangular window
window_duration_ms = 100  # 100 ms
window_samples = int(200 * window_duration_ms / 1000)  # Convert window duration to samples
if window_samples % 2 == 0:
    window_samples += 1  # Make sure the window length is odd
triangular_window = triang(window_samples)

# Normalize the triangular window
triangular_window /= triangular_window.sum()

# Apply zero-phase filtering using filtfilt
smoothed_signal = filtfilt(triangular_window, 1, grand_average)
smoothed_signal_c = filtfilt(triangular_window, 1, grand_average_c)
smoothed_signal_or = filtfilt(triangular_window, 1, grand_average_or)

#smoothed_signal = savgol_filter(grand_average, window_length=101, polyorder=3)
#smoothed_signal_c = savgol_filter(grand_average_c, window_length=101, polyorder=3)

# Plot the original and smoothed signals
plt.figure()
plt.plot(times, smoothed_signal, label='HEAR(B)', linewidth=1)
plt.plot(times, smoothed_signal_c, label='HEAR(O)', linewidth=1)
plt.plot(times, smoothed_signal_or, label='original signal', linewidth=1)
plt.title("Signal Smoothing with a 100 ms Triangular Window")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
