import numpy as np

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