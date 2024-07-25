import pandas as pd
import numpy as np
from scipy.signal import welch
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load your data
#df = pd.read_csv('original.csv')
#df = pd.read_csv('corrected_ca.csv')
df = pd.read_csv('corrected_bi.csv')

print(df)

######################

# Pivot the DataFrame to have channels as columns
# For each epoch and time point, get values across all channels
data_pivoted = df.pivot_table(index=['epoch', 'time'], columns='channel', values='value')

# Check the pivoted data
print(data_pivoted.head())

# Reshape the data for each epoch
epochs_data = data_pivoted.groupby('epoch').apply(lambda x: x.values).tolist()

# Convert list of arrays to a 3D numpy array (n_epochs, n_times, n_channels)
epochs_data = np.array(epochs_data)

# Get labels
labels = df.drop_duplicates('epoch')['condition'].values

# Feature extraction using temporal features
def extract_features(data):
    mean_features = np.mean(data, axis=1)  # Mean across time
    std_features = np.std(data, axis=1)    # Std across time
    features = np.concatenate([mean_features, std_features], axis=1)
    return features

# Extract features from epochs data
features = extract_features(epochs_data)

# Encode labels to numeric
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)

print("Accuracy: {:.3f}".format(accuracy))
print("Classification Report:\n", report)


######################################

# Feature extraction using FFT and log scale frequencies
""" def extract_features(data, fs=160):
    n_epochs, n_times, n_channels = data.shape
    # FFT parameters
    n_fft = n_times  # Number of points in FFT
    freq_bins = np.fft.fftfreq(n_fft, 1/fs)  # Frequency bins
    
    # Initialize feature matrix
    features = np.zeros((n_epochs, n_channels * (n_fft // 2)))

    for epoch in range(n_epochs):
        for ch in range(n_channels):
            # Compute FFT
            fft_values = np.abs(fft(data[epoch, :, ch]))[:n_fft // 2]  # Magnitude of FFT and positive frequencies
            
            # Convert frequencies to log scale
            log_freqs = np.log1p(freq_bins[:n_fft // 2])  # Log scale frequencies
            
            # Normalize FFT values
            fft_values_normalized = fft_values / np.sum(fft_values)  # Normalize by total power
            
            # Combine log frequencies and FFT values
            features[epoch, ch * (n_fft // 2):(ch + 1) * (n_fft // 2)] = fft_values_normalized

    return features

# Extract features from epochs data
features = extract_features(epochs_data)

# Encode labels to numeric
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
clf = RandomForestClassifier(n_estimators=5, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)

# Print accuracy with 2 decimal places
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)
 """