# importing standard python packages
import pandas as pd
import numpy as np
import pywt
from scipy.stats import kurtosis, skew, median_abs_deviation
from scipy import signal
from collections import Counter

import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)
# Suppress UserWarning
warnings.simplefilter(action='ignore', category=UserWarning)

def compute_statistical_features(data):
    """
    Computes statistical features for a given dataset.
    
    Args:
        data (pd.DataFrame): Input data with columns ['X', 'Y', 'Z', 'ENMO'].
        
    Returns:
        pd.DataFrame: A DataFrame containing the computed statistical features.
    """
    stats = {}
    metrics = ['mean', 'median', 'mad', 'std', 'min', 'max', '5Per', '25Per', '75Per', '95Per']
    axes = ['X', 'Y', 'Z', 'ENMO']
    quantiles = [0.05, 0.25, 0.75, 0.95]
    
    for ax in axes:
        stats[f'{ax}_mean'] = data[ax].mean()
        stats[f'{ax}_median'] = data[ax].median()
        stats[f'{ax}_mad'] = median_abs_deviation(data[ax])
        #stats[f'{ax}_std'] = data[ax].std() 
        std_value = np.nanstd(data[ax])
        stats[f'{ax}_std'] = np.where(std_value == 0, 1e-10, std_value)
        stats[f'{ax}_min'] = data[ax].min()
        stats[f'{ax}_max'] = data[ax].max()
        for q, metric in zip(quantiles, metrics[6:]):
            stats[f'{ax}_{metric}'] = data[ax].quantile(q)
        if ax != 'ENMO':
            for ax2 in axes:
                if ax != ax2:
                    stats[f'{ax}{ax2}corr'] = np.nan_to_num(np.corrcoef(data[ax], data[ax2])[0, 1])
        stats[f'kurtosis_{ax}'] = kurtosis(data[ax], axis=0, bias=True)
        stats[f'skew_{ax}'] = skew(data[ax], axis=0, bias=True)
    
    return pd.DataFrame.from_dict(stats, orient='index').T


def compute_inter_quartile_range(data):
    """
    Computes the Inter Quartile Range (IQR) for each axis in the dataset.
    
    Args:
        data (pd.DataFrame): Input data with columns ['X', 'Y', 'Z', 'ENMO'].
        
    Returns:
        pd.DataFrame: A DataFrame containing the IQR for each axis.
    """
    iqr_values = {}
    axes = ['X', 'Y', 'Z', 'ENMO']

    for ax in axes:
        Q1 = np.percentile(data[ax], 25)
        Q3 = np.percentile(data[ax], 75)
        iqr_values[f'IQR_{ax.lower()}'] = Q3 - Q1

    return pd.DataFrame.from_dict(iqr_values, orient='index').T

def compute_time_domain_features(data):
    """
    Computes Root Mean Square (RMS) value for each axis in the dataset.
    
    Args:
        data (pd.DataFrame): Input data with columns ['X', 'Y', 'Z', 'ENMO'].
        
    Returns:
        pd.DataFrame: A DataFrame containing the RMS for each axis.
    """
    rms_values = {}
    axes = ['X', 'Y', 'Z', 'ENMO']

    for ax in axes:
        rms_values[f'{ax}_rms'] = np.sqrt(np.mean(data[ax]**2))

    return pd.DataFrame.from_dict(rms_values, orient='index').T



def compute_signal_magnitude_area(data):
    """
    Computes the Signal Magnitude Area (SMA) for each axis in the dataset.
    
    Args:
        data (pd.DataFrame): Input data with columns ['X', 'Y', 'Z', 'ENMO'].
        
    Returns:
        pd.DataFrame: A DataFrame containing the SMA for each axis.
    """
    sma_values = {}
    axes = ['X', 'Y', 'Z', 'ENMO']

    for ax in axes:
        sma_values[f'{ax}_sma'] = np.sum(np.abs(data[ax]))

    return pd.DataFrame.from_dict(sma_values, orient='index').T

def compute_wavelet_features(data, wavelet_name='db5', level=1):
    """
    Computes wavelet features for each axis in the dataset.
    
    Args:
        data (pd.DataFrame): Input data with columns ['X', 'Y', 'Z', 'ENMO'].
        wavelet_name (str): The wavelet name to be used in the transformation.
        level (int): The level of the wavelet decomposition.
        
    Returns:
        pd.DataFrame: A DataFrame containing wavelet features for each axis.
    """
    def extract_features(A, D, prefix):
        return {
            f'{prefix}mean_Approx_coefficients': np.mean(A),
            f'{prefix}std_Approx_coefficients': np.std(A),
            f'{prefix}mean_Detail_coefficients': np.mean(D),
            f'{prefix}std_Detail_coefficients': np.std(D)
        }

    wavelet_features = {}
    axes = ['X', 'Y', 'Z', 'ENMO']

    for ax in axes:
        coeffs = pywt.wavedec(data[ax], wavelet_name, level=level)
        features = extract_features(coeffs[0], coeffs[1], f'{ax}_')
        wavelet_features.update(features)

    return pd.DataFrame(wavelet_features, index=[0])



def dominant_frequency(data, sample_rate=60):
    """
    Returns the dominant frequency for each axis in the input dataset.
    
    Args:
        data (pd.DataFrame): Input data with columns ['X', 'Y', 'Z', 'ENMO'].
        sample_rate (int): Sampling rate of the data in samples per second (default is 60).
        
    Returns:
        pd.DataFrame: A DataFrame containing the dominant frequency for each axis.
    """
    dom_freq_feature = {}
    axes = ['X', 'Y', 'Z', 'ENMO']

    for ax in axes:
        centered_data = data[ax] - data[ax].mean()
        fft_vals = np.fft.fft(centered_data)
        fft_abs = np.abs(fft_vals[:len(centered_data) // 2])
        freqs = np.fft.fftfreq(len(centered_data), 1.0 / sample_rate)[:len(centered_data) // 2]
        dom_freq_feature[f'dom_freq_{ax}'] = freqs[np.argmax(fft_abs)]

    return pd.DataFrame.from_dict(dom_freq_feature, orient='index').T

def top_frequencies(data, sample_rate=60, top=3):
    """
    Returns the top N frequencies for each axis in the input dataset.
    
    Args:
        data (pd.DataFrame): Input data with columns ['X', 'Y', 'Z', 'ENMO'].
        sample_rate (int): Sampling rate of the data in samples per second (default is 60).
        top (int): Number of top frequencies to return (default is 3).
        
    Returns:
        pd.DataFrame: A DataFrame containing the top N frequencies for each axis.
    """
    top_frequencies = {}
    axes = ['X', 'Y', 'Z', 'ENMO']

    for ax in axes:
        signal = data[ax]
        if len(signal) < top:
            signal = np.pad(signal, (0, top - len(signal)))

        n = len(signal)
        fft_vals = np.fft.fft(signal)
        magnitudes = np.abs(fft_vals[:n // 2])
        frequencies = np.fft.fftfreq(n, 1 / sample_rate)[:n // 2]
        top_indices = np.argpartition(magnitudes, -top)[-top:]
        top_freqs = list(frequencies[top_indices])

        for i, freq in enumerate(top_freqs):
            top_frequencies[f'dom_freq_{ax}{i}'] = freq

    return pd.DataFrame.from_dict(top_frequencies, orient='index').T



def power_spectrum(data, sample_rate=60, nperseg=None):
    """
    Computes the power spectral density for each axis in the input data.
    
    Args:
        data (pd.DataFrame): Input data with columns ['X', 'Y', 'Z', 'ENMO'].
        sample_rate (int): Sampling rate of the data in samples per second (default is 60).
        nperseg (int): Length of each segment (default is None).
        
    Returns:
        pd.DataFrame: A DataFrame containing the mean, std, max, and min of power spectral density for each axis.
    """
    power_features = {}
    axes = ['X', 'Y', 'Z', 'ENMO']

    for ax in axes:
        f, P_den = signal.welch(data[ax], fs=sample_rate, nperseg=nperseg)
        power_features[f'P{ax}_mean'] = np.mean(P_den)
        power_features[f'P{ax}_std'] = np.std(P_den)
        power_features[f'P{ax}_max'] = np.max(P_den)
        power_features[f'P{ax}_min'] = np.min(P_den)

    return pd.DataFrame.from_dict(power_features, orient='index').T

def zero_crossing_rate(data):
    """
    Computes the zero crossing rate for each axis in the input data.
    
    Args:
        data (pd.DataFrame): Input data with columns ['X', 'Y', 'Z', 'ENMO'].
        
    Returns:
        pd.DataFrame: A DataFrame containing the zero crossing rate for each axis.
    """
    zero_crossing_features = {}
    axes = ['X', 'Y', 'Z', 'ENMO']

    for ax in axes:
        sign_changes = np.where(np.diff(np.sign(data[ax])))[0]
        zero_crossing_features[f'zero_crossing_{ax}'] = len(sign_changes)

    return pd.DataFrame.from_dict(zero_crossing_features, orient='index').T

def mean_crossing_rate(data):
    """
    Computes the mean crossing rate for each axis in the input data.
    
    Args:
        data (pd.DataFrame): Input data with columns ['X', 'Y', 'Z', 'ENMO'].
        
    Returns:
        pd.DataFrame: A DataFrame containing the mean crossing rate for each axis.
    """
    mean_crossing_features = {}
    axes = ['X', 'Y', 'Z', 'ENMO']

    for ax in axes:
        mean_value = data[ax].mean()
        sign_changes = np.where(np.diff(np.sign(data[ax] - mean_value)))[0]
        mean_crossing_features[f'mean_crossing_{ax}'] = len(sign_changes)

    return pd.DataFrame.from_dict(mean_crossing_features, orient='index').T

def zero_crossing(data):
    """
    Calculate zero crossing rate and zero crossings count for each axis.

    Args:
        data (pd.DataFrame): Input data with columns ['X', 'Y', 'Z', 'ENMO'].

    Returns:
        pd.DataFrame: A DataFrame containing the zero crossing rate and count for each axis.
    """
    zero_crossing_feature = {}
    
    for axis in ['X', 'Y', 'Z', 'ENMO']:
        sign_changes = np.diff(np.sign(data[axis]))
        zero_crossings = np.count_nonzero(sign_changes)
        zcr = zero_crossings / len(data[axis])
        
        zero_crossing_feature[f'zero_crossing_{axis}'] = zero_crossings
        zero_crossing_feature[f'zero_crossing_rate_{axis}'] = zcr
    
    return pd.DataFrame.from_dict(zero_crossing_feature, orient='index').T

def mean_crossing_rate_not_used(data):
    """
    Calculate mean crossing rate and mean crossings count for each axis.

    Args:
        data (pd.DataFrame): Input data with columns ['X', 'Y', 'Z', 'ENMO'].

    Returns:
        pd.DataFrame: A DataFrame containing the mean crossing rate and count for each axis.
    """
    mean_crossing = {}
    for axis in ['X', 'Y', 'Z', 'ENMO']:    
        signal_mean = data[axis].mean()
        signal_above_mean = data[axis] > signal_mean
        mean_crossings = np.where(np.diff(signal_above_mean))[0]
        mean_crossing_rate = len(mean_crossings) / len(data[axis])
        
        mean_crossing[f'mean_crossing_{axis}'] = len(mean_crossings)
        mean_crossing[f'mean_crossing_rate_{axis}'] = mean_crossing_rate
        
    return pd.DataFrame.from_dict(mean_crossing, orient='index').T

def peak_features(data, sample_rate=60):
    """
    Extracts features of the signal peaks.

    Args:
        data (pd.DataFrame): Input data with columns ['X', 'Y', 'Z', 'ENMO'].
        sample_rate (int): Sampling rate of the data in samples per second (default is 60).

    Returns:
        pd.DataFrame: A DataFrame containing the number of peaks and peak prominence for each axis.
    """
    feats_peak = {}
    axes = ['X', 'Y', 'Z', 'ENMO']
    
    for ax in axes:
        peaks, peak_props = signal.find_peaks(
            data[ax], distance=0.2 * sample_rate, prominence=0.25)
        feats_peak[f'numPeaks_{ax}'] = len(peaks)
        feats_peak[f'peakPromin_{ax}'] = np.median(peak_props['prominences']) if len(peak_props['prominences']) > 0 else 0
        
    return pd.DataFrame.from_dict(feats_peak, orient='index').T



def angular_features(data):
    """
    Calculate angular features (roll, pitch, yaw) from the data.

    Args:
        data (pd.DataFrame): Input data with columns ['X', 'Y', 'Z'].

    Returns:
        pd.DataFrame: A DataFrame containing the average and standard deviation of roll, pitch, and yaw.
    """
    ang_feats = {}

    roll  = np.arctan2(data['Y'], data['Z'])
    pitch = np.arctan2(data['X'], data['Z'])
    yaw   = np.arctan2(data['Y'], data['X'])

    for angle, name in zip([roll, pitch, yaw], ['roll', 'pitch', 'yaw']):
        ang_feats[f'avg{name}'] = np.mean(angle)
        ang_feats[f'sd{name}'] = np.std(angle)
    
    return pd.DataFrame.from_dict(ang_feats, orient='index').T

def entropy(data):
    """
    Calculate entropy for each axis.

    Args:
        data (pd.DataFrame): Input data with columns ['X', 'Y', 'Z', 'ENMO'].

    Returns:
        pd.DataFrame: A DataFrame containing the entropy for each axis.
    """
    entropy_value = {}
    for axis in ['X', 'Y', 'Z', 'ENMO']: 
        counts = Counter(data[axis])
        total_samples = len(data[axis])
        probabilities = [count / total_samples for count in counts.values()]
        entropy_value[f'entropy_{axis}'] = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
    return pd.DataFrame.from_dict(entropy_value, orient='index').T

def calculate_jerk(data, dt=1/60, step=5):
    """
    Calculate jerk features from accelerometer data.

    Args:
        data (pd.DataFrame): Input data with columns ['X', 'Y', 'Z', 'ENMO'].
        dt (float): Time interval between consecutive samples (default is 1/60).
        step (int): Step size for gradient calculation (default is 5).

    Returns:
        pd.DataFrame: A DataFrame containing jerk features for each axis.
    """
    jerk = {}
    for axis in ['X', 'Y', 'Z', 'ENMO']: 
        jerk_temp = np.gradient(data[axis], dt * step)
        jerk[f'jerk_mean_{axis}'] = np.mean(jerk_temp)
        jerk[f'jerk_median_{axis}'] = np.median(jerk_temp)
        jerk[f'jerk_min_{axis}'] = np.min(jerk_temp)
        jerk[f'jerk_max_{axis}'] = np.max(jerk_temp)
        
    return pd.DataFrame.from_dict(jerk, orient='index').T

def extract_feature_all(data):
    """
    Extract a wide range of features from the input sliced data.

    Args:
        sliceddata (pd.DataFrame): Input data with columns ['X', 'Y', 'Z', 'ENMO', 'Selected_memberid', 
        'PRIMARY activity', 'Other SIMULTANEOUS activity/activities', 
        'Doing anything else while you did the PRIMARY activity/activities', 'Start Time', 'Finish Time'].

    Returns:
        pd.DataFrame: A DataFrame containing the extracted features.
    """
    sliceddata = data.reset_index(drop=True)
    data_subset = data[['X', 'Y', 'Z', 'ENMO']]

    features = [
        compute_statistical_features(data_subset),
        compute_inter_quartile_range(data_subset),
        compute_time_domain_features(data_subset),
        compute_signal_magnitude_area(data_subset),
        compute_wavelet_features(data_subset),
        top_frequencies(data_subset),
        power_spectrum(data_subset),
        zero_crossing(data_subset),
        mean_crossing_rate(data_subset),
        peak_features(data_subset),
        angular_features(data_subset),
        entropy(data_subset),
        calculate_jerk(data_subset)
    ]

    feature_all = pd.concat(features, axis=1)
    
    attributes_app = [
        'Selected_memberid',
        'PRIMARY activity',
        'Start Time',
        'Finish Time'
    ]
    
    attributes_recall = [
        'Selected_memberid', 
        'PRIMARY activity',
        'Other SIMULTANEOUS activity/activities',
        'Doing anything else while you did the PRIMARY activity/activities',
        'Start Time',
        'Finish Time'
    ]
    
    for attr in attributes_app:
        feature_all[attr] = sliceddata[attr].iloc[0]

    try:
        feature_all['timeofday'] = (sliceddata.time.dt.hour >= 12).astype(int).iloc[0]
        feature_all['dayofweek'] = sliceddata.time.dt.dayofweek.iloc[0]
    except AttributeError:
        feature_all['timeofday'] = 0
        feature_all['dayofweek'] = 0

    feature_all.columns = feature_all.columns.str.replace(r"ENMO", "E")

    return feature_all
