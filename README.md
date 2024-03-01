This Python script is designed to extract a wide range of features from accelerometer data. It imports various standard Python packages and defines several functions to compute statistical, time-domain, wavelet, frequency-domain, and other features from the input accelerometer data. The extracted features are useful for various applications such as activity recognition, health monitoring, and motion analysis.

## Usage

1. **Importing Packages**: The script starts by importing standard Python packages such as pandas, numpy, pywt, scipy, and collections.

2. **Defining Functions**: The script defines the following functions:

    - `compute_statistical_features`: Computes statistical features for each axis of the accelerometer data.
    - `compute_inter_quartile_range`: Computes the Inter Quartile Range (IQR) for each axis.
    - `compute_time_domain_features`: Computes Root Mean Square (RMS) value for each axis.
    - `compute_signal_magnitude_area`: Computes the Signal Magnitude Area (SMA) for each axis.
    - `compute_wavelet_features`: Computes wavelet features for each axis.
    - `dominant_frequency`: Computes the dominant frequency for each axis.
    - `top_frequencies`: Returns the top N frequencies for each axis.
    - `power_spectrum`: Computes the power spectral density for each axis.
    - `zero_crossing_rate`: Computes the zero crossing rate for each axis.
    - `mean_crossing_rate`: Computes the mean crossing rate for each axis.
    - `zero_crossing`: Calculates zero crossing rate and zero crossings count for each axis.
    - `peak_features`: Extracts features of the signal peaks.
    - `angular_features`: Calculates angular features (roll, pitch, yaw) from the data.
    - `entropy`: Calculates entropy for each axis.
    - `calculate_jerk`: Calculates jerk features from accelerometer data.
    - `extract_feature_all`: Extracts a wide range of features from the input accelerometer data.

3. **Function Parameters**: Each function takes the accelerometer data as input and computes specific features. Some functions may have additional parameters such as `sample_rate`, `wavelet_name`, `level`, `dt`, and `step`.

4. **Output**: The output of the `extract_feature_all` function is a DataFrame containing all the extracted features along with additional attributes such as member ID, activity, start time, and finish time.

5. **Attributes Handling**: The script handles additional attributes such as member ID, activity, and timestamp to provide comprehensive feature extraction.

6. **Execution**: To use the script, simply call the `extract_feature_all` function with the accelerometer data as input. Ensure that the input data is in the form of a DataFrame with columns ['X', 'Y', 'Z', 'ENMO'].

## Note

This script provides a comprehensive set of features for accelerometer data analysis. Users can further customize or extend the script based on their specific requirements.

## For Processing the data 

This Python script is designed to process accelerometer data and extract features for activity prediction. It imports standard Python packages such as os, pandas, numpy, and multiprocessing, as well as user-defined modules Clean and Feature_generation.

## Usage

1. **Importing Packages**: The script imports the following standard Python packages:
    - `os`: Provides a portable way of using operating system-dependent functionality.
    - `pandas`: Offers data structures and operations for manipulating numerical tables and time series.
    - `numpy`: Provides support for multi-dimensional arrays and matrices, along with mathematical functions.
    - `multiprocessing`: Offers support for concurrent execution of processes using subprocesses.

2. **Importing User-Defined Modules**: The script imports the following user-defined modules:
    - `Clean`: Contains functions to clean and preprocess the accelerometer data.
    - `Feature_generation`: Contains functions to extract features from the accelerometer data.

3. **Parameters**: The script defines parameters such as `epochtime`, `sampling_frequency`, and `no_timestamp` to control the processing of accelerometer data.

4. **Class Definition**: The `DataProcessor` class contains static methods to perform data processing tasks:
    - `remove_unused_data`: Removes unused data from the accelerometer dataset.
    - `extract_features_from_interval`: Extracts features from intervals of accelerometer data.
    - `process_file`: Processes individual files containing accelerometer data.

5. **Main Function**: The `main` function serves as the entry point of the script. It defines paths for input and output data, retrieves a list of files to process, and processes each file using the `DataProcessor` class.

6. **Execution**: To use the script, simply execute the `main` function. Ensure that the necessary input data files are present in the specified directory.

## Note

This script is designed for processing accelerometer data and extracting features for activity prediction. Users can modify the script as needed to accommodate different data sources or processing requirements.
