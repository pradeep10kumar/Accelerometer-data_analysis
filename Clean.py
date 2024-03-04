"""
Created on Wed Jul 5 11:11:08 2023
@author: Pradeep Kumar
"""

from datetime import time

def remove_ten_to_fourthirty(sensor_data):
    """
    Removes the data between 10:00 PM and 4:30 AM.

    Args:
        sensor_data (pd.DataFrame): Raw sensor data.

    Returns:
        pd.DataFrame: Cleaned data.
    """
    sensor_data['Temp Time'] = sensor_data.time.dt.time
    clean_data = sensor_data[(sensor_data['Temp Time'] >= time(4, 30)) & (sensor_data['Temp Time'] <= time(22, 0))]
    clean_data = clean_data.drop(columns=['Temp Time'])

    return clean_data


def remove_sleep_hour(sensor_data_s):
    """
    Removes timestamps with 'Sleeping' as a primary activity.

    Args:
        sensor_data (pd.DataFrame): Raw sensor data.

    Returns:
        pd.DataFrame: Cleaned data.
    """
    sensor_data_s['PRIMARY activity'] = sensor_data_s['PRIMARY activity'].str.strip()
    clean_data_s = sensor_data_s[sensor_data_s['PRIMARY activity'] != '1 Sleeping']
    clean_data_s = clean_data_s[clean_data_s['PRIMARY activity'] != '9']
    return clean_data_s


def remove_if_secondary_activity(sensor_data):
    """
    Removes rows if a secondary activity exists.

    Args:
        sensor_data (pd.DataFrame): Raw sensor data.

    Returns:
        pd.DataFrame: Cleaned data.
    """
    sensor_data['Doing anything else while you did the PRIMARY activity/activities'] = sensor_data['Doing anything else while you did the PRIMARY activity/activities'].str.strip().str.upper()
    clean_data = sensor_data[sensor_data['Doing anything else while you did the PRIMARY activity/activities'] != 'YES']
    return clean_data


def remove_if_other_activity(sensor_data):
    """
    Removes rows if 'Specify the other activity' is nonzero.

    Args:
        sensor_data (pd.DataFrame): Raw sensor data.

    Returns:
        pd.DataFrame: Cleaned data.
    """
    clean_data = sensor_data[sensor_data['Specify the other activity'] == 0]
    return clean_data
