# importing standard python packages
import os
import pandas as pd
import numpy as np
import multiprocessing as mp
# importing user defined modules
import Clean     
from Feature_generation import extract_feature_all
#from joblib import Parallel, delayed

#%%

# parameters

epochtime           = 30  # segment selection for activity prediction
sampling_frequency  = 60     # 60 data points collected in one seconds
no_timestamp        = epochtime*sampling_frequency


#%%
class DataProcessor:

    @staticmethod
    def remove_unused_data(data):
        data = Clean.remove_ten_to_fourthirty(data)
        data = Clean.remove_sleep_hour(data)
        data = data.reset_index(drop= True)
        return data

    @staticmethod
    def extract_features_from_interval(activity_data, start_time, interval_length, feature_store):
        end_time = start_time + interval_length

        while start_time <= activity_data['time'].max():
            interval_df = activity_data[(activity_data['time'] >= start_time) & (activity_data['time'] < end_time)]
            
            if not interval_df.empty: 
                result = extract_feature_all(interval_df)
                result['epoch_start_time'] = start_time
                result['epoch_end_time'] = end_time
                result['no_data_points']   = len(interval_df)

                feature_store = pd.concat([result, feature_store], axis=0)
                start_time = end_time
                end_time = end_time+interval_length
            else:
                start_time = end_time
                end_time = end_time+interval_length
        
        return feature_store

    @staticmethod
    def process_file(filepath, feature_store, epochtime, Selected_memberid):
        print("Started processing file {}".format(filepath))
        tempdata = pd.read_pickle(filepath)
        #tempdata = tempdata[:5000]
        tempdata['Selected_memberid']= Selected_memberid
        tempdata.rename(columns={
            'Activity ID': 'PRIMARY activity',
            'Start time': 'Start Time',
            'Finish time': 'Finish Time'
        }, inplace=True)
        tempdata = DataProcessor.remove_unused_data(tempdata)
        tempdata = tempdata.sort_values(by='time', ascending=True).reset_index(drop=True)
        tempdata['ENMO'] = np.sqrt(tempdata.X**2 + tempdata.Y**2 + tempdata.Z**2) - 1
        
        if tempdata.empty:
            return feature_store
        
        tempdata['date_only'] = tempdata['time'].dt.date
        unique_dates = tempdata['date_only'].unique()

        for selected_date in unique_dates:
            tempdata_filtered = tempdata[tempdata['date_only'] == selected_date]
            activity_list = list(set(tempdata_filtered['PRIMARY activity']))
            
            for activity in activity_list:     
                activity_data = tempdata_filtered[tempdata_filtered['PRIMARY activity'] == activity]
                activity_data = activity_data.sort_values(by='time', ascending=True).reset_index(drop=True)
                
                feature_store = DataProcessor.extract_features_from_interval(activity_data, activity_data['time'].min(), pd.Timedelta(seconds=epochtime), feature_store)
        
        return feature_store


#%%


def main():
    src_path = "H://World_Bank_2023//Trainingdata//"
    processed_path = "G://DS_data_repository//8495_World_Bank_Malawi_vervolg//Appbased_TrainData//train//"
    background_info_path = "G://DS_data_repository//8495_World_Bank_Malawi_vervolg//Data//background_info//background_info.xlsx"
    
    filelist = set(os.listdir(src_path))
    processed_files = os.listdir(processed_path)
    processed_files = [file.replace('.pklSubjects_epochs30','') for file in processed_files]
    filelist -= set(processed_files)
    filelist = list(filelist)

    # Prepare the arguments for the map function
    #args = [(os.path.join(src_path, file), pd.DataFrame(), epochtime, file.split('.')[0]) for file in filelist]

    # Using Pool.map for parallel processing of files
    # with mp.Pool() as pool:
    #     #file_results = pool.starmap(DataProcessor.process_file, args)
    #     file_results = Parallel(n_jobs=1)(delayed(DataProcessor.process_file)(*arg) for arg in args)

    # pool.close()
    # pool.join()
    
    for file in filelist:
        file_results = DataProcessor.process_file((os.path.join(src_path, file)), pd.DataFrame(), epochtime, file.split('.')[0])
        
        background_data = pd.read_excel(background_info_path, index_col=None)
        
        if file.split('.')[0] in background_data.Selected_memberid:
            master_feature_store = pd.merge(file_results, background_data, on='Selected_memberid')
        else:
            master_feature_store = file_results
            print("Background information is missing")
       
        master_feature_store.to_pickle(os.path.join(processed_path, f"{file}Subjects_epochs30.pkl"))

if __name__ == "__main__":
   #if mp.get_start_method(allow_none=True) is None:
   #    mp.set_start_method('spawn')
   main()


