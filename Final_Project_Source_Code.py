#!/usr/bin/env python
# coding: utf-8

# # ENGR-E599
# # Final Project: Analysing physiological signals to detect deception

# ### Importing required libraries

# In[ ]:


# Import library
import glob
import numpy as np
import pandas as pd
import scipy.io
import re
import random
import neurokit2 as nk
from scipy.signal import find_peaks
import scipy.signal as signal
from detecta import detect_peaks
from numpy.lib.recfunctions import append_fields
import pandas as pd
from statsmodels.stats.anova import AnovaRM
import matplotlib.pyplot as plt


# ### Plotting ECG data

# In[ ]:


# fig, ax = plt.subplots()

# # Plot the ECG data
# ax.plot(ECG_data)

# # Set the title and axis labels
# ax.set_title('ECG Plot')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Amplitude (mV)')
# ax.set_yticklabels([])

# # Show the plot
# plt.show()


# ### Creating a dataframe for ANOVA

# In[ ]:


def create_df(num_blocks, num_subjects, blocks_list):

#     num_blocks = 3
#     num_subjects = 22
    subject_list = [None]*num_subjects*num_blocks

    idx_sub = 0
    idx_blk = 0
    sub = 1

    while idx_sub < num_subjects*num_blocks:
        for idx_blk in range(num_blocks):
            subject_list[idx_sub] = sub
            idx_sub += 1
        sub += 1

    blocks = blocks_list*num_subjects

    data_df = {'subject': subject_list,
            'block':   blocks,
            'mean_IBI': [None]*num_blocks*num_subjects,
            'mean_EDA': [None]*num_blocks*num_subjects}

    df = pd.DataFrame(data_df)
    
    return df


# ### Helper functions for extracting features

# In[ ]:


b, a = signal.butter(4, [0.1, 0.15], btype='band', analog=True)
w, h = signal.freqs(b, a)
plt.semilogx(w, 20 * np.log10(abs(h)))
plt.show()


# In[ ]:


def filter_resp_signal(rsp_signal):

    # Design the filter
    low = 0.1
    high = 0.5
    b,a = signal.butter(4, [low, high], btype='band', analog='True')

    # Apply filter to respiratory signal
    filtered_rsp_signal = signal.filtfilt(b, a, rsp_signal)
    
    return filtered_rsp_signal


# In[ ]:


def ECG_features(ecg_data):
    
    signals, info = nk.ecg_process(ecg_data, sampling_rate=1000)
    clean_ECG = signals['ECG_Clean']
    
    peaks = signals[(signals['ECG_R_Peaks'] == 1)].index
    IBIs = np.diff(peaks)
    
    return round(np.mean(IBIs),2)
    
    
def RSP_features(resp_data):
    
    signals, info = nk.rsp_process(resp_data, sampling_rate=1000)
    
    clean_resp_data = signals['RSP_Clean']
    
    filtered_data = filter_resp_signal(clean_resp_data)

    # Detecting peaks in the signal
    peaks, _ = find_peaks(filtered_data, height=0)

    # Calculating respiration rate
    duration = len(resp_data) / 1000
    respiration_rate = len(peaks) / duration * 60

    # Calculating IE ratio
    inhale_durations = np.diff(peaks[:-1])
    exhale_durations = np.diff(peaks[1:])
    ie_ratio = inhale_durations / exhale_durations
    
    return round(respiration_rate,2), round(np.mean(ie_ratio),2)

def EDA_features(eda_data):
    
    signals, info = nk.eda_process(eda_data, sampling_rate=1000)
    clean_EDA = signals['EDA_Clean']
    
    highpass = nk.eda_phasic(eda_data, method='highpass')
    EDA_Phasic = highpass['EDA_Phasic']
    
    analyze_df = nk.eda_analyze(signals, sampling_rate=1000)
    
    return np.mean(eda_data), np.mean(EDA_Phasic), analyze_df['SCR_Peaks_Amplitude_Mean']


# In[ ]:


# import neurokit2 as nk

# # Simulate EDA signal
# eda_signal = nk.eda_simulate(duration=100, scr_number=5, drift=0.1)

# # # Decompose using different algorithms
# # # cvxEDA = nk.eda_phasic(eda_signal, method='cvxeda')
# # smoothMedian = nk.eda_phasic(eda_signal, method='smoothmedian')

# # highpass = nk.eda_phasic(eda_signal, method='highpass')

# # # print(highpass['EDA_Phasic'])

# df, info = nk.eda_process(eda_signal, sampling_rate=1000)
# analyze_df = nk.eda_analyze(df, sampling_rate=1000)

# print(analyze_df['SCR_Peaks_Amplitude_Mean'])


# ### Extracting ECG, EDA and Respiratory features features

# In[ ]:


def write_data_to_df(data, blocks, block_start_time):
    
    plot_data = []
    plot_legend = []
    
    for block in blocks:
        start_time = block_start_time[block-1]+30000
        end_time = block_start_time[block]

        block_data = data[start_time:end_time]
        block_ECG = block_data[:,3]
        
#         mean_IBI = ECG_features(block_ECG)
        
        # Cleaning the ECG data
        signals, info = nk.ecg_process(block_ECG, sampling_rate=1000)
        clean_ECG = signals['ECG_Clean']
        
        plt.plot(block_ECG, clean_ECG)
        plt.clf()

        # Detecting R peaks and calculating IBIs
        peaks = signals[(signals['ECG_R_Peaks'] == 1)].index
        IBIs = np.diff(peaks)
#         print(IBIs)
        plt.hist(IBIs)
        plt.savefig("/Users/saikrishna/Downloads/Physiological_Time_Series/Final Project/Plots/IBI_histogram_" + str(subject_number) + "_" + str(block) + ".png")
        plt.clf()
#         print(IBIs)

        
        
        block_EDA = block_data[:,2]
        signals, info = nk.eda_process(block_EDA, sampling_rate=1000)
        clean_EDA = signals['EDA_Clean']
        
#         block_resp = block_data[:,1]
#         filtered_data = filter_resp_signal(block_resp)
        
#         # Detect peaks in the signal
#         peaks, _ = find_peaks(filtered_data, height=0)

#         # Calculate respiration rate
#         duration = len(block_data) / 1000 # sampling_rate is the frequency of the signal
#         respiration_rate = len(peaks) / duration * 60
        
#         # Calculating IE ratio
#         inhale_durations = np.diff(peaks[:-1]) # Durations of inhalation phases
#         exhale_durations = np.diff(peaks[1:]) # Durations of exhalation phases (exclude first peak)
#         ie_ratio = inhale_durations / exhale_durations

#         final_df.loc[(final_df.subject == subject_number) & (final_df.block == block),"mean_IBI"] = int(np.mean(IBIs))
#         final_df.loc[(final_df.subject == subject_number) & (final_df.block == block),"mean_EDA"] = round(np.mean(block_EDA),2)
#         final_df.loc[(final_df.subject == subject_number) & (final_df.block == block),"mean_ResRate"] = respiration_rate
#         final_df.loc[(final_df.subject == subject_number) & (final_df.block == block),"mean_IERatio"] = np.mean(ie_ratio)
    
        plot_data.append(clean_EDA)
        plot_legend.append(block)

    # Plot the dataframe
    plt.boxplot(plot_data)
    plt.xticks([1, 2, 3], plot_legend)
    plt.xlabel('Block number')
    plt.ylabel('Cleaned EDA')

    plt.savefig("/Users/saikrishna/Downloads/Physiological_Time_Series/Final Project/Plots/Boxplot_for_" + str(subject_number) + ".png")
    plt.clf()
    
    return


# In[ ]:


def write_data_to_df(data, blocks, block_start_time):
    
    plot_data = []
    plot_legend = []
    
    for block in blocks:
        
        # Only task period in the block is considered (+ 30 sec or 30k ms)
        start_time = block_start_time[block-1]+30000
        end_time = block_start_time[block]

        block_data = data[start_time:end_time]
        
        # Extracting features
        mean_IBI = ECG_features(block_data[:,3])
        respiration_rate, mean_IE_ratio = RSP_features(block_data[:,1])
        mean_EDA, mean_phasic_EDA, mean_SCR_Peaks_Amplitude = EDA_features(block_data[:,2])
        
        # Writing the measures to final dataframe
        final_df.loc[(final_df.subject == subject_number) & (final_df.block == block),"mean_IBI"] = mean_IBI
        final_df.loc[(final_df.subject == subject_number) & (final_df.block == block),"mean_EDA"] = mean_EDA
        final_df.loc[(final_df.subject == subject_number) & (final_df.block == block),"mean_phasic_EDA"] = mean_phasic_EDA
        final_df.loc[(final_df.subject == subject_number) & (final_df.block == block),"mean_SCR_Peaks_Amplitude"] = mean_SCR_Peaks_Amplitude
        final_df.loc[(final_df.subject == subject_number) & (final_df.block == block),"mean_ResRate"] = respiration_rate
        final_df.loc[(final_df.subject == subject_number) & (final_df.block == block),"mean_IERatio"] = mean_IE_ratio
    
    return


# ### Building the ANOVA table

# In[ ]:


file_list = glob.glob("/Users/saikrishna/Downloads/Physiological_Time_Series/Final Project/New Physio_Events records/*")

blocks_list = [4,5,7]
num_subjects = 29
    
final_df = create_df(len(blocks_list), num_subjects, blocks_list)

# print(final_df)
    
for file in file_list:
    
    mat = scipy.io.loadmat(file)['bp_data'][0][0]

    file_name = file.split('/')[-1]
    pattern = r'\d+'
    match = re.search(pattern, file_name)
    subject_number = int(match.group())

    file_data = mat['data']
    
    file_block_start_time = []
    
    for i in mat['BLOCK_START']:
        file_block_start_time.append(int(i[0]))
        
    
    file_block_start_time.insert(0,0)
    file_block_start_time.append(len(file_data))
    
    write_data_to_df(file_data, blocks_list, file_block_start_time)


# In[ ]:


print(final_df.to_markdown())

# print(final_df.groupby(['subject']).mean())


# In[ ]:


# baseline = final_df.loc[final_df["block"] == 5].groupby("subject")["mean_IBI"].mean()
# final_df["mean_ibi_diff"] = final_df.apply(lambda x: x["mean_IBI"] - baseline[x["subject"]], axis=1)

# print(final_df)


# In[ ]:


blocks = [4,5,7]
subjects_a = [21, 22, 23, 24, 27, 29]
subjects_b = [8,9,10,11,12,13,14,15,16,17,18,19,20,25,26,28]

test_df_a_deceptive = final_df.loc[(final_df["block"] == 4) & (final_df["subject"].isin(subjects_a))]
test_df_a_nondeceptive = final_df.loc[(final_df["block"] != 4) & (final_df["subject"].isin(subjects_a))]
# print(test_df_a_deceptive['mean_EDA'].mean())
# print(test_df_a_nondeceptive['mean_EDA'].mean())
test_df_b_deceptive = final_df.loc[(final_df["block"] == 7) & (final_df["subject"].isin(subjects_b))]
test_df_b_nondeceptive = final_df.loc[(final_df["block"] != 7) & (final_df["subject"].isin(subjects_b))]

plt.boxplot([test_df_a_deceptive['mean_EDA'], test_df_a_nondeceptive['mean_EDA']])
plt.xticks([1, 2], ['Deceptive', 'Non-deceptive'])
plt.ylabel('Mean EDA values')
plt.savefig("/Users/saikrishna/Downloads/Physiological_Time_Series/Final Project/group_summary.png")


# ## ANOVA experimentation

# In[ ]:


anova_df = test_df

result = AnovaRM(anova_df, depvar='mean_EDA', subject='subject', within=['block']).fit()
print(result.summary())

result = AnovaRM(anova_df, depvar='mean_phasic_EDA', subject='subject', within=['block']).fit()
print(result.summary())

result = AnovaRM(anova_df, depvar='mean_IBI', subject='subject', within=['block']).fit()
print(result.summary())

result = AnovaRM(anova_df, depvar='mean_IERatio', subject='subject', within=['block']).fit()
print(result.summary())


# ### Plotting Raw vs Cleaned ECG data

# In[ ]:


# cleaned_ECG = nk.ecg_clean(block_ECG, sampling_rate=1000, method="neurokit")

# plt.rcParams['figure.figsize'] = [10, 8]
# plt.plot(block_ECG[:3000])
# plt.plot(cleaned_ECG[:3000])
# plt.legend(['raw','cleaned'])

