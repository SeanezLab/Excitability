import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import re

def get_npy_file_paths(dataDir):
    npy_file_paths = []
    
    for root, dirs, files in os.walk(dataDir):
        for file in files:
            if file.endswith(".npy"):
                npy_file_path = os.path.join(root, file)
                npy_file_paths.append(npy_file_path)

    return npy_file_paths
def process_data_and_plot(dataDir, fileName, sensors_csv_path, amps, trigger_height):

    output_folder = os.path.join(dataDir, 'Plots')
    subject_pattern = r'RCT(\d{3})'
    trial_pattern = r'(_\d+p_[A-Za-z]{2}|_rest)'
    # Search for the subject ID pattern
    subject_match = re.search(subject_pattern, fileName)
    if subject_match:
        # Extract and save the subject ID
        subject_id = subject_match.group()
    trial_match = re.search(trial_pattern, fileName)
    trial = ""
    if trial_match:
            # Extract and save the trial
            trial = trial_match.group()
    SID_trial = subject_id + trial
    print(SID_trial)

    # Check if the output folder exists, and create it if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Import data
    data = pd.read_csv(f"{dataDir}/{fileName}").values

    # EMG names
    #muscles = ['LRF', 'RRF', 'LST', 'RST', 'LVL', 'RVL', 'LTA', 'RTA', 'LMG', 'RMG', 'LSL', 'RSL'] #read in from sensors file
    sensors = pd.read_csv(sensors_csv_path, header=None)
    muscles = sensors.iloc[:, 2].tolist()
    muscles = muscles[:-1]
    amps = amps #[140, 117, 94, 71, 48, 25] #read in from stims file


    # Setup Parameters
    #TODO:
    sampFq = 2000 #SET THIS YOURSELF FROM SAMPLING FREQUENCY OF TRIGNO
    nAmps = len(amps)
    #TODO:
    nReps = 10 #SET THIS YOURSELF BASED ON PROTOCOL
    nMuscles = len(muscles)
    trigChan = nMuscles + 1

    # Data processing
    data[:, trigChan-1] = -data[:, trigChan-1] - np.mean(-data[:, trigChan-1])  # Remove offset and flip
    data[:, 0:nMuscles] = data[:, 0:nMuscles] * 1000  # Convert from V to mV

    # Finding peaks in the trigger channel data
    peaks, _ = find_peaks(data[:, trigChan-1], height=trigger_height, distance = 50) 
    peaks = peaks[-120:] #Used to ignore any extra peaks at the start of recording
    peakLoc = peaks[::2] #Only want first peak of paired pulses

    plt.plot(data[:, trigChan-1],'gray',alpha=0.6)
    plt.plot(peakLoc, data[:, trigChan-1][peakLoc], 'x', color='green')
    #plt.plot(peaks, data[:, trigChan-1][peaks], 'o', color='blue')

    #plt.show()

    # Muscle color coding
    muscleColors = np.array([
        [0.24313725490196078, 0.4235294117647059, 0.7019607843137254],  # RF
        [0.24313725490196078, 0.4235294117647059, 0.7019607843137254],  # RF

        [0.5058823529411764, 0.7803921568627451, 0.9333333333333333],  # ST
        [0.5058823529411764, 0.7803921568627451, 0.9333333333333333],  # ST

        [0.10588235294117647, 0.7686274509803922, 0.8823529411764706],  # VL
        [0.10588235294117647, 0.7686274509803922, 0.8823529411764706],  # VL

        [0.17254901960784313, 0.7215686274509804, 0.5843137254901961],  # TA
        [0.17254901960784313, 0.7215686274509804, 0.5843137254901961],  # TA

        [0, 0.5058823529411764, 0.5686274509803921],  # MG
        [0, 0.5058823529411764, 0.5686274509803921],  # MG

        [0.7568627450980392, 0.7411764705882353, 0.1843137254901961],  # SL
        [0.7568627450980392, 0.7411764705882353, 0.1843137254901961]   # SL
    ])

    ####################################################################################################################
    # Plotting and create region of interests of each musch from trigger peaks
    ####################################################################################################################

    # Define the window for extraction
    tPlot = 100
    window = int(tPlot * sampFq / 1000)
    timeVector = np.linspace(0, tPlot, window)

    # Define the specific range for maximum calculation
    minRangeIdx = int(3 * sampFq / 1000)  # 3 ms in indices
    maxRangeIdx = int(29 * sampFq / 1000) # 29 ms in indices

    # Preallocate a list to store data segments
    segments = [[[] for _ in range(nMuscles)] for _ in range(nAmps)]

    fig_all, axs_all = plt.subplots(len(amps), len(muscles), figsize=(18, 9), sharex=True) #figure for all 10 responses at each amplitude and muscle
    fig, axs = plt.subplots(len(amps), len(muscles), figsize=(18, 9), sharex=True) #figure for mean of all 10 responses at each amplitude and muscle

    # Loop over each amplitude and muscle to extract data
    mean_values_df = pd.DataFrame(index=amps, columns=muscles) #setup data frame with amplitudes as rows and muscles as columns
    for i, amp in enumerate(amps):
        #print(amp)
        for j, muscle in enumerate(muscles):
            #print(muscle)
            all_reps_avg = []
            for rep in range(nReps):
                # Calculate the start index for each repetition
                startIndex = peakLoc[(i * nReps) + rep]
                endIndex = min(startIndex + window, len(data))

                # Extract and store the segment
                segment = data[startIndex:endIndex, j]
                #plot all traces
                axs_all[i][j].plot(timeVector, segment, color=muscleColors[j], alpha = 0.6, linewidth = 0.8)
                segments[i][j].append(segment)

                all_reps_avg.append(segment)

            array_of_lists = np.array(all_reps_avg)
            #find average trace
            mean_values = np.mean(array_of_lists, axis=0) 
            mean_values_df.loc[amp, muscle] = mean_values
            
            #plot average trace
            axs[i][j].plot(mean_values, color=muscleColors[j]) 

            if j == 0:
                axs[i, j].set_ylabel(f'Amp {amp}')
                axs_all[i, j].set_ylabel(f'Amp {amp}')

            # Set x-label only for the top row of subplots (outer edge)
            if i == 0:
                axs[i, j].xaxis.set_label_position('top')
                axs[i, j].set_xlabel(muscle)
                axs_all[i, j].xaxis.set_label_position('top')
                axs_all[i, j].set_xlabel(muscle)

            axs[i, j].set_xticks([])
            axs[i, j].tick_params(axis='y', labelsize=8)
            axs_all[i, j].set_xticks([])
            axs_all[i, j].tick_params(axis='y', labelsize=8)

    fig.suptitle('Mean Responses for Each Muscle and Amplitude (mV)', fontsize=16)
    fig.tight_layout(h_pad=0.1, w_pad=0.1)

    fig_all.suptitle('Overlayed All Responses for Each Muscle and Amplitude (mV): ' + SID_trial, fontsize=16)
    fig_all.tight_layout(h_pad=0.1, w_pad=0.1)
    title = SID_trial + '_overlayed_all_responses.png'
    fig_all.savefig(f'{output_folder}/{title}')

    #plt.show()

    ######################################################################################################################################
    #Find responses from max amplitude row, and then search through the rest
    ######################################################################################################################################

    peakLocations = {}
    for i, muscle in enumerate(muscles):
        max_amp = mean_values_df.iloc[0]
        max_muscle_amp = max_amp[muscle]
        peakIdx = np.argmax(max_muscle_amp[10:100])
        #plt.figure()
        #plt.plot(max_muscle_amp, 'gray')
        #plt.plot(peakIdx + 10, max_muscle_amp[peakIdx + 10], marker ='x', color = 'red', markersize = 10)
        #plt.title(muscle)
        peakLocations[muscle] = peakIdx + 10


    windowSize = int(30 * sampFq / 1000)
    minRangeIdx = int(5 * sampFq / 1000)
    maxRangeIdx = int(29 * sampFq / 1000)

    df = mean_values_df
    fig_res, axs_res = plt.subplots(len(amps), len(muscles), figsize=(18, 9), sharex=True) #figure for mean traces with max and mins
    peakToPeakValues = {} #saving amplitude responses
    for i, amp in enumerate(amps):
        for j, muscle in enumerate(muscles):
            if muscle in peakLocations:

                value = df.iloc[i][j]
                searchStartIdx = max(minRangeIdx, int(peakLocations[muscle] - windowSize / 2))
                searchEndIdx = min(maxRangeIdx, int(peakLocations[muscle] + windowSize / 2))
                
                windowData = value[searchStartIdx:searchEndIdx]
                
                peakIdx = np.argmax(windowData) + searchStartIdx
                minIdx = np.argmin(windowData) + searchStartIdx

                peakToPeakValues[amp, muscle] = abs(value[peakIdx] - value[minIdx])
            
            axs_res[i][j].plot(value, color=muscleColors[j], linewidth = 2)
            axs_res[i][j].plot(peakIdx, value[peakIdx], 'k.', markersize=15)  # Highlight peak
            axs_res[i][j].plot(minIdx, value[minIdx], 'k.', markersize=15)  # Highlight min

            if j == 0:
                axs_res[i, j].set_ylabel(f'Amp {amp}')

            # Set x-label only for the top row of subplots (outer edge)
            if i == 0:
                axs_res[i, j].xaxis.set_label_position('top')
                axs_res[i, j].set_xlabel(muscle)

            axs_res[i, j].set_xticks([])
            axs_res[i, j].tick_params(axis='y', labelsize=8)

    fig_res.suptitle('Min Max Responses for Each Muscle and Amplitude (mV):' + SID_trial, fontsize=16)
    fig_res.tight_layout(h_pad=0.1, w_pad=0.1)
    title = SID_trial + '_mean_responses_with_labeled_min_max.png'
    fig_res.savefig(f'{output_folder}/{title}')

    #plt.show()


    ######################################################################################################################################
    #Find maximum response and normalize all values
    ######################################################################################################################################


    max_values = {muscle: float('-inf') for muscle in muscles}

    # Find the maximum value for each muscle
    for key, value in peakToPeakValues.items():
        muscle = key[1]
        max_values[muscle] = max(max_values[muscle], value)

    # Normalize the values by dividing each value by the corresponding maximum value
    normalized_data = {key: value / max_values[key[1]] for key, value in peakToPeakValues.items()}

    fig_n, ax_n = plt.subplots()
    for muscle, color in zip(muscles, muscleColors):
        muscle_data = [(key[0], value) for key, value in normalized_data.items() if key[1] == muscle]
        muscle_data.sort(key=lambda x: x[0])  # Sort by amplitude values

        x_values, y_values = zip(*muscle_data)
        if muscle.startswith('L'):
            ax_n.plot(x_values, y_values, label=muscle, color=color)
        else:
            ax_n.plot(x_values, y_values, label=muscle, color=color, linestyle='--')

    # Add labels and legend
    plt.xlabel('Amplitude Values')
    plt.ylabel('Normalized Response')
    plt.legend()
    plt.title('Normalized Responses:' + SID_trial)

    title = SID_trial + '_normalized_responses.png'
    fig_n.savefig(f'{output_folder}/{title}')
    # Show the plot
    #plt.show()

    print("ALL DONE :D")
def find_sensors_csv_path(dataDir):
    for root, dirs, files in os.walk(dataDir):
        for file in files:
            if file == 'sensors.csv':
                return os.path.join(root, file)
    return None

# SELECT SUBJECT data directory
#dataDir = r'C:\Users\Lab\Box\Seanez_Lab\SharedFolders\RAW DATA\RCT\RCT001\RCT001_20240116'
#amps = [230, 190, 150, 110, 70, 30]
#trigger_height = 0.03

# dataDir = r'C:\Users\Lab\Box\Seanez_Lab\SharedFolders\RAW DATA\RCT\RCT002\RCT002_20240118'
# amps = [200, 166, 132, 98, 64, 30]
# trigger_height = 0.01

# dataDir = r'C:\Users\Lab\Box\Seanez_Lab\SharedFolders\RAW DATA\RCT\RCT003\RCT003_20240119'
# amps = [140, 117, 94, 71, 48, 25]
# trigger_height = 0.05

# dataDir = r'C:\Users\Lab\Box\Seanez_Lab\SharedFolders\RAW DATA\RCT\RCT004\RCT004_20240125'
# amps = [150, 126.5, 103, 79.5, 56, 32.5]
# trigger_height = 0.01

# dataDir = r'C:\Users\Lab\Box\Seanez_Lab\SharedFolders\RAW DATA\RCT\RCT005\RCT005_20240126'
# amps = [140, 119, 98, 77, 56, 35]
# trigger_height = 0.01

dataDir = r'C:\Users\Lab\Box\Seanez_Lab\SharedFolders\RAW DATA\RCT\RCT006\RCT006_20240126'
amps = [140, 119, 98, 77, 56, 35]
trigger_height = 0.01

###########################################################################################################

csv_files = [file for file in os.listdir(dataDir) if file.endswith('.csv')]
filenames_list = []
for csv_file in csv_files:
    _, filename = os.path.split(csv_file)
    filenames_list.append(filename)


#TO RUN MAIN CODE
for i in range(len(filenames_list)-1):
    npy_files = get_npy_file_paths(dataDir)
    fileName = filenames_list[i]
    sensors_csv_path = find_sensors_csv_path(dataDir)
    process_data_and_plot(dataDir, fileName, sensors_csv_path, amps, trigger_height)
    plt.close()

