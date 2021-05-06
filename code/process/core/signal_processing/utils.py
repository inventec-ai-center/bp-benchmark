import matplotlib.pyplot as plt 
import scipy
from scipy import signal
import numpy as np 
import wfdb
import os
from pyampd.ampd import find_peaks
import pandas as pd
import torch

def load_wfdb(path, verbose=False, secure_load=True):
    # First fast check, sometimes unrealiable
    try:
        f = open(path+"_layout.hea")
        file = f.read()
        is_valid = ("PLETH" in file) & ("II" in file) & ("ABP" in file)
        f.close()
    except:
        is_valid = False
    
    if not is_valid: 
        if verbose: print("Record {} does not contain: PPG or ECG or ABP".format(path))
        return None
    
    try:
        r = wfdb.rdrecord(path)    
        ppg = r.p_signal[:, r.sig_name.index("PLETH")]
        ecg = r.p_signal[:, r.sig_name.index("II")]
        abp = r.p_signal[:, r.sig_name.index("ABP")]
        return r, ecg, ppg, abp
    except Exception as e:
        if verbose: print("Failed to process {}, {}".format(path, e))
        return None

def load_green_board(path, verbose=False):
    
    ecg_path = path + os.sep + "ecg_" + os.path.basename(path) + ".txt"
    ppg_path = path + os.sep + "ppg_" + os.path.basename(path) + ".txt"
    try:    
        ecg_df = pd.read_csv(ecg_path, sep="\t")
        ecg = ecg_df.iloc[:,4].values
        ppg_df = pd.read_csv(ppg_path, sep="\t")
        ppg = ppg_df.iloc[:,4].values
    except Exception as e:
        if verbose: print("Failed to process {}, {}".format(path, e))
        return None
    
    return ecg, ppg
    
def to_segments(data, fs, segment_length):
    '''
    Transform data into chunks of fs * segment_length
    - data: 1D-signal
    - fs: sample frequency in Hz (MIMIC's default is 125)
    - segment_length: in seconds
    '''

    segment_length = fs * segment_length  # Number of Samples
    N_segment = data.shape[0] // segment_length
    data_trunc = data[:(N_segment * segment_length)]
    data_stack = np.reshape(data_trunc, [N_segment, segment_length])
    return np.nan_to_num(data_stack)

def get_bp_labels(data, fs):
    idata = data.max() - data
    
    try:
        sbp_peaks = find_peaks(data, fs)
        dbp_peaks = find_peaks(idata, fs)
    except Exception as e:
        return -1, -1, -1, -1  # no any peak
    
    if 0 in [len(sbp_peaks), len(dbp_peaks)]: return -1, -1, -1, -1
    return np.median(data[sbp_peaks]), np.median(data[dbp_peaks]), np.std(data[sbp_peaks]), np.std(data[dbp_peaks])

def is_flat(x):
    dx = x[1:] - x[:-1]
    flat_parts = (np.abs(dx) < 0.00001).astype("int").sum()
    flat_parts = flat_parts/x.shape[0]

    # Reject the segment if 10% of the signals are flat
    return flat_parts > 0.1

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print ("Toc: start time not set")

# http://www.bloodpressureuk.org/BloodPressureandyou/Thebasics/Bloodpressurechart
def global_norm(x, signal_type): 
    if signal_type == "sbp": (x_min, x_max) = (60, 200)   # mmHg
    elif signal_type == "dbp": (x_min, x_max) = (30, 110)   # mmHg
    elif signal_type == "ptt": (x_min, x_max) = (100, 900)  # 100ms - 900ms
    else: return None

    # Return values normalized between 0 and 1
    return (x - x_min) / (x_max - x_min)
    
def global_denorm(x, signal_type):
    if signal_type == "sbp": (x_min, x_max) = (60, 200)   # mmHg
    elif signal_type == "dbp": (x_min, x_max) = (30, 110)   # mmHg
    elif signal_type == "ptt": (x_min, x_max) = (100, 900)  # 100ms - 900ms
    else: return None

    # Return values normalized between 0 and 1
    return x * (x_max-x_min) + x_min

# https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.health.harvard.edu%2Fheart-health%2Freading-the-new-blood-pressure-guidelines&psig=AOvVaw3MHRh3NwnByzOAIXjZw_cb&ust=1591435609080000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCICfleKt6ukCFQAAAAAdAAAAABA6
def get_hyp(sbp, dbp):
    if (sbp >= 180) or (dbp >= 120):  return 4  # Critical hypertension
    elif (sbp >= 140) or (dbp >= 90): return 3  # Stage 2 hypertension
    elif (sbp >= 130) or (dbp >= 80): return 2  # Stage 1 hypertension
    elif (sbp >= 120) and (dbp < 80): return 1  # Eleveated / Pre-hypertension
    return 0                                    # Normotension

def waveform_norm(x):
    # x: Target signal
    return (x - x.min())/(x.max() - x.min() + 1e-6)

def signal_fft(data, fs, norm):

    '''
    Get the frequency range of signal, we can get signal frequency plot with plt.plot(freq, abs_org_fft)

    Return:
        freq: discrete frequency range
        abs_org_fft: the number of samples of each freq
        norm: None or "ortho"
    '''

    org_fft = np.fft.fft(data, norm='ortho')
    abs_org_fft = np.abs(org_fft)
    freq = np.fft.fftfreq(data.shape[0], 1/fs)
    abs_org_fft = abs_org_fft[freq > 0]
    freq = freq[freq > 0]
    
    return freq, abs_org_fft

def extract_cycles(data, peaks, offset):
    '''
    extract cycles from a signal
    '''
    
    offset = np.round(offset).astype("int")  # Compute the window cut
    
    # Extract cycle based on peaks
    cycles = []  
    for p in peaks:
        start, end = (p-offset, p+offset) 

        if (start < 0) or (end > data.shape[0]):
            continue
        cycles.append(data[start:end])
        
    if len(cycles) > 0:
        cycles = np.array(cycles)
    
    return cycles

def get_fft_peaks(fft, freq, fft_peak_distance, num_iter):
    """
    Description:
    get fft peaks
    
    Paramters:
    freq, fft: from function signal_fft
    fft_peak_distance: distance from peak to peak
    num_iter: num of peaks
    """
    
    peaks = scipy.signal.find_peaks(fft[:len(fft)//6], distance=fft_peak_distance)[0]  # all of observed distance > 28
    peaks = peaks[peaks>fft_peak_distance]
    peaks = peaks[0:num_iter]
    return peaks

def fft_peaks_neighbor_avg(fft, fft_peaks, fft_neighbor_avg_interval):
    
    """
    avg of values nearby fft peaks
    """
    
    fft_peaks_neighbor_avgs = []
    for peak in fft_peaks:
        start_idx = peak - fft_neighbor_avg_interval
        end_idx = peak + fft_neighbor_avg_interval
        fft_peaks_neighbor_avgs.append(fft[start_idx:end_idx].mean())
    return np.array(fft_peaks_neighbor_avgs)

def extract_cycles_all_ppgs(waveforms, ppg_peaks, hr_offset, match_position):
    
    """
    Descrption:
    extract cycles from a signal
    
    parameters:
    waveforms: {"ppg": ppg, "vpg": vpg, "apg": apg, "ppg3": ppg3, "ppg4": ppg4}
    ppg_peaks: ppg_peaks
    hr_offset: distance from peak to peak
    match_position: "sys_peak", "dia_notches"
    """

    offset = np.round(hr_offset).astype("int")  # Compute the window cut

    # Extract cycle based on ppg_peaks

    waveforms_cycles = {
        "ppg_cycles": [],
        "vpg_cycles": [],
        "apg_cycles": [],
        "ppg3_cycles": [],
        "ppg4_cycles": []
    }

    # remove head and tail peaks
    ppg_peaks = ppg_peaks[1:-1]
    lower_offset = offset * 0.25
    upper_offset = offset * 0.75

    for p in ppg_peaks:

        start = np.round(p - lower_offset).astype("int")
        end = np.round(p + upper_offset).astype("int")

        # Align two diastolic notches of cycles
        if match_position == "dia_notches":
            tolerance = 0.1

            start = np.round(p - offset * (0.25 + tolerance)).astype("int")
            end = np.round(p + offset * (0.75 + tolerance)).astype("int")
            
            # check range
            if (start < 0) or (p <= start) or \
               (int(p + offset * (0.75-tolerance)) < 0) or (end <= int(p + offset * (0.75-tolerance))) or \
               (len(waveforms["ppg"][start:p])) == 0 or len(waveforms["ppg"][int(p + offset * (0.75-tolerance)):end]) <= 0:
                continue
            
            # stand and end from valleys
            start = start + np.argmin(waveforms["ppg"][start:p])
            end = int(p + offset * (0.75-tolerance)) + np.argmin(waveforms["ppg"][int(p + offset * (0.75-tolerance)):end])

        if (start < 0) or (end > waveforms["ppg"].shape[0]):
            continue

        for waveform_name in waveforms:
            waveforms_cycles[waveform_name + "_cycles"].append(waveforms[waveform_name][start:end])

    for waveform_name in waveforms:
        waveforms_cycles[waveform_name + "_cycles"] = np.array(waveforms_cycles[waveform_name + "_cycles"], dtype=object)
        
    return waveforms_cycles

def mean_norm_cycles(cycles, resample_length):
    
    """
    Descrption:
    calculate mean of cycles, which are normalized for both x and y
    """
    
    normalized_cycles = []
    for cycle in cycles:
        normalized_cycle = scipy.signal.resample(cycle, resample_length)
        normalized_cycle = waveform_norm(normalized_cycle)
        normalized_cycles.append(normalized_cycle)

    if len(normalized_cycles) > 0:
        normalized_cycles = np.array(normalized_cycles)
        
    #avg_normalized_cycles = normalized_cycles.mean(axis=0)
    avg_normalized_cycles = np.median(normalized_cycles, axis=0)
    return avg_normalized_cycles, normalized_cycles

def align_peaks_notches(peaks, notches):
    
    """
    Descrption:
    align peaks and notches of ppg
    """
    
    if len(peaks) == 0 or len(notches) == 0:
        return None

    aligned_peaks = []
    aligned_notches = []
    for i in range(peaks.shape[0]-1):
        min_peak = peaks[i]
        max_peak = peaks[i+1]

        try: notch = notches[(notches > min_peak) & (notches < max_peak)][0]
        except: continue

        aligned_ecg_peaks.append(min_peak)
        aligned_ppg_peaks.append(notch)

    aligned_peaks = np.array(aligned_peaks)
    aligned_notches = np.array(aligned_notches)

    if len(aligned_peaks) == 0 or len(aligned_notches) == 0:
        return None

    return aligned_peaks, aligned_notches

def max_neighbor_mean(mean_cycles, neighbor_mean_size):

    """
    Descrption:
    avg of signal values nearby max
    
    Source: 
    https://www.researchgate.net/profile/Aman_Gaurav/publication/328994912_InstaBP_Cuff-less_Blood_Pressure_Monitoring_on_Smartphone_using_Single_PPG_Sensor/links/5bf68e3da6fdcc3a8de93629/InstaBP-Cuff-less-Blood-Pressure-Monitoring-on-Smartphone-using-Single-PPG-Sensor.pdf
    """
    
    ppg_start_idx = max(np.argmax(mean_cycles) - neighbor_mean_size, 0)
    ppg_end_idx = min(np.argmax(mean_cycles) + neighbor_mean_size, len(mean_cycles))

    if ppg_start_idx == ppg_end_idx: 
        ppg_end_idx += 1

    return mean_cycles[ppg_start_idx:ppg_end_idx].mean()

def histogram_up_down(mean_cycles, num_up_bins, num_down_bins, ppg_max_idx):

    """
    Descrption:
    histogram feature
    
    Source: 
    https://www.researchgate.net/profile/Aman_Gaurav/publication/328994912_InstaBP_Cuff-less_Blood_Pressure_Monitoring_on_Smartphone_using_Single_PPG_Sensor/links/5bf68e3da6fdcc3a8de93629/InstaBP-Cuff-less-Blood-Pressure-Monitoring-on-Smartphone-using-Single-PPG-Sensor.pdf
    """
    
    H_up, bins_up = np.histogram(mean_cycles[:ppg_max_idx], bins=num_up_bins, range=(0,1), density=True)
    H_down, bins_down = np.histogram(mean_cycles[ppg_max_idx:], bins=num_down_bins, range=(0,1), density=True)

    return H_up, H_down, bins_up, bins_down

def USDC(cycles, USDC_resample_length):

    """
    Descrption:
    USDC feature
    
    Source: 
    https://www.researchgate.net/profile/Aman_Gaurav/publication/328994912_InstaBP_Cuff-less_Blood_Pressure_Monitoring_on_Smartphone_using_Single_PPG_Sensor/links/5bf68e3da6fdcc3a8de93629/InstaBP-Cuff-less-Blood-Pressure-Monitoring-on-Smartphone-using-Single-PPG-Sensor.pdf
    """
    
    usdc_features = []
    for cycle in cycles:

        # calculate usdc of one cycle
        max_idx = np.argmax(cycle)
        cycle = scipy.signal.resample(cycle[:max_idx+1], USDC_resample_length)
        max_idx = len(cycle) - 1
        usdc = (cycle * (cycle[max_idx] - cycle[0]) * np.arange(len(cycle)) - cycle * max_idx + cycle[0] * max_idx) / (np.sqrt((cycle[max_idx] - cycle[0]) ** 2 + max_idx ** 2))

        # calculate usdc feature, similar to convolute on usdc
        interval = 3
        usdc_feature = []
        for idx in range(0, max_idx, interval):
            if idx+interval < len(usdc):
                usdc_feature.append(usdc[idx:idx+interval].mean())

        usdc_features.append(usdc_feature)

    usdc_features = np.array(usdc_features)
    mean_usdc_features = usdc_features.mean(axis=0)

    return mean_usdc_features

def DSDC(cycles, DSDC_resample_length):

    """
    Descrption:
    DSDC feature
    
    Source: 
    https://www.researchgate.net/profile/Aman_Gaurav/publication/328994912_InstaBP_Cuff-less_Blood_Pressure_Monitoring_on_Smartphone_using_Single_PPG_Sensor/links/5bf68e3da6fdcc3a8de93629/InstaBP-Cuff-less-Blood-Pressure-Monitoring-on-Smartphone-using-Single-PPG-Sensor.pdf
    """
    
    dsdc_features = []
    for cycle in cycles:

        # calculate dsdc of one cycle
        max_idx = np.argmax(cycle)
        cycle = scipy.signal.resample(cycle[max_idx:], DSDC_resample_length)
        l = len(cycle) - 1
        max_idx = 0
        dsdc = (cycle * (cycle[l] - cycle[max_idx]) * np.arange(len(cycle)) - (l - max_idx) * cycle + cycle[max_idx] * l - cycle[l] * max_idx) / np.sqrt((cycle[l] - cycle[max_idx]) ** 2 + (l - max_idx) ** 2)

        # calculate dsdc feature, similar to convolute on dsdc
        interval = 3
        dsdc_feature = []
        for idx in range(max_idx, len(dsdc), interval):
            if idx+interval < len(dsdc):
                dsdc_feature.append(dsdc[idx:idx+interval].mean())

        dsdc_features.append(dsdc_feature)

    dsdc_features = np.array(dsdc_features)
    mean_udsc_features = dsdc_features.mean(axis=0)

    return mean_udsc_features

def generate_features_csv_string(features):

    """
    input features dict, output headers and features in csv string format.
    """
    
    header = []
    features_csv = []

    for feature_name in features:
        feature = features[feature_name]

        try:
            len_feature = len(feature)
        except:
            len_feature = 1

        for i in range(len_feature):
            header.append(feature_name + "_" + str(i))
        features_csv.append(feature)

    # flatten list
    features_csv = list(np.hstack(features_csv))
    
    features_csv = str(features_csv).strip("[]")

    return header, features_csv

def to_numpy(x):
    if x.is_cuda:
        return x.data.cpu().numpy()
    return x.data.numpy()

def to_tensor(x):
    if type(x) != type(torch.tensor(0)): 
        x = torch.tensor(x)
    if torch.cuda.is_available():
        return x.cuda()
    return x
