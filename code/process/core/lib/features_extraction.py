'''
This module includes functions and classes to extract different features from PPG signals. The module is able to extract two kind of features: event features, such as peaks or cycles, and statisticals features for future data analysis. The main functions and classes are:

'''

import numpy as np
from pyampd.ampd import find_peaks
import scipy.signal
import copy

from .sqi import kurtosis, skew
from .preprocessing import waveform_norm, mean_filter_normalize


#-------- General functions -------# 


def _compute_cyle_pks_vlys(sig, fs, pk_th=0.6, remove_start_end = True):
    """
    Extract the peaks and valleys that delimits the cardiac cycles of the signal waveform passed parameter.
    Peaks with an amplitude under 'pk_th' of previous peak are considered diastolic peaks and ignored.

    Parameters
    ----------
    sig: array
        Signal waveform
    fs: int
        Frequency sampling rate (Hz)
    pk_th: float
        Threshold to identify diastolic peaks (0.6 by default)  
    remove_start_end: bool
        Enable to remove first and last peak or valley found.
    
    Returns
    -------
    bool
        Flag indicating if there are peaks identified as diastolic peak (True).
    bool
        Flag indicating if signal does not follow peak-valley-peak structure (True).
    array
        Indices of the peaks of the signal waveform.
    array
        Indices of the valleys of the signal waveform.
    """

    peaks = find_peaks(sig, scale=int(fs))
    valleys = find_peaks(sig.max()-sig, scale=int(fs))

    flag1, flag2 = False, False
    
    ### Remove first or last if equal to 0 or len(sig)-1
    if peaks[0] == 0: peaks = peaks[1:]
    if valleys[0] == 0: valleys = valleys[1:]
    if peaks[-1] == len(sig)-1: peaks = peaks[:-1]
    if valleys[-1] == len(sig)-1: valleys = valleys[:-1]

    ### HERE WE SHOULD REMOVE THE FIRST AND LAST PEAK/VALLEY
    if remove_start_end:
        if peaks[0] < valleys[0]: peaks = peaks[1:]
        else: valleys = valleys[1:]

        if peaks[-1] > valleys[-1]: peaks = peaks[:-1]
        else: valleys = valleys[:-1]
    
    ### START AND END IN VALLEYS
    while len(peaks)!=0 and peaks[0] < valleys[0]:
        peaks = peaks[1:]
    
    while len(peaks)!=0 and peaks[-1] > valleys[-1]:
        peaks = peaks[:-1]
    
    if len(peaks)==0 or len(valleys)==0:
        return True, True, [], []
        
    ## Remove consecutive peaks with one considerably under the other
    new_peaks = []
    mean_vly_amp = np.mean(sig[valleys])
    # define base case:
    
    for i in range(len(peaks)-1):
        if sig[peaks[i]]-mean_vly_amp > (sig[peaks[i+1]]-mean_vly_amp)*pk_th:
            new_peaks.append(peaks[i])
            a=i
            break
            
    if len(peaks) == 1:
        new_peaks.append(peaks[0])
        a=0

    for j in range(a+1,len(peaks)):
        if sig[peaks[j]]-mean_vly_amp > (sig[new_peaks[-1]]-mean_vly_amp)*pk_th:
            new_peaks.append(peaks[j])
            
    if not np.array_equal(peaks,new_peaks):
        flag1 = True
        
    if len(valleys)-1 != len(new_peaks):
        flag2 = True
        
    if len(valleys)-1 == len(new_peaks):
        for i in range(len(valleys)-1):
            if not(valleys[i] < new_peaks[i] and new_peaks[i] < valleys[i+1]):
                flag2 = True
        
    return flag1, flag2, new_peaks, valleys


def compute_sp_dp(sig, fs, pk_th=0.6, remove_start_end = False):
    """
    Compute SBP and DBP as the median of the amplitude of the systolic peaks and diastolic valleys.
    The extracted peaks and valleys that delimits the cardiac cycles are extracted from the signal waveform passed parameter.
    Peaks whose amplitude is under 'pk_th' of previous peak are considered diastolic peaks and ignored.

    Parameters
    ----------
    sig : array
        Signal waveform
    fs: int
        Frequency sampling rate (Hz)
    pk_th: float
        Threshold to identify diastolic peaks (0.6 by default)  
    remove_start_end: bool
        Enable to remove first and last peak or valley found.
    
    Returns
    -------
    bool
        Flag indicating if there are peaks identified as diastolic peak (True).
    bool
        Flag indicating if signal does not follow peak-valley-peak structure (True).
    array
        Indices of the peaks of the signal waveform.
    array
        Indices of the valleys of the signal waveform.
    """

    flag1, flag2, new_peaks, valleys = _compute_cyle_pks_vlys(sig, fs, pk_th=pk_th, remove_start_end = remove_start_end)

    if len(new_peaks)!=0 and len(valleys)!=0:
        sp, dp = np.median(sig[new_peaks]), np.median(sig[valleys])
    else:
        sp, dp = -1 , -1

    return sp, dp, flag1, flag2, new_peaks, valleys


def extract_cycle_check(sig, fs, pk_th=0.6, remove_start_end = True):
    """
    Extract the cardiac cycles of the signal waveform passed parameter.
    Peaks whose amplitude is under 'pk_th' of previous peak are considered diastolic peaks and ignored.

    Parameters
    ----------
    sig : array
        Signal waveform
    fs: int
        Frequency sampling rate (Hz)
    pk_th: float
        Threshold to identify diastolic peaks (0.6 by default)  
    remove_start_end: bool
        Enable to remove first and last peak or valley found.
    
    Returns
    -------
    array
        cycles of the signal. 
    array
        Indices of the systolic peaks of each cycle (normalized to each cycle length)
    bool
        Flag indicating that a peak was identified as diastolic peak (if True).
    bool
        Flag indicating that signal does not follow peak-valley-peak structure (if True).
    array
        Indices of the peaks of the signal waveform.
    array
        Indices of the valleys of the signal waveform.
    """

    flag1, flag2, new_peaks, valleys = _compute_cyle_pks_vlys(sig, fs, pk_th=pk_th, remove_start_end = remove_start_end)

    cycles = []
    peaks_norm = []

    if len(new_peaks)!=0 and len(valleys) !=0:
        ## Save segments
        for i in range(len(valleys)-1):
            #print((valleys[i],valleys[i+1]))
            cycles.append(sig[valleys[i]:valleys[i+1]])
            
        ## Save peaks
        if len(valleys)-1 == len(new_peaks):
            for i in range(len(new_peaks)):
                peaks_norm.append(new_peaks[i]-valleys[i])
    
    return cycles, peaks_norm, flag1, flag2, new_peaks, valleys


def extract_feat_cycle(cycles, peaks_norm, fs):
    """
    Extracts the time-based features of each cycle and ouputs their average.

    Parameters
    ----------
    cycles : array
        Cycles of a PPG signal waveform. 
    peaks_norm : array
        Indices of the systolic peaks of each cycle (normalized to each cycle length)
    fs: int
        Frequency sampling rate (Hz)

    Returns
    -------
    array
        Header with a list of the name of each feature. 
    array
        Average of the cycles' features
    """
    
    feats = []
    feat_name = []

    for c, p in zip(cycles, peaks_norm):
        try:
            feat_name,feat= extract_temp_feat(c, p, fs)
            feats.append(feat)
        except:
            print("Cycle ignored")

    if len(feats)>0: feats = np.vstack(feats).mean(axis=0)
    else: feats = np.array([])
        
    return feat_name, feats


def extract_feat_original(sig, fs, filtered=True, remove_start_end=True):
    """
    Extracts other set of features from PPG signal waveform

    Parameters
    ----------
    sig : array
        PPG signal waveform. 
    fs: int
        Frequency sampling rate (Hz)
    filtered : array
        Filter derivatives to compute the features.
    remove_start_end : array
        Enable to remove first and last peak/valley in cycle identification.

    
    Returns
    -------
    array
        Header with a list of the name of each feature. 
    array
        Extracted features.
    """

    ppg = PPG(sig,fs)
    _, head, feat_str = ppg.features_extractor(filtered=filtered, remove_first=remove_start_end)
    
    feat = [float(s) for s in feat_str.split(', ')]
    
    return head, feat


#-------- Cycle based temporal features -------# 

def width_at_per(per, cycle, peak, fs):
    """
    Extract the width of the systolic and diastolic phases at ('per'*100)% of amplitude of systolic peak

    Parameters
    ----------
    per: float
        ratio of the amplitude of the systolic peak to compute the width.
    cycle: array
        PPG cycle waveform. 
    peak: int
        Index of the systolic peak.
    fs: int
        Frequency sampling rate (Hz)

    Returns
    -------
    float
        Width of the systolic phase at ('per'*100)%  of amplitude of systolic peak.
    float
        Width of the diastolic phase at ('per'*100)%  of amplitude of systolic peak.
    """
    
    height_to_reach = cycle[peak]*per
    
    i = 0 
    while i < peak and cycle[i] < height_to_reach:
        i+=1
    
    SW = peak - i
    
    i = peak
    while i < len(cycle) and cycle[i] > height_to_reach:
        i +=1
    i -= 1
    
    DW = i - peak
    
    return SW, DW 
    

def vpg_points(vpg, peak):
    """
    Extract VPG or FDPPG interest points:
        w: maximum value between start and peak, same as Steepest
        y: relevant valley after w (after peak), same as NegSteepest
        z: maximum value after w with limit search, peak after y, same as TdiaRise

    Parameters
    ----------
    vpg : array
        VPG or FDPPG signal waveform. 
    peak: int
        Index of the systolic peak.

    Returns
    -------
    int
        Location (index) in VPG of w.
    int
        Location (index) in VPG of y.
    int
        Location (index) in VPG of z.
    """
    
    pks = find_peaks(vpg)
    vlys = find_peaks(-vpg)
    
    #Time from cycle start to first peak in VPG (steepest point)
    #steepest point == max value between start and peak
    Tsteepest = np.argmax(vpg[:peak])
    w = Tsteepest 
    
    pks = pks[pks > peak]
    vlys = vlys[vlys > peak]

    if len(vlys) < 1:
        # min slope in ppg' and max slope in ppg'
        end = int((len(vpg)-peak)*0.4)+peak
        y = np.argmin(vpg[peak:end])+peak
    else:
        y = vlys[0]
        
    min_slope_idx = y

    #pks = pks[pks > y]
    
    # Find the max (diatolic rise) from the prev. min to the end/2
    end = int((len(vpg)-min_slope_idx)*0.4)+min_slope_idx
    #TdiaRise. Max positive slope after 'peak'
    TdiaRise = np.argmax(vpg[min_slope_idx:end])+min_slope_idx
    z = TdiaRise
    
    return w, y, z
   

def apg_points(apg, peak, w, y, z):
    """
    Extract APG or SDPPG interest points:
        a: point with the highest acceleration of the systolic upstroke (early systolic positive wave).
        b: deceleration point of the systolic phase (early systolic negative wave).
        c: first relevant peak in APG after the systolic peak (late systolic reincreasing wave).
        d: relevant and last valley of APG (late systolic redecreasing wave).
        e: dicrotic notch (early diastolic positive wave).

    Parameters
    ----------
    apg: array
        APG or SDPPG signal waveform. 
    peak: int
        Index of the systolic peak.
    w: int
        Index of the point w.
    y: int
        Index of the point y.
    z: int
        Index of the point z.

    Returns
    -------
    int
        Location (index) in APG of a.
    int
        Location (index) in APG of b.
    int
        Location (index) in APG of c.
    int
        Location (index) in APG of d.
    int
        Location (index) in APG of e.
    """
    a = b = c = d = e = 0
    
    #Limit the search to 60% of the cycle len
    pks = find_peaks(apg[:int(len(apg)*0.6)])
    vly, _ = scipy.signal.find_peaks(-apg[:int(len(apg)*0.6)])
    
    if len(pks) < 1 or len(vly) < 1: 
        return a, b, c, d, e
    
    #### compute 'a' as the first peak of apg, if not max val before peak
    #a = np.argmax(apg[0:peak])
    a = pks[0]
    if a > peak:
        a = np.argmax(apg[0:peak])
    else:
        pks = pks[1:]
        
    vly = vly[vly > a]
    if len(vly) < 1: 
        return a, b, c, d, e
    
    # Reduce the valleys after w
    vly = vly[vly > w]
    if len(vly) < 1: 
        return a, b, c, d, e
    
    
    #### compute 'e' as the max peak after systolic peak 
    #pks_tmp = pks[pks > peak] # peaks after systolic peak
    pks_tmp = pks[pks > y] # peaks after y
    if len(pks_tmp)!=0:
        e = pks_tmp[np.argmax(apg[pks_tmp])]
    else: # max value after y
        e =  np.argmax(apg[y+1:])
    pks = pks[pks < e]
    vly = vly[vly < e]
    
    #### compute 'b' as the first valley of apg after a, if not min val before peak
    if len(vly[vly < peak-2]) < 1:
        vly_b, _ = scipy.signal.find_peaks(-apg[w:peak-2])
        vly_b+=w
        
        if len(vly_b) < 1:
            b = np.argmin(apg[w:peak-2])+w
        else:
            b = vly_b[0]
    else:
        b = vly[0]
    
    
    #### compute 'd' as the min value after sys peak and e (or last valley)
    # Last valley between peak and e
    vly_tmp, _ = scipy.signal.find_peaks(-apg[peak:e])
    vly_tmp+=peak
    if len(vly_tmp) < 1:
        d = np.argmin(apg[peak:e])+peak
    else: # min valley after peak
        d = vly_tmp[np.argmin(apg[vly_tmp])]
        #d = vly_tmp[-1]
        
    #d = np.argmin(apg[peak:e])+peak
      
    #### compute 'c' as max peak between b and d or the max value between b and d
    pks_tmp, _ = scipy.signal.find_peaks(apg[b:d+1])
    pks_tmp+=b
    if len(pks_tmp) < 1:
        c = np.argmax(apg[b:d])+b
    else:
        c = pks_tmp[np.argmax(apg[pks_tmp])]
        #c = pks_tmp[-1]
    
    #c = np.argmax(apg[b:d])+b
    
    return a, b, c, d, e


def extract_apg_feat(cycle, vpg, peak, w, y, z, fs):
    """
    Extract features related to interest points of APG (one cycle).

    Parameters
    ----------
    cycle: array
        PPG cycle waveform. 
    vpg: array
        VPG or FDPPG signal waveform. 
    peak: int
        Index of the systolic peak.
    w: int
        Index of the point w.
    y: int
        Index of the point y.
    z: int
        Index of the point z.
    fs: int
        Frequency sampling rate (Hz)

    Returns
    -------
    array
        Header with a list of the name of each feature. 
    array
        Extracted features related to interest points of APG (one cycle).
    """

    
    feats = []
    feats_header = []

    apg = np.diff(vpg)
    
    Tc = len(cycle)
    
    a, b, c, d, e = apg_points(apg,peak,w, y, z)
    apg_p = [a, b, c, d, e]
    apg_p_names = ['a', 'b', 'c', 'd', 'e']
    
    #apg amplitudes
    feats += [apg[i] for i in apg_p]
    feats_header += ['apg_'+i for i in apg_p_names]
    
    #ppg amplitudes
    feats += [cycle[i+2] for i in apg_p]
    feats_header += ['ppg_'+i for i in apg_p_names]
    
    #ratio amplitudes
    feats += [apg[i]/apg[a] for i in apg_p[1:]]
    feats_header += ['ratio_apg_'+i for i in apg_p_names[1:]]
    
    feats += [cycle[i+2]/cycle[a+2] for i in apg_p[1:]]
    feats_header += ['ratio_ppg_'+i for i in apg_p_names[1:]]
    
    #Time apg points
    feats += [a/fs,(b-a)/fs,(c-b)/fs,(d-c)/fs,(e-d)/fs]
    feats_header += ['T_'+i for i in apg_p_names]
    
    feats += [a/Tc,(b-a)/Tc,(c-b)/Tc,(d-c)/Tc,(e-d)/Tc]
    feats_header += ['T_'+i+'_norm' for i in apg_p_names]
    
    #Time apg points 2
    feats += [(peak-a)/fs,(peak-b)/fs,(c-peak)/fs,(d-peak)/fs,(e-peak)/fs]
    feats_header += ['T_peak_'+i for i in apg_p_names]
    
    feats += [(peak-a)/Tc,(peak-b)/Tc,(c-peak)/Tc,(d-peak)/Tc,(e-peak)/Tc]
    feats_header += ['T_peak_'+i+'_norm' for i in apg_p_names]
    
    # Aging Index
    feats += [(apg[b]-apg[c]-apg[d]-apg[e])/apg[a]]
    feats_header += ['AI']
    
    # Others ratios (taken from the article)
    bd = (vpg[d+1] - vpg[b+1])/(d-b)
    bcda = (apg[b] - apg[c] - apg[d])/apg[a]
    sdoo = np.sum(vpg[peak:d+1]*vpg[peak:d+1])/np.sum(vpg*vpg)
    
    feats += [bd, bcda, sdoo]
    feats_header += ['bd', 'bcda', 'sdoo']
    
    return feats_header, feats


def extract_temp_feat(cycle, peak, fs):
    """
    Extract temporal features related to interest points of VPG & APG of one cycle PPG.

    Parameters
    ----------
    cycle: array
        PPG cycle waveform. 
    peak: int
        Index of the systolic peak.
    fs: int
        Frequency sampling rate (Hz)

    Returns
    -------
    array
        Header with a list of the name of each feature. 
    array
        Extracted features related to interest points of VPG & APG of one cycle PPG.
    """
    
    # REMEMBER TO DIVIDE BY FS
    feat = []
    
    #Time of the cycle
    Tc = len(cycle)
    #Time from start to sys peak
    Ts = peak
    
    #Time from systolic to end
    Td = len(cycle) - peak
    
    vpg = np.diff(cycle)
    w, y, z = vpg_points(vpg, peak)
    
    
    #Time from cycle start to first peak in VPG (steepest point)
    #steepest point == max value between start and peak
    Tsteepest = w
    Steepest = vpg[w]
    
    TNegSteepest = y
    
    #TdiaRise. Max positive slope after 'peak'
    TdiaRise = z
        
    #Greatest negative steepest (slope) from peak to end. (Slope, Time)
    NegSteepest = vpg[TNegSteepest]
    
    # Amplitude to DiaRise
    DiaRise = cycle[TdiaRise]
    # SlopeDiaRise
    SteepDiaRise = vpg[TdiaRise]
    
    #Time from Systolic peak to Diastolic Rise
    TSystoDiaRise = TdiaRise - Ts
    
    #Time from Diastolic Rise to End
    TdiaToEnd = Tc - TdiaRise
    
    #Ratio between systolic peak and diastolic rise amplitude
    Ratio = cycle[peak]/DiaRise
    
    point_feat_name = ['Tc', 'Ts', 'Td', 'Tsteepest', 'Steepest', 'TNegSteepest', 'NegSteepest', 
            'TdiaRise', 'DiaRise', 'SteepDiaRise', 'TSystoDiaRise', 'TdiaToEnd', 'Ratio']
    point_feat = [Tc/fs, Ts/fs, Td/fs, Tsteepest/fs, Steepest, TNegSteepest/fs, NegSteepest, 
            TdiaRise/fs, DiaRise, SteepDiaRise, TSystoDiaRise/fs, TdiaToEnd/fs, Ratio]
    
    #norm by cycle
    point_feat_name = point_feat_name + ['Ts_norm', 'Td_norm', 'Tsteepest_norm', 'TNegSteepest_norm',
                       'TdiaRise_norm', 'TSystoDiaRise_norm', 'TdiaToEnd_norm']
    point_feat = point_feat + [Ts/Tc, Td/Tc, Tsteepest/Tc, TNegSteepest/Tc, 
            TdiaRise/Tc, TSystoDiaRise/Tc, TdiaToEnd/Tc]
    
    #width_at_per
    width_names = []
    width_feats = []

    for per in [0.25,0.50,0.75]:
        SW, DW = width_at_per(per, cycle, peak, fs)
        per_str = str(int(per*100))
        width_names += ['SW'+per_str, 'SW'+per_str+'_norm', 
                               'DW'+per_str, 'DW'+per_str+'_norm', 
                               'SWaddDW'+per_str, 'SWaddDW'+per_str+'_norm',
                               'DWdivSW'+per_str]
        width_feats +=[SW/fs, SW/Tc,
                        DW/fs, DW/Tc,
                        (SW+DW)/fs, (SW+DW)/Tc,
                        DW/SW]
        
    point_feat_name += width_names
    point_feat += width_feats
    
    min_val=np.min(cycle)
    #Area under the curve (AUC) from start of cycle to max upslope point
    S1 = np.trapz(cycle[:Tsteepest]-min_val)
    #AUC from max upslope point to systolic peak
    S2 = np.trapz(cycle[Tsteepest:peak]-min_val)
    #AUC from systolic peak to diastolic rise 
    S3 = np.trapz(cycle[peak:TdiaRise]-min_val)
    #AUC from diastolic rise to end of cycle
    S4 = np.trapz(cycle[TdiaRise:]-min_val)
    #AUC of systole area S1+S2
    AUCsys = S1+S2
    #AUC of diastole area S3+S4
    AUCdia = S3+S4
    area_feat_name = ['S1','S2','S3','S4','AUCsys','AUCdia']
    area_feat = [S1,S2,S3,S4,AUCsys,AUCdia]
    
    area_feat_name += ['S1_norm','S2_norm','S3_norm','S4_norm','AUCsys_norm','AUCdia_norm']
    area_feat += [S1/AUCsys,S2/AUCsys,S3/AUCdia,S4/AUCdia,AUCsys/(AUCsys+AUCdia),AUCdia/(AUCsys+AUCdia)]
    
    # SQI feats
    SQI_skew = skew(cycle,0.3)
    SQI_kurtosis = kurtosis(cycle)
    sqi_feat_name = ['SQI_skew','SQI_kurtosis']
    sqi_feat = [SQI_skew,SQI_kurtosis]
    
    feat_name = point_feat_name + area_feat_name +sqi_feat_name
    feat = point_feat + area_feat + sqi_feat
    
    feats_header_apg, feats_apg = extract_apg_feat(cycle, vpg, peak, w, y, z, fs)
    
    feat_name += feats_header_apg
    feat += feats_apg
    
    return np.array(feat_name),np.array(feat)


### ---------- Previous Feature Extraction Functions ----------

def signal_fft(data, fs, norm='ortho'):
    """Get the frequency range of signal using FFT
     We can get signal frequency plot with plt.plot(freq, abs_org_fft)

    Parameters
    ----------
    data : array
        1D array of the signal.
    fs : float
        Sampling rate of the signal.
    norm : {None, "ortho"}, default="ortho"
        Norm used to compute FFT

    Returns
    -------
    freq : array
        Discrete frequency range
    abs_org_fft : array
        Number of samples of each frequency
    """ 

    org_fft = np.fft.fft(data, norm=norm)
    abs_org_fft = np.abs(org_fft)
    freq = np.fft.fftfreq(data.shape[0], 1/fs)
    abs_org_fft = abs_org_fft[freq > 0]
    freq = freq[freq > 0]
    
    return freq, abs_org_fft

def get_fft_peaks(fft, freq, fft_peak_distance = 28, num_iter = 5):

    """ Extract the peaks of the signal's FFT given as parameter.

    Parameters
    ----------
    fft : array
        Number of samples of each frequency computed with `signal_fft`.
    freq : array
        Discrete frequency range computed with `signal_fft`.
    fft_peak_distance : int
        Minimum distance from peak to peak.
    num_iter : int
        Number of peaks to consider.

    See Also
    --------
    signal_fft

    Returns
    -------
    array
        Peaks of the signal's FFT.

    """ 
    
    peaks = scipy.signal.find_peaks(fft[:len(fft)//6], distance=fft_peak_distance)[0]  # all of observed distance > 28
    peaks = peaks[peaks>fft_peak_distance]
    peaks = peaks[0:num_iter]
    return peaks

def fft_peaks_neighbor_avg(fft, fft_peaks, fft_neighbor_avg_interval = 6):
    """ Compute the average values nearby the fft_peaks.
    This feature is used mainly for PPG, but it can be used for ECG.

    Parameters
    ----------
    fft : array
        Number of samples of each frequency computed with `signal_fft`.
    fft_peaks : array
        Peaks of FFT computed with `get_fft_peaks`.
    fft_neighbor_avg_interval : int
        Minimum distance from peak to peak.
    fft_neighbor_avg_interval : int
        Range of neighbors to consider.

    See Also
    --------
    signal_fft, get_fft_peaks 

    Returns
    -------
    array
        Average values nearby the fft_peaks.

    """ 
    
    fft_peaks_neighbor_avgs = []
    for peak in fft_peaks:
        start_idx = peak - fft_neighbor_avg_interval
        end_idx = peak + fft_neighbor_avg_interval
        fft_peaks_neighbor_avgs.append(fft[start_idx:end_idx].mean())
    return np.array(fft_peaks_neighbor_avgs)

def extract_cycles_all_ppgs(waveforms, ppg_peaks, hr_offset, match_position, remove_first = True):
    """ Extract cycles from the waveforms of PPG and its derivatives.

    Parameters
    ----------
    waveforms : dict {"ppg": ppg, "vpg": vpg, "apg": apg, "ppg3": ppg3, "ppg4": ppg4}
        All the waveforms of PPG and its derivatives.
    ppg_peaks: array
        Peaks of the PPG signal.
    hr_offset: float
        Distance from peak to peak.
    match_position: {"sys_peak", "dia_notches"}
        Position to match the cycles.
        Systolic peaks ("sys_peak") or Diastolic notches ("dia_notches")

    Returns
    -------
    dict 
        {"ppg_cycles": array, "vpg_cycles": array, "apg_cycles": array, "ppg3_cycles": array, "ppg4_cycles": array}
        Cycles from the waveforms of PPG and its derivatives.
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
    if remove_first:
        ppg_peaks = ppg_peaks[1:-1] ## Original
    else: ## Addition for small waveforms
        if ppg_peaks[0] == 0:
            ppg_peaks = ppg_peaks[1:]
        if ppg_peaks[-1] == len(waveforms['ppg'])-1:
            ppg_peaks = ppg_peaks[:-1]
            
    
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

def mean_norm_cycles(cycles, resample_length = 80):
    """ Calculate mean of cycles which is normalized with min-max normalization and resampled.

    Parameters
    ----------
    cycles : array
        Cycles of the signal to compute the mean (2D-array)
    resample_length: int
        Resample length.

    Returns
    -------
    avg_normalized_cycles : array
        Normalized mean of the cycles.
    normalized_cycles : array
        Normalized and Resampled cycles.
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

def max_neighbor_mean(mean_cycles, neighbor_mean_size = 5):
    """ Compute the mean of values near the maximum value.

    Parameters
    ----------
    mean_cycles : array
        Mean of all cycles of the signal.
    neighbor_mean_size: int
        Range of near values to consider in the average.

    Returns
    -------
    float 
        Mean of values near the maximum value.
    
    References
    ----------
    .. [1] https://www.researchgate.net/profile/Aman_Gaurav/publication/328994912_InstaBP_Cuff-less_Blood_Pressure_Monitoring_on_Smartphone_using_Single_PPG_Sensor/links/5bf68e3da6fdcc3a8de93629/InstaBP-Cuff-less-Blood-Pressure-Monitoring-on-Smartphone-using-Single-PPG-Sensor.pdf

    """
    
    ppg_start_idx = max(np.argmax(mean_cycles) - neighbor_mean_size, 0)
    ppg_end_idx = min(np.argmax(mean_cycles) + neighbor_mean_size, len(mean_cycles))

    if ppg_start_idx == ppg_end_idx: 
        ppg_end_idx += 1

    return mean_cycles[ppg_start_idx:ppg_end_idx].mean()

def histogram_up_down(mean_cycles, num_up_bins, num_down_bins, ppg_max_idx):
    """ Compute histogram features of the cycle given as parameter. 
    Two histograms are computed.
        - Up: From the start of the cycle to the maximum value.
        - Down: From the maximum value to the end of the cycle.


    Parameters
    ----------
    mean_cycles : array
        Cycle or Mean of all cycles of the signal.
    num_up_bins: int
        Number of bins for the Up histogram.
    num_down_bins: int
        Number of bins for the down histogram.
    ppg_max_idx: int
        Index markind the maximum value of the cycle.

    Returns
    -------
    H_up : array 
        Values of the Up histogram. 
    H_down: array 
        Values of the Down histogram.
    bins_up : array 
        Bin edges of the Up histogram.
    bins_down : array 
        Bin edges of the Down histogram.

    References
    ----------
    .. [1] https://www.researchgate.net/profile/Aman_Gaurav/publication/328994912_InstaBP_Cuff-less_Blood_Pressure_Monitoring_on_Smartphone_using_Single_PPG_Sensor/links/5bf68e3da6fdcc3a8de93629/InstaBP-Cuff-less-Blood-Pressure-Monitoring-on-Smartphone-using-Single-PPG-Sensor.pdf

    """
    
    H_up, bins_up = np.histogram(mean_cycles[:ppg_max_idx], bins=num_up_bins, range=(0,1), density=True)
    H_down, bins_down = np.histogram(mean_cycles[ppg_max_idx:], bins=num_down_bins, range=(0,1), density=True)

    return H_up, H_down, bins_up, bins_down

def USDC(cycles, USDC_resample_length):
    """ Compute mean feature with UpSlope Deviation curve. 
    Deviation of each point from the mean upslope on the rising edge 
    and depict the relative speed of systolic activity.

    This feature is designed for PPG cycles.
    
    Parameters
    ----------
    cycles : array
        Cycles of the signal.
    USDC_resample_length: int
        Resample of the systolic segment.

    Returns
    -------
    array 
        Average of the USDC features extracted for each cycle.

    References
    ----------
    .. [1] https://www.researchgate.net/profile/Aman_Gaurav/publication/328994912_InstaBP_Cuff-less_Blood_Pressure_Monitoring_on_Smartphone_using_Single_PPG_Sensor/links/5bf68e3da6fdcc3a8de93629/InstaBP-Cuff-less-Blood-Pressure-Monitoring-on-Smartphone-using-Single-PPG-Sensor.pdf
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
    """ Compute mean feature with DownSlope Deviation curve. 
    Deviation of each point from the mean downslope on the falling edge 
    and depict the relative speed of diastolic activity.

    This feature is designed for PPG cycles.
    
    Parameters
    ----------
    cycles : array
        Cycles of the signal.
    DSDC_resample_length: int
        Resample of the diastolic segment.

    Returns
    -------
    array 
        Average of the DSDC features extracted for each cycle.

    References
    ----------
    .. [1] https://www.researchgate.net/profile/Aman_Gaurav/publication/328994912_InstaBP_Cuff-less_Blood_Pressure_Monitoring_on_Smartphone_using_Single_PPG_Sensor/links/5bf68e3da6fdcc3a8de93629/InstaBP-Cuff-less-Blood-Pressure-Monitoring-on-Smartphone-using-Single-PPG-Sensor.pdf
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
    """ Transform a dictionary of extracted features to a string in CSV format.
    
    Parameters
    ----------
    features : dict
        Dictionary of the extracted features with their names and values.

    Returns
    -------
    header : list 
        Header of with the names of the different features.
        [{feature_name_0}, {feature_name_1}, {feature_name_2}...] - Features with the same name will add _1 _2 _3 as suffix, _0 is a fixed suffix.
    features_csv : str 
        Values of the features in csv string format.

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



### ---------- Implementation of PPG ----------

class PPG:
    """ Implementation PPG class for features extraction of PPG signals.

    Parameters
    ----------
    data : array
        PPG signal 1D-array.
    fs : int
        Sampling rate.
    cycle_size : int
        Resampling length when the cycles are extracted.
        
    Attributes
    ----------
    data : array
        PPG signal array.
    idata : array
        Inverse of the signal used to compute the valleys.
    fs : array
        Sampling rate.
    cycle_size : int
        Resampling length when the cycles are extracted.

    """
    def __init__(self, data, fs, cycle_size=128):
        self.data = data
        self.idata = data.max() - data
        self.fs = fs
        self.cycle_size = cycle_size

    def peaks(self, **kwargs): # systolic peaks
        """ Extract the peaks of the PPG signal 

        Returns
        -------
        array 
            Indeces marking the extracted peaks.
        """
        # x: ppg signal
        return find_peaks(self.data, scale=int(self.fs))

    def vpg(self, **kwargs):
        """ Compute the 1st Derivative of the PPG.

        Returns
        -------
        array 
            1st Derivative of the PPG signal (vpg).
        """
        vpg = self.data[1:] - self.data[:-1]
        padding = np.zeros(shape=(1))
        vpg = np.concatenate([padding, vpg], axis=-1)
        return vpg

    def apg(self, **kwargs):
        """ Compute the 2nd Derivative of the PPG.

        Returns
        -------
        array 
            2nd Derivative of the PPG signal (apg).
        """
        apg = self.data[1:] - self.data[:-1]  # 1st Derivative
        apg = apg[1:] - apg[:-1]  # 2nd Derivative

        padding = np.zeros(shape=(2))
        apg = np.concatenate([padding, apg], axis=-1)
        return apg
    
    def ppg3(self, **kwargs):
        """ Compute the 3rd Derivative of the PPG.

        Returns
        -------
        array 
            3rd Derivative of the PPG signal.
        """
        ppg3 = self.data[1:] - self.data[:-1]  # 1st Derivative
        ppg3 = ppg3[1:] - ppg3[:-1]  # 2nd Derivative
        ppg3 = ppg3[1:] - ppg3[:-1]  # 3nd Derivative

        padding = np.zeros(shape=(3))
        ppg3 = np.concatenate([padding, ppg3], axis=-1)
        return ppg3
    
    def ppg4(self, **kwargs):
        """ Compute the 4th Derivative of the PPG.

        Returns
        -------
        array 
            4th Derivative of the PPG signal.
        """
        ppg4 = self.data[1:] - self.data[:-1]  # 1st Derivative
        ppg4 = ppg4[1:] - ppg4[:-1]  # 2nd Derivative
        ppg4 = ppg4[1:] - ppg4[:-1]  # 3nd Derivative
        ppg4 = ppg4[1:] - ppg4[:-1]  # 4nd Derivative

        padding = np.zeros(shape=(4))
        ppg4 = np.concatenate([padding, ppg4], axis=-1)
        return ppg4

    def hr(self, **kwargs):
        """ Compute the heart rate from the peak distances in BPM, returns 0 if peaks are not present.
        Returns
        -------
        float 
            Heart rate in BPM of the signal.
        """ 
        try: return self.fs / np.median(np.diff(self.peaks())) * 60
        except: return 0

    def diastolic_notches(self, **kwargs):
        """ Extract the diastolic notches of the PPG signal.
        Similar to `valleys` function.

        Returns
        -------
        array 
            Indeces marking the diastolic notches.
        """
        notches = find_peaks(-self.data, scale=int(self.fs))
        #notches = scipy.signal.find_peaks(-self.data, distance=35, height=-self.data.mean())[0]
        
        return notches    
       
    def features_extractor(self, filtered=False, 
        fft_peak_distance = 28, fft_neighbor_avg_interval = 6, 
        resample_length = 80,
        neighbor_mean_size = 5,
        num_up_bins = 5,
        num_down_bins = 10, 
        remove_first = True,
        one_cycle_sig = False):
        """ Extract all features from PPG

        Returns
        -------
        features : dict 
            Extracted features with their names and values as {"{feature_name}": feature_value, ...}
        header: list
            Names of the different features as:
            [{feature_name_0}, {feature_name_1}, {feature_name_2}...] - Features with the same name will add _1 _2 _3 as suffix, _0 is a fixed suffix
        features_csv_str : str
            Values of features in csv string format.

        Examples
        --------
        >>> ppg = PPG(ppg, 125)
        >>> features, header, features_csv_str = ppg.features_extractor()
        """
        
        if one_cycle_sig:
            remove_first = False
        
        features = {}
        
        # default parameters
        USDC_resample_length = resample_length // 4
        DSDC_resample_length = resample_length // 4 * 3

        # normalize ppg, vpg, apg, ppg3, ppg4
        ppg = self.data
        ppg = waveform_norm(ppg)
        if filtered:
            vpg = mean_filter_normalize( self.vpg(), int(self.fs), 0.75, 10, 1)
            apg = mean_filter_normalize( self.apg(), int(self.fs), 0.75, 10, 1)
            ppg3 = mean_filter_normalize( self.ppg3(), int(self.fs), 0.75, 10, 1)
            ppg4 = mean_filter_normalize( self.ppg4(), int(self.fs), 0.75, 10, 1)
        else:
            vpg = waveform_norm(self.vpg())
            apg = waveform_norm(self.apg())
            ppg3 = waveform_norm(self.ppg3())
            ppg4 = waveform_norm(self.ppg4())
            
        # get peaks and valleys
        sys_peaks = find_peaks(self.data, scale=int(self.fs))
        dia_notches = self.diastolic_notches()
        
        if one_cycle_sig:
            sys_peaks = np.array([self.data.argmax()])
            dia_notches = np.array([0,len(self.data)-1])
        
        # all signals from a ppg signal
        waveforms = {
            "ppg": ppg,
            "vpg": vpg,
            "apg": apg,
            "ppg3": ppg3,
            "ppg4": ppg4
        }
        
        # Frequency domain features
        for waveform_name in waveforms:
            freq, fft = signal_fft(waveforms[waveform_name], self.fs)
            fft = fft / np.linalg.norm(fft)
            fft_peaks = get_fft_peaks(fft, freq, fft_peak_distance, num_iter=5)
            fft_peaks_neighbor_avgs = fft_peaks_neighbor_avg(fft, fft_peaks, fft_neighbor_avg_interval)
            features[waveform_name + "_fft_peaks"] = fft_peaks
            features[waveform_name + "_fft_peaks_heights"] = fft[fft_peaks]
            features[waveform_name + "_fft_peaks_neighbor_avgs"] = fft_peaks_neighbor_avgs
        
        # Time domain features
        p2p = np.median(np.diff(sys_peaks))
        if one_cycle_sig:
            p2p = len(self.data)
        
        waveforms_cycles_match_peak = extract_cycles_all_ppgs(waveforms, sys_peaks, p2p, "sys_peak", remove_first)
        waveforms_cycles_match_valleys = extract_cycles_all_ppgs(waveforms, sys_peaks, p2p, "dia_notches", remove_first)
        
        if one_cycle_sig:
            waveforms_cycles_match_valleys = {
                "ppg_cycles": [],
                "vpg_cycles": [],
                "apg_cycles": [],
                "ppg3_cycles": [],
                "ppg4_cycles": []
            }
            for waveform_name in waveforms:
                waveforms_cycles_match_valleys[waveform_name + "_cycles"] = np.array([waveforms[waveform_name]], dtype=object)
            
            waveforms_cycles_match_peak = copy.deepcopy(waveforms_cycles_match_valleys)
            
        
        if len(waveforms_cycles_match_peak["ppg_cycles"]) == 0:
            raise RuntimeError("Feature extractor warning: There are no cycles")
        if len(waveforms_cycles_match_valleys["ppg_cycles"]) == 0:
            raise RuntimeError("Feature extractor warning: There are no cycles 2")
        
        # HR and peak to peak distance
        #features["hr"] = self.hr()
        features["hr"] = self.fs / p2p * 60 
        features["p2p"] = p2p
        
        # average avg of signals nearby max. 
        # min positions
        for waveform_name in waveforms:
            cycles = waveforms_cycles_match_peak[waveform_name + "_cycles"]
            mean_cycles, norm_cycles = mean_norm_cycles(cycles, resample_length)
            if "ppg" == waveform_name:
                features["ppg_mean_cycles_match_peak"] = mean_cycles
            neighbor_mean = max_neighbor_mean(mean_cycles, neighbor_mean_size)
            features[waveform_name + "_max_neighbor_mean"] = neighbor_mean
            features[waveform_name + "_min"] = np.argmin(mean_cycles)
            
        # calculate ppg_max_idx
        ppg_cycles_match_valleys = waveforms_cycles_match_valleys["ppg_cycles"]
        ppg_mean_cycles_match_valleys, ppg_norm_cycles_match_valleys = mean_norm_cycles(ppg_cycles_match_valleys, resample_length)
        ppg_max_idx = np.argmax(ppg_mean_cycles_match_valleys)
        
        # average avg of signals nearby max. 
        for waveform_name in waveforms:
            cycles = waveforms_cycles_match_valleys[waveform_name + "_cycles"]
            mean_cycles, norm_cycles = mean_norm_cycles(cycles, resample_length)
            if "ppg" == waveform_name:
                features["ppg_mean_cycles_match_valleys"] = mean_cycles
            H_up, H_down, bins_up, bins_down = histogram_up_down(mean_cycles, num_up_bins, num_down_bins, ppg_max_idx)
            features[waveform_name + "_histogram_up"] = H_up
            features[waveform_name + "_histogram_down"] = H_down
            features[waveform_name + "_max"] = np.argmax(mean_cycles)
        
        # using cycles_match_peak to fix len of features
        usdc = USDC(ppg_norm_cycles_match_valleys, USDC_resample_length)
        dsdc = DSDC(ppg_norm_cycles_match_valleys, DSDC_resample_length)
        features["usdc"] = usdc
        features["dsdc"] = dsdc
        
        # generate header and csv
        header, features_csv_str = generate_features_csv_string(features)
        
        return features, header, features_csv_str
        


