import pandas as pd
import numpy as np
import math


from pyampd.ampd import find_peaks

import scipy.signal
from scipy.signal import correlate
from scipy.interpolate import CubicSpline

def normalize_data(x):
    """Min-max normalization of signal waveform.

    Parameters
    ----------
    x : array
        Signal waveform.

    Returns
    -------
    array
        Normalized signal waveform.
    """
    return (x - x.min()) / (x.max() - x.min() + 1e-10)  # 1e-10 avoid division by zero

def waveform_norm(x):
    """Min-max normalization of signal waveform.

    Parameters
    ----------
    x : array
        Signal waveform.

    Returns
    -------
    array
        Normalized signal waveform.
    """
    return (x - x.min())/(x.max() - x.min() + 1e-6)


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    """ Butterworth band-pass filter
    Parameters
    ----------
    data : array
        Signal to be filtered.
    lowcut : float
        Frequency lowcut for the filter. 
    highcut : float}
        Frequency highcut for the filter.
    fs : float
        Sampling rate.
    order: int
        Filter's order.

    Returns
    -------
    array
        Signal filtered with butterworth algorithm.
    """  
    nyq = fs * 0.5  # https://en.wikipedia.org/wiki/Nyquist_frequency
    lowcut = lowcut / nyq  # Normalize
    highcut = highcut / nyq
    # Numerator (b) and denominator (a) polynomials of the IIR filter
    b, a = scipy.signal.butter(order, [lowcut, highcut], btype='band', analog=False)
    return scipy.signal.filtfilt(b, a, data)


def remove_mean(data):
    """Remove the mean from the signal.

    Parameters
    ----------
    x : array
        Signal waveform.

    Returns
    -------
    array
        Processed signal waveform.
    """
    return data-np.mean(data)


def mean_filter_normalize(data, fs, lowcut, highcut, order):
    """
    Wrapper for removing mean, bandpass filter and normalizing the signals between 0-1

    Parameters
    ----------
    x : array
        Signal waveform.
    fs : int
        Sampling rate.
    lowcut : float
        Frequency lowcut for the filter. 
    highcut : float}
        Frequency highcut for the filter.
    order: int
        Filter's order.

    Returns
    -------
    array
        Processed signal waveform.
    """
    data = data-np.mean(data)
    data = butter_bandpass_filter(data, lowcut, highcut, fs, order)
    data = normalize_data(data)
    
    return data


def align_pair(abp, raw_ppg, windowing_time, fs):
    """
    Align ABP and PPG signal passed as parameters using the maximum cross-correlation.
    Only PPG is shifted to align with ABP. The shift is limited to a second as maximum.

    Parameters
    ----------
    abp : array
        ABP signal waveform
    raw_ppg : array
        PPG signal waveform
    windowing_time: int
        Length of the signals in seconds
    fs: int
        Frequency sampling rate (Hz)
    
    Returns
    -------
    array
        Aligned ABP signal waveform
    array
        Aligned PPG signal waveform
    Int
        Number of samples shifted

    """

    window_size = fs * windowing_time # original segment length
    extract_size = fs * (windowing_time-1)

    cross_correlation = correlate(abp, raw_ppg)
    shift = np.argmax(cross_correlation[extract_size:window_size]) #shift must happened within 1s
    shift += extract_size
    start = np.abs(shift-window_size)

    a_abp = abp[:extract_size]
    a_rppg = raw_ppg[start:start+extract_size]

    return a_abp, a_rppg, shift-window_size


def rm_baseline_wander(ppg, vlys, add_pts = True):
    """
    Remove baseline wander (BW) from a signal subtracting an BW estimated with Cubic Spline.

    Parameters
    ----------
    ppg : array
        signal waveform to process.
    vlys: array
        Indices of the valleys of the signal.
    add_pts: bool
        Enable to add points to cover all the signal.
    
    Returns
    -------
    array
        Processed signal waveform without BW.
    array
        Estimated baseline wander.
    array
        Values used to estimate BW.
    array
        Indices of the values used to estimate BW.
    """    

    rollingmin_idx = vlys
    rollingmin = ppg[vlys]
    
    mean = np.mean(rollingmin)
    
    if add_pts == True:
        dist = np.median(np.diff(rollingmin_idx))
        med = np.median(rollingmin)
        
        add_pts_head = math.ceil(rollingmin_idx[0] / dist)
        head_d = [rollingmin_idx[0]-i*dist for i in reversed(range(1,add_pts_head+1))] 
        head_m = [med]*add_pts_head
        
        
        add_pts_tail = math.ceil((len(ppg)-rollingmin_idx[-1]) / dist)
        tail_d = [rollingmin_idx[-1]+ i*dist for i in range(1,add_pts_tail+1)] 
        tail_m = [med]*add_pts_tail 
        
        rollingmin_idx = np.concatenate((head_d, rollingmin_idx, tail_d))
        rollingmin = np.concatenate((head_m, rollingmin, tail_m))
    # polyfit

    cs = CubicSpline(rollingmin_idx, rollingmin)

    # polyval

    baseline = cs(np.arange(len(ppg)))

    # subtract the baseline
    
    rem_line = ppg - (baseline-mean)

    return rem_line, baseline, rollingmin, rollingmin_idx


def identify_out_pk_vly(sig, pk, vly, th=3):
    """
    Identify outliers in the peaks and valleys of the signal passed as parameters.
    Peak or valley is an outlier if it exceeds 'th' times the standard deviation w.r.t. the mean of the peaks/valleys.

    Parameters
    ----------
    sig : array
        signal waveform.
    pk: array
        Indices of the peaks of the signal.
    vly: array
        Indices of the valleys of the signal.
    th: float
        Threshold to identify outliers
    
    Returns
    -------
    list
        Indices of the identified outliers
    """ 

    out_pk, out_vly = -1, -1
    outs = []
    
    vly_val = sig[vly]
    pk_val = sig[pk]
    
    vly_val_idx = vly_val.argmin()
    vly_val_min = vly_val[vly_val_idx]
    vly_val_argmin = vly[vly_val_idx]
    vly_val_mean = vly_val.mean()
    vly_val_std = vly_val.std()
    
    if vly_val_min < vly_val_mean - vly_val_std * th:
        outs.append(vly_val_argmin)
    
    pk_val_idx = pk_val.argmax()
    pk_val_max = pk_val[pk_val_idx]
    pk_val_argmax = pk[pk_val_idx]
    pk_val_mean = pk_val.mean()
    pk_val_std = pk_val.std()
    
    if pk_val_max > pk_val_mean + pk_val_std * th:
        outs.append(pk_val_argmax)
    
    return outs


def my_find_peaks(sig, fs, remove_start_end = True):
    """ 
    Wrapper of find peaks function. If find_peaks fails, an empty array is returned.

    Parameters
    ----------
    sig : array
        signal waveform.
    fs : int
        Sampling rate.
    remove_start_end: bool
        Indices of the valleys of the signal.

    Returns
    -------
    array
        Indices of the peaks found.

    """

    try:
        pks = find_peaks(sig, fs)
        if remove_start_end:
            if pks[0] == 0: pks = pks[1:]

            if pks[-1] == len(sig)-1: pks = pks[:-1]
        return pks
    except:
        return np.array([])






