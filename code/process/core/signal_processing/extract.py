from pyampd.ampd import find_peaks
from core.signal_processing.utils import waveform_norm, signal_fft, get_fft_peaks, fft_peaks_neighbor_avg, extract_cycles_all_ppgs, \
                   mean_norm_cycles, align_peaks_notches, max_neighbor_mean, histogram_up_down, USDC, DSDC, generate_features_csv_string
import biosppy
import numpy as np
import scipy

from sklearn.decomposition import PCA

class PPG:
    def __init__(self, data, fs):
        self.data = data
        self.idata = data.max() - data
        self.fs = fs
        self.cycle_size = 128
    def peaks(self, **kwargs): # systolic peaks
        # x: ppg signal
        peaks = find_peaks(self.data, scale=int(self.fs))  # 210 bpm, scale is window size
        
        # Vital sign data range https://www.researchgate.net/figure/Human-Vital-Sign-Data-Ranges_tbl1_220987882
        #peaks = scipy.signal.find_peaks(self.data, distance=35, height=self.data.mean())[0]  # Acounts of 210 bpm
        return peaks      
    
    def valleys(self, **kwargs): # systolic peaks
        # x: ppg signal
        peaks = find_peaks(self.idata, scale=int(self.fs))  # 210 bpm, scale is window size

        # Vital sign data range https://www.researchgate.net/figure/Human-Vital-Sign-Data-Ranges_tbl1_220987882
        # peaks = scipy.signal.find_peaks(self.data, distance=35, height=self.data.mean())[0]  # Acounts of 210 bpm
        return peaks      
    
    def cycles(self, **kwargs): 
        '''
        Extract cylces of the ppg data
        '''
        x = self.data
        fs = self.fs 

        #peaks = scipy.signal.find_peaks(x, distance=35, height=x.mean())[0]
        peaks = find_peaks(self.data, scale=int(self.fs))
        cycle_width = np.median(peaks[1:] - peaks[:-1])

        cycles = []
        cycle_width = int(cycle_width)
        for p in peaks:
            start = p - np.round(cycle_width * 0.25).astype("int")
            end = p + np.round(cycle_width * 0.75).astype("int")
            data = x[start:end]

            if len(data) != int(cycle_width):
                continue
            cycles.append(data.reshape([-1, 1]))
        cycles = waveform_norm(np.concatenate(cycles, -1))
        p_curr = int(cycle_width * 0.25)

        return scipy.signal.resample(cycles, self.cycle_size).transpose((1, 0))  # Returns a matrix of size: (N, 150)

    def vpg(self, **kwargs):
        vpg = self.data[1:] - self.data[:-1]
        padding = np.zeros(shape=(1))
        vpg = np.concatenate([padding, vpg], axis=-1)
        return vpg

    def apg(self, **kwargs):
        apg = self.data[1:] - self.data[:-1]  # 1st Derivative
        apg = apg[1:] - apg[:-1]  # 2nd Derivative

        padding = np.zeros(shape=(2))
        apg = np.concatenate([padding, apg], axis=-1)
        return apg
    
    def ppg3(self, **kwargs):
        ppg3 = self.data[1:] - self.data[:-1]  # 1st Derivative
        ppg3 = ppg3[1:] - ppg3[:-1]  # 2nd Derivative
        ppg3 = ppg3[1:] - ppg3[:-1]  # 3nd Derivative

        padding = np.zeros(shape=(3))
        ppg3 = np.concatenate([padding, ppg3], axis=-1)
        return ppg3
    
    def ppg4(self, **kwargs):
        ppg4 = self.data[1:] - self.data[:-1]  # 1st Derivative
        ppg4 = ppg4[1:] - ppg4[:-1]  # 2nd Derivative
        ppg4 = ppg4[1:] - ppg4[:-1]  # 3nd Derivative
        ppg4 = ppg4[1:] - ppg4[:-1]  # 4nd Derivative

        padding = np.zeros(shape=(4))
        ppg4 = np.concatenate([padding, ppg4], axis=-1)
        return ppg4

    def time_span(self, **kwargs): pass
    def waveform_area(self, **kwargs): pass
    def power_area(self, **kwargs): pass
    def ratio(self, **kwargs): pass
    def slope(self, **kwargs): pass
    def hr(self, **kwargs):
        '''
        Compute the hr from the peak distances in BPM, returns 0 if peaks are not present.
        '''
        try: return self.fs / np.median(np.diff(self.peaks())) * 60
        except: return 0

    def diastolic_notches(self, **kwargs):
        notches = find_peaks(-self.data, scale=int(self.fs))
        #notches = scipy.signal.find_peaks(-self.data, distance=35, height=-self.data.mean())[0]
        
        return notches
        
    def clean_cycles(self, n_sample=None, n_std=1, tolerance = 0.8):
        '''
        Extracts good cycles by removing bad ones from self.cycles

        Parameters:
        - n_sample (int): defines the number of samples we want to randomly sample, if set to None the function would return all signals. In cases where n_sample are not met, the function would return None

        '''
        try:
            old_cycles = self.cycles()
        except:
            N = max(n_sample, 1)
            return np.zeros([N,self.cycle_size])
            
        mean_cycle = np.median(old_cycles, axis=0)
        std_cycle = np.std(old_cycles, axis=0) * n_std

        # Set upper and lower bound
        upper_bound =  np.expand_dims(mean_cycle + std_cycle, axis=0)
        lower_bound = np.expand_dims(mean_cycle - std_cycle, axis=0)

        # Get cycles between upper and lower bound (Set passing threshold to be [80,90]% of the signals are within the std)
        idx_up = (old_cycles < upper_bound).sum(axis=1) >= (mean_cycle.shape[0] * tolerance)
        idx_down = (old_cycles > lower_bound).sum(axis=1) >= (mean_cycle.shape[0] * tolerance)
        
        # Filter the cycles
        clean_cycles = old_cycles[idx_up & idx_down]
        
        # Include the mean into the clean cycles
        mean_cycle = np.expand_dims(mean_cycle, axis=0)
        
        # If none of the cycles satisfy our standard, padd the data with zeros
        if clean_cycles.shape[0] == 0:
            clean_cycles = np.zeros([1, self.cycle_size])
            
        # Append the mean into the clean cycles
        clean_cycles = np.concatenate([mean_cycle, clean_cycles], axis=0)
            
        # Normalize the clean cycles between 0 and 1
        clean_cycles = (clean_cycles - clean_cycles.min()) / (clean_cycles.max() - clean_cycles.min())

        # Output
        if n_sample < 0: return clean_cycles
        
        elif clean_cycles.shape[0] < n_sample:
            pad_size = n_sample - clean_cycles.shape[0]
            clean_cycles = np.concatenate([clean_cycles, np.zeros([pad_size, self.cycle_size])])
            return clean_cycles

        # Normal case
        N = np.arange(clean_cycles.shape[0])
        idx = np.random.choice(N, n_sample, replace=False)
        return clean_cycles[idx]
    
    def cycles_stat_representation(self,n_pca_components, n_std, tolerance):
        '''
        Extracts good cycles by removing bad ones from self.cycles

        Parameters:
        - n_sample (int): defines the number of samples we want to randomly sample, if set to None the function would return all signals. In cases where n_sample are not met, the function would return None

        '''
        try:
            old_cycles = self.cycles()
        except:
            return np.zeros([4 + n_pca_components,self.cycle_size])
            
        mean_cycle = np.median(old_cycles, axis=0)
        std_cycle = np.std(old_cycles, axis=0) * n_std

        # Set upper and lower bound
        upper_bound =  np.expand_dims(mean_cycle + std_cycle, axis=0)
        lower_bound = np.expand_dims(mean_cycle - std_cycle, axis=0)

        # Get cycles between upper and lower bound (Set passing threshold to be [80,90]% of the signals are within the std)
        idx_up = (old_cycles < upper_bound).sum(axis=1) >= (mean_cycle.shape[0] * tolerance)
        idx_down = (old_cycles > lower_bound).sum(axis=1) >= (mean_cycle.shape[0] * tolerance)
        
        # Filter the cycles
        clean_cycles = old_cycles[idx_up & idx_down]
        
        N = clean_cycles.shape[0]
        if N <=1 :
            return np.zeros([4 + n_pca_components,self.cycle_size])

        # Stack the features
        out = []
        out.append(waveform_norm(np.mean(clean_cycles, axis=0)))
        out.append(waveform_norm(np.min(clean_cycles, axis=0)))
        out.append(waveform_norm(np.max(clean_cycles, axis=0)))
        out.append(waveform_norm(np.std(clean_cycles, axis=0)))
        
        if N < n_pca_components:
            pad_size = n_pca_components - N
            padding = np.zeros([pad_size, self.cycle_size])
            clean_cycles = np.concatenate([clean_cycles, padding], axis=0)
            
        pca = PCA(n_pca_components)
        pca.fit(clean_cycles)

            
        for i in range(n_pca_components):
            pca_component = waveform_norm(pca.components_[i] * pca.explained_variance_ratio_[i])
            out.append(pca_component)
        out = np.stack(out)
        
        return out
        
        
    def features_extractor(self, **kwargs):
        
        """
        Description:
        Extract all features from ppg
        
        Return:
        features: {"{feature_name}": feature_value, ...}
        header: [{feature_name_0}, {feature_name_1}, {feature_name_2}...]       features with the same name will add _1 _2 _3 as suffix, _0 is a fixed suffix
        features_csv_str: values of features in csv string format
        
        Usage:
        ppg = PPG(ppg, 125)
        features, header, features_csv_str = ppg.features_extractor()
        """
        
        features = {}
        
        # default parameters
        fft_peak_distance = 28
        fft_neighbor_avg_interval = 6
        resample_length = 80
        neighbor_mean_size = 5
        num_up_bins = 5
        num_down_bins = 10
        USDC_resample_length = resample_length // 4
        DSDC_resample_length = resample_length // 4 * 3

        # normalize ppg, vpg, apg, ppg3, ppg4
        ppg = self.data
        ppg = waveform_norm(ppg)
        vpg = waveform_norm(self.vpg())
        apg = waveform_norm(self.apg())
        ppg3 = waveform_norm(self.ppg3())
        ppg4 = waveform_norm(self.ppg4())
        
        # get peaks and valleys
        sys_peaks = find_peaks(self.data, scale=int(self.fs))
        dia_notches = self.diastolic_notches()
        
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
            freq, fft = signal_fft(waveforms[waveform_name], self.fs, None)
            fft = fft / np.linalg.norm(fft)
            fft_peaks = get_fft_peaks(fft, freq, fft_peak_distance, num_iter=5)
            fft_peaks_neighbor_avgs = fft_peaks_neighbor_avg(fft, fft_peaks, fft_neighbor_avg_interval)
            features[waveform_name + "_fft_peaks"] = fft_peaks
            features[waveform_name + "_fft_peaks_heights"] = fft[fft_peaks]
            features[waveform_name + "_fft_peaks_neighbor_avgs"] = fft_peaks_neighbor_avgs
        
        # Time domain features
        p2p = np.median(np.diff(sys_peaks))
        waveforms_cycles_match_peak = extract_cycles_all_ppgs(waveforms, sys_peaks, p2p, "sys_peak")
        waveforms_cycles_match_valleys = extract_cycles_all_ppgs(waveforms, sys_peaks, p2p, "dia_notches")
        
        if len(waveforms_cycles_match_peak["ppg_cycles"]) == 0:
            raise RuntimeError("Feature extractor warning: There are no cycles")
        if len(waveforms_cycles_match_valleys["ppg_cycles"]) == 0:
            raise RuntimeError("Feature extractor warning: There are no cycles 2")
        
        # HR and peak to peak distance
        features["hr"] = self.hr()
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
        
        
class ECG:
    def __init__(self, data, fs):
        self.data = data
        self.fs = fs
    def peaks(self, **kwargs): # R peaks
        """
        Extract RR peaks from an ECG one-dimensional data, following
        the approach described in http://www.cinc.org/old/Proceedings/2002/pdf/101.pdf

        Parameters
        ----------
        cleaned_data : array
            Input filtered ECG data.
        fs : int, float
            Sampling frequency (Hz).
        correction_tol : int, float, optional
            Correction tolerance (seconds).

        Returns
        -------
        rr_peaks : array
            R-peak location indices.
        """
        # Extract RR peaks
        rr_peaks, = biosppy.ecg.hamilton_segmenter(self.data, self.fs)
        # Correct RR peaks locations
        rr_peaks, = biosppy.ecg.correct_rpeaks(self.data, rr_peaks, self.fs, 0.05)   # correction_tol = 0.05 can get good performance, but it hasn't been tuned.
        return rr_peaks
    def cycles(self, **kwargs): pass
    def rr_peaks(self, **kwargs): pass
    def hr(self, **kwargs):
        return self.fs / np.median(np.diff(self.peaks())) * 60          # bpm

def align_peaks_ecg_ppg(ecg_peaks, ppg_peaks):

    if len(ecg_peaks) == 0 or len(ppg_peaks) == 0:
        return None

    aligned_ecg_peaks = []
    aligned_ppg_peaks = []
    for i in range(ecg_peaks.shape[0]-1):
        min_ecg = ecg_peaks[i]
        max_ecg = ecg_peaks[i+1]

        try: ppg_peak = ppg_peaks[(ppg_peaks > min_ecg) & (ppg_peaks < max_ecg)][0]
        except: continue

        aligned_ecg_peaks.append(min_ecg)
        aligned_ppg_peaks.append(ppg_peak)

    aligned_ecg_peaks = np.array(aligned_ecg_peaks)
    aligned_ppg_peaks = np.array(aligned_ppg_peaks)

    if len(aligned_ecg_peaks) == 0 or len(aligned_ppg_peaks) == 0:
        return None

    return aligned_ecg_peaks, aligned_ppg_peaks
    
def get_ptt(ecg_peaks, ppg_peaks, fs): 
    
    # ecg_peaks: unaligned peaks
    # ppg_peaks: unaligned peaks
    
    aligned = align_peaks_ecg_ppg(ecg_peaks, ppg_peaks)
    
    if (aligned is None) or (len(aligned[0]) == 0):
        # Missing peaks, unable to compute PTT
        return None
    
    (aligned_ecg, aligned_ppg) = aligned
    # Compute the PTT
    ptt = aligned_ppg - aligned_ecg
    zscore = (ptt - ptt.mean()) / ptt.std()
    ptt = ptt[np.abs(zscore) <=3]
    
    if ptt.shape[0] == 0:
        # There's no agreement between the ptts
        return None
    
#     ptt = np.median(aligned_ppg-aligned_ecg) / fs * 1000.0
    ptt = np.mean(aligned_ppg-aligned_ecg) / fs * 1000.0
    return ptt
    
if __name__ == "__main__":
    pass
