import numpy as np 
import scipy
from scipy import signal
import os
import json
from core.signal_processing.utils import waveform_norm, signal_fft, extract_cycles
from core.signal_processing.extract import ECG, PPG

class SQI:
    def __init__(self, algorithm):
        '''
        Takes in 1-D signal and return the corresponding score
        '''        
        algorithms = {  "skew": self.skew,
                        "perfusion": self.perfusion,
                        "kurtosis": self.kurtosis,
                        "entropy": self.entropy,
                        "zero_crossing": self.zero_crossing,
                        "snr": self.snr,
                        "relative_power": self.relative_power,
                        "cycle_l2": self.cycle_l2,
                        "cycle_corr": self.cycle_corr,
                        "skew_moving_average": self.skew_moving_average,
                        "moving_average": self.moving_average,
                        "smas": self.smas,                                        # smas: sqaure of MA * skew
                        "smas_out": self.smas_out,                                # smas_out: smas and remove outliers
                        "ma_out": self.ma_out,                                   # ma_out: moving_average and remove outlier
                        "template_matching": self.template_matching}              
        self.score = algorithms[algorithm.lower()]

    def skew(self, x, **kwargs): 
        
        # Parameters
        # x: Target signal, of which we want to measure the SQI    
        # kwargs['threshold']: threshold of percentage of plat region
        #
        # Comments
        # threshold=0.3 can get normal boxplot. See more details in comments for function is_flat.

        if 'configs' in kwargs:
            assert len(kwargs) == 1                  # prevent from using both kwargs and alg_configs
            alg_configs = kwargs['configs']
            is_flat_threshold = alg_configs['is_flat_threshold']
        else:
            alg_configs = kwargs
            is_flat_threshold = alg_configs['is_flat_threshold']
        
        return np.sum(((x - x.mean())/(x.std()+1e-6))**3)/x.shape[0] * ~self.is_flat(x, threshold=is_flat_threshold)
    
    def perfusion(self, x, y): 
        # x: Raw signal
        # kwargs['y']: Cleaned signal
        try:
            return (y.max()-y.min())/(np.abs(x.mean())+1e-6) * 100
        except Exception as e:
            raise
    
    def kurtosis(self, x): 
        # x: Target signal, of which we want to measure the SQI
        sqi_score = 1/(np.sum(((x - x.mean())/(x.std()+1e-6))**4)/(x.shape[0]+1e-6) + 1e-6) # reciprocal kurtosis
    
        if np.isnan(sqi_score) or np.isinf(sqi_score):
            return 0.0

        return sqi_score
    
    def entropy(self, data, **kwargs): pass
    def zero_crossing(self, data, **kwargs): pass
    def snr(self, data, **kwargs): pass
    def relative_power(self, x, **kwargs):
        
        if 'configs' in kwargs:
            assert len(kwargs) == 1                  # prevent from using both kwargs and alg_configs
            alg_configs = kwargs['configs']
            fs = alg_configs['fs']
        else:
            alg_configs = kwargs
            fs = alg_configs['fs']
        
        # get frequencies of the signal
        freq, abs_org_fft = signal_fft(x, fs, False)
        
        # ratio of signal strength between 1 ~ 2.25 Hz and signal strength between 0 ~ 8 Hz
        # From paper: Optimal Signal Quality Index for Photoplethysmogram Signals
        sqi_score = abs_org_fft[(freq>=1) & (freq<=2.25)].sum() / (abs_org_fft[(freq>=0) & (freq<=8)].sum() + 1e-6)
        if np.isnan(sqi_score):
            return 0.0
        
        return sqi_score
        
    def cycle_l2(self, data, **kwargs): pass
    def cycle_corr(self, data, **kwargs): pass
    
    def skew_moving_average(self, x, **kwargs):
        
        # Description
        # Good SQI for PPG only, score < 3.46 is BAD, 3.46 <= score <= 22.56 is OK score > 22.56 is GOOD, the boundaries are defined by Q3 + 1.5 * IRS
        # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.boxplot.html
        # final score = skew score * moving average score
        # return final score
        #
        # Parameters
        # x: Target signal
        #
        # Comments 
        # window_size = 100 has better performance on both training set and test set on an experiment for SQI
        
        if 'configs' in kwargs:                                           # using configs
            assert len(kwargs) == 1                                       # prevent from using both kwargs and configs
            alg_configs = kwargs['configs']
            is_flat_threshold = alg_configs['is_flat_threshold']
            window_size = alg_configs['ma_window_size']
        else:
            alg_configs = kwargs
            is_flat_threshold = alg_configs['is_flat_threshold']          # using parameters
            window_size = alg_configs['ma_window_size']
        
        # calculate moving average score
        data = waveform_norm(x)
        moving_avg = np.convolve(data, np.ones((window_size,))/window_size, mode='valid')
        ma_score = (1 / (np.std(moving_avg) + 1e-6)) * ~self.is_flat(data, threshold=is_flat_threshold)
        
        # calculate skew score after remove outliers by std, outliers are defined by large than Q3 + 1.5 * IRS
        # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.boxplot.html
        Q3 = np.percentile(data, 75)
        Q1 = np.percentile(data, 25)
        IRS = Q3 - Q1
        upper_bound = Q3 + IRS * 1.5
        max_outliers_mask = data > upper_bound
        data[max_outliers_mask] = upper_bound
        skew_score = self.skew(data, is_flat_threshold=is_flat_threshold, ma_window_size=window_size) * \
                     ~self.is_flat(data, threshold=is_flat_threshold)
        
        return skew_score * (ma_score**2)
        
    def moving_average(self, x, **kwargs):
        
        # Description
        # When we want good signals, this method can be applied for both ECG and PPG, but for PPG we can use skew_moving_average instead
        # For ECG, score > 213.51 is GOOD on self-labeled dataset. but this SQI can't distinguish bad and good signals when scores are low.
        #
        # Parameters
        # x: Target signal
        # window_size: window_size of moving avg
        #
        # Comments
        # window_size = 100 has better performance on both training set and test set on an experiment for SQI
        
        if 'configs' in kwargs:                                           # using configs
            assert len(kwargs) == 1                                       # prevent from using both kwargs and configs
            alg_configs = kwargs['configs']
            is_flat_threshold = alg_configs['is_flat_threshold']
            window_size = alg_configs['ma_window_size']
        else:
            alg_configs = kwargs
            is_flat_threshold = alg_configs['is_flat_threshold']          # using parameters
            window_size = alg_configs['ma_window_size']
        
        data = waveform_norm(x)
        moving_avg = np.convolve(data, np.ones((window_size,))/window_size, mode='valid')
        return (1 / (np.std(moving_avg) + 1e-6)) * ~self.is_flat(data, threshold=is_flat_threshold)
    
    def smas(self, x, **kwargs):
        
        # Description
        # Square of moving average SQI * skew SQI
        #
        # x: signal
        
        if 'configs' in kwargs:                                           # using configs
            assert len(kwargs) == 1                                       # prevent from using both kwargs and configs
            alg_configs = kwargs['configs']
            is_flat_threshold = alg_configs['is_flat_threshold']
            window_size = alg_configs['ma_window_size']
        else:
            alg_configs = kwargs
            is_flat_threshold = alg_configs['is_flat_threshold']          # using parameters
            window_size = alg_configs['ma_window_size']
        
        # calculate skew score after remove outliers by std, outliers are defined by large than Q3 + 1.5 * IRS
        # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.boxplot.html
        Q3 = np.percentile(x, 75)
        Q1 = np.percentile(x, 25)
        IRS = Q3 - Q1
        upper_bound = Q3 + IRS * 1.5
        max_outliers_mask = x > upper_bound
        skew_x = x.copy()
        skew_x[max_outliers_mask] = upper_bound
        
        skew_score = self.skew(skew_x, is_flat_threshold=is_flat_threshold, ma_window_size=window_size)
        ma_score = self.moving_average(x, is_flat_threshold=is_flat_threshold, ma_window_size=window_size)
               
        return  skew_score * (ma_score ** 2)
    
    def smas_out(self, x, signal_type, fs, **kwargs):
        
        # Description
        # Square of moving average SQI * skew SQI and remove outliers
        #
        # x: singal
        # signal_type: "PPG" or "ECG", this SQI is only tested on PPG
        # fs: sample rate
        
        if 'configs' in kwargs:                                           # using configs
            assert len(kwargs) == 1                                       # prevent from using both kwargs and configs
            alg_configs = kwargs['configs']
            is_flat_threshold = alg_configs['is_flat_threshold']
            window_size = alg_configs['ma_window_size']
        else:
            alg_configs = kwargs
            is_flat_threshold = alg_configs['is_flat_threshold']          # using parameters
            window_size = alg_configs['ma_window_size']
        
        # check signal_type for peak detection
        if signal_type == "ECG":
            data = ECG(x, fs)
        elif signal_type == "PPG":
            data = PPG(x, fs)
        else:
            raise Exception("signal_type doesn't exist.")
        
        # peak detection for removing outlier later
        try:
            peaks = data.peaks()
        except: # No peak
            return 0.0
        
        if len(peaks) < len(data.data) / fs / (60/30):           # number of peaks are less than peaks with 30 bpm
            return 0.0
        
        # remove first and last peak, because the start point and end point might be misjudged as a peak
        peaks = peaks[1:-1]
        
        # score before remove outliers
        base_SQI = self.smas(x, is_flat_threshold=is_flat_threshold, ma_window_size=window_size)
        
        # remove outliers by height
        Q3 = np.percentile(data.data[peaks], 75)
        Q1 = np.percentile(data.data[peaks], 25)
        IRS = Q3 - Q1
        upper_bound = Q3 + IRS * 1.5
        lower_bound = Q1 - IRS * 1.5
        max_outliers_mask = data.data[peaks] > upper_bound
        min_outliers_mask = data.data[peaks] < lower_bound
        factor = sum(max_outliers_mask + min_outliers_mask) + 1            # count number of outliers + 1
        
        return base_SQI / factor
    
    def ma_out(self, x, signal_type, fs, **kwargs):
        
        # Description
        # Square of moving average SQI and remove outliers
        #
        # x: singal
        # signal_type: "PPG" or "ECG", this SQI is only tested on PPG
        # fs: sample rate
        
        if 'configs' in kwargs:                                           # using configs
            assert len(kwargs) == 1                                       # prevent from using both kwargs and configs
            alg_configs = kwargs['configs']
            is_flat_threshold = alg_configs['is_flat_threshold']
            window_size = alg_configs['ma_window_size']
        else:
            alg_configs = kwargs
            is_flat_threshold = alg_configs['is_flat_threshold']          # using parameters
            window_size = alg_configs['ma_window_size']
        
        # check signal_type for peak detection
        if signal_type == "ECG":
            data = ECG(x, fs)
        elif signal_type == "PPG":
            data = PPG(x, fs)
        else:
            raise Exception("signal_type doesn't exist.")
        
        # peak detection for removing outlier later
        try:
            peaks = data.peaks()
        except: # No peak
            return 0.0
        
        if len(peaks) < len(data.data) / fs / (60/30):           # number of peaks are less than peaks with 30 bpm
            return 0.0
        
        # remove first and last peak, because the start point and end point might be misjudged as a peak
        peaks = peaks[1:-1]
        
        # score before remove outliers
        base_SQI = self.moving_average(x, is_flat_threshold=is_flat_threshold, ma_window_size=window_size)
        
        # remove outliers by height
        Q3 = np.percentile(data.data[peaks], 75)
        Q1 = np.percentile(data.data[peaks], 25)
        IRS = Q3 - Q1
        upper_bound = Q3 + IRS * 1.5
        lower_bound = Q1 - IRS * 1.5
        max_outliers_mask = data.data[peaks] > upper_bound
        min_outliers_mask = data.data[peaks] < lower_bound
        factor = sum(max_outliers_mask + min_outliers_mask) + 1            # count number of outliers + 1
        
        return base_SQI / factor
    
    def template_matching(self, x, signal_type, fs, **kwargs):
        
        # Description
        # template matching SQI
        #
        # x: singal
        # signal_type: "PPG" or "ECG", this SQI is only tested on PPG
        # fs: sample rate
        
        if 'configs' in kwargs:                                           # using configs
            assert len(kwargs) == 1                                       # prevent from using both kwargs and configs
            alg_configs = kwargs['configs']
        else:
            alg_configs = kwargs
        
        # check signal_type for peak detection
        if signal_type == "ECG":
            data = ECG(x, fs)
        elif signal_type == "PPG":
            data = PPG(x, fs)
        else:
            raise Exception("signal_type doesn't exist.")
        
        # peak detection for template matching
        try:
            peaks = data.peaks()
        except: # No peak
            return 0.0
        
        if len(peaks) < len(data.data) / fs / (60/30):           # number of peaks are less than peaks with 30 bpm
            return 0.0
        
        # remove first and last peak, because the start point and end point might be misjudged as a peak
        peaks = peaks[1:-1]
    
        # template matching
        cycles = extract_cycles(data.data, peaks, int(data.hr() / 2))
        template = cycles.mean(axis=1)
        corrs = []
        for idx in range(cycles.shape[1]):
            corr = np.corrcoef(cycles[:,idx], template)
            corrs.append(corr[0,1])
        return np.array(corrs).mean()
    
    #-------------------------- utils --------------------------
        
    def is_flat(self, x, **kwargs):
        # x: signal
        # comment: 10% misclassifies both GOOD and OK signals as flat signals, but they are not 10% flat signals when we eyeball them, those flat regions are part of waveform.
        #
        # Parameters
        # x: Target signal, of which we want to measure the SQI    
        # kwargs['threshold']: threshold of percentage of flat region
        #
        # Comments
        # threshold=0.3 can get normal boxplot. See more details in comments for function is_flat.
        
        if 'configs' in kwargs:                                           # using configs
            assert len(kwargs) == 1                                       # prevent from using both kwargs and configs
            alg_configs = kwargs['configs']
            is_flat_threshold = alg_configs['is_flat_threshold']
        else:
            alg_configs = kwargs
            is_flat_threshold = alg_configs['threshold']                  # using parameters
        
        delta=1e-5
        dx = x[1:] - x[:-1]
        flat_parts = (np.abs(dx) < delta).astype("int").sum()
        flat_parts = flat_parts/x.shape[0]
        
        # Reject the segment if 30% of the signals are flat
        return flat_parts > is_flat_threshold
    
class filtering:
    def __init__(self, algorithm):
        '''
        Either or both lowcut and highcut must be set. 
        lowcut != None --> lowpass
        highcut != None --> highpass
        both != None --> bandpass
        '''
        algorithms = {  "butterworth": self.butterworth,
                        "cheby2": self.cheby2,
                        "remove_baseline_wander": self.remove_baseline_wander}
        self.apply = algorithms[algorithm.lower()]
        
    def butterworth(self, data, lowcut, highcut, fs, order):             
                       
        nyq = fs * 0.5  # https://en.wikipedia.org/wiki/Nyquist_frequency
        if (lowcut is not None) & (highcut is not None):
            # Band Pass filter
            lowcut = lowcut / nyq
            highcut = highcut / nyq
            b, a = scipy.signal.butter(order, [lowcut, highcut], btype='band', analog=False)
            
        elif lowcut:
            # Low Pass filter
            lowcut = lowcut / nyq  
            b, a = scipy.signal.butter(order, lowcut, btype='low', analog=False)
            
        elif highcut:
            # High Pass filter
            highcut = highcut / nyq
            b, a = scipy.signal.butter(order, highcut, btype='high', analog=False)
        
        return scipy.signal.filtfilt(b, a, data)
    
    def cheby2(self, data, rs, lowcut, highcut, fs, order): 
    
        # Parameters
        # rs can be 20, it means reduce 20dB, the intensity will reduce 100 times.
    
        nyq = fs * 0.5  # https://en.wikipedia.org/wiki/Nyquist_frequency
        if (lowcut is not None) & (highcut is not None):
            # Band Pass filter
            lowcut = lowcut / nyq  
            highcut = highcut / nyq
            b, a = scipy.signal.cheby2(order, rs, [lowcut, highcut], btype='bandpass', analog=False)
            
        elif lowcut:
            # Low Pass filter
            lowcut = lowcut / nyq  
            b, a = scipy.signal.cheby2(order, rs, lowcut, btype='low', analog=False)
            
        elif highcut:
            # High Pass filter
            highcut = highcut / nyq
            b, a = scipy.signal.cheby2(order, rs, highcut, btype='high', analog=False)
            
        return scipy.signal.filtfilt(b, a, data)
    
    def remove_baseline_wander(self, data, **kwargs): pass

if __name__ == "__main__":
    '''
    Examples:
        > SQI("skew").score(signal_1d)
        > filtering("butterworth", lowcut=0.05, highcut=10, fs=125, order=4).apply(signal_1d)
    '''
    filt = filtering("butterworth")
    print(filt.apply(np.random.randn(100), fs=125, order=2, lowcut=0.05, highcut=5))
    
