"""
The Signal Quality Index module provides multiple metrics to assess the quality of ECG and PPG signals. These scores can be used to filter the bad-quality signals. 

"""

import numpy as np

def _is_flat(x, threshold=0.3):
    """ Identify if a signal has flat regions greater than a given threshold

    Parameters
    ----------
    x : array
        Raw signal
    threshold : float
        Threshold of percentage of flat region [0-1].
    
    Returns
    -------
    bool
        True if the signal is flat for longer than the threshold

    Notes
    -----
    threshold=0.3 is generally recomended.
    """

    delta=1e-5
    dx = x[1:] - x[:-1]
    flat_parts = (np.abs(dx) < delta).astype("int").sum()
    flat_parts = flat_parts/x.shape[0]
    
    # Reject the segment if 30% of the signals are flat
    return flat_parts > threshold


def skew(x, is_flat_threshold): 
    """ Skewness of the signal given as parameter.
    Skewness measures of the symmetry of a probability distribution. It is associated with 
    Implemention of different metrics to assess the quality of a signal (PPG or ECG).

    Parameters
    ----------
    x : array
        Target signal, of which we want to measure the SQI.  
    is_flat_threshold : float
        Function parameters. kwargs['is_flat_threshold'] is the threshold of percentage of flat region.

    Returns
    -------
    float
        SQI skewness of the target signal.

    Notes
    -----
    threshold=0.3 can get normal boxplot. See more details in comments for function is_flat.
    Higher values of Skewness usually represent better quality.
    It is recommended not to exceed a signal window 3-5 seconds.

    References
    ----------
    .. [1] Elgendi, M. (2016). Optimal signal quality index for photoplethysmogram signals. Bioengineering, 3(4), 21. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5597264/)
    """
    return np.sum(((x - x.mean())/(x.std()+1e-6))**3)/x.shape[0] * ~_is_flat(x, threshold=is_flat_threshold)


def kurtosis(x): 
    """ Kurtosis of the signal given as parameter. 
    Kurtosis describes the distribution of observed data around the mean.

    Parameters
    ----------
    x : array
        Raw signal, of which we want to measure the SQI.
    
    Returns
    -------
    float
        SQI kurtosis of the target signal.
    
    References
    ----------
    .. [1] Elgendi, M. (2016). Optimal signal quality index for photoplethysmogram signals. Bioengineering, 3(4), 21. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5597264/)
    """

    sqi_score = 1/(np.sum(((x - x.mean())/(x.std()+1e-6))**4)/(x.shape[0]+1e-6) + 1e-6) # reciprocal kurtosis

    if np.isnan(sqi_score) or np.isinf(sqi_score):
        return 0.0

    return sqi_score

