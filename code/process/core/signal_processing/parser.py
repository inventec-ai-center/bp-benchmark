import numpy as np
import pickle as pkl
import pandas as pd


def load_mimicv1_csv(signal_path, ppg_sqi_th = None):
    '''
    This function is to loead the data from selected subjects with good profile:
    see "src/notebooks/exploration_mimicv1.ipynb" for more detail
    '''
    # Load the raw data 
    _subjects = np.array(["484","225","437","216","417","284","438","471","213","439","237","240","446","281",
                                "476","224","226","427","482","485","443","276","452","472","230"])
    _xsubjectfold = {}
    # mimicv1_meta = pkl.load(open(signal_path+"mimicv1-metadata.pkl", "rb"))
    # mimicv1_meta = json.load(open(signal_path+"../mimic-1.0.0-meta.json"))

    anno = []
    for s in _subjects:
        # Load the .csv
        # df = pd.read_csv(signal_path+"mimicv1-anno/{}.csv".format(s))  
        df = pd.read_csv(signal_path+"anno/{}.csv".format(s)) 
        # Apply filtering over the rows, the following thresholds (obtained through eye-balling)
        mask = (df.sbp_std < 10) \
                & ~(pd.isnull(df.ptt)) \
                & (df.dbp_std < 5) \
                & (df.ptt.between(100,600)) \
                & (df.sbp_mean.between(60,200)) \
                & (df.dbp_mean.between(30,160))
        df = df[mask].reset_index(drop=True)

        if (ppg_sqi_th is None) == False:
            df = df[df.ppg_sqi >= ppg_sqi_th]            
        anno.append(df)
        
    anno = pd.concat(anno).reset_index(drop=True)
    anno["sbp"] = anno["sbp_mean"]
    anno["dbp"] = anno["dbp_mean"]
    anno["subject_id"] = anno["subject_id"].astype("str")
    
    return anno
