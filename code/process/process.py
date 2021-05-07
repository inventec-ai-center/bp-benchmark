import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import pickle as pkl
from glob import glob 
import os
import wfdb
import json
from tqdm import tqdm

import sys
PATHS = json.load(open("./paths.json"))
for k in PATHS:  sys.path.append(PATHS[k])
import core.signal_processing
from core.signal_processing.clean import filtering, SQI
from core.signal_processing.extract import PPG, ECG, get_ptt, align_peaks_ecg_ppg
from core.signal_processing.utils import global_norm, global_denorm, get_bp_labels

import argparse

DEFAULT_CONF_PATH = "core/config/default_config.json"


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", type=str, default=DEFAULT_CONF_PATH,
                        help="path to the config file")
    parser.add_argument("--db_path", default="../../datasets/mimic-database-1.0.0/") 

    return parser

def mimicv1_get_valid_records(db_path):
    valid, invalid = ([], [])
    
    # Get all the records name
    records = [x for x in os.listdir(db_path) if (len(x) == 3) & (x != "DOI")]
    
    for rec in records:
        # Get headers
        paths = glob(db_path + "{}/*00*.hea".format(rec))

        # Check all the sub-records if they are valid, otherwise skip the records
        rec_valid = True
        for path in paths:
            tmp = "".join(open(path).readlines())
            is_ppg = "PLETH" in tmp
            is_ecg = " II\n" in tmp
            is_abp = "ABP" in tmp
            rec_valid = is_ppg & is_ecg & is_abp & rec_valid
        if rec_valid: valid.append(rec)
        else: invalid.append(rec)
            
    # This assertion is a strong requirement as the previous functions are deterministic and should not change the number of valid records
    print("Number of valid records: ",len(valid))
    print("Number of invalid records: ", len(invalid))
    assert (len(valid) == 38) & (len(invalid) == 34)
    
    return valid, invalid

def get_add_feature_cols():
    mapping={}
    mapping["ppg_histogram_up"]=5
    mapping["ppg_histogram_down"]=10
    mapping["vpg_histogram_up"]=5
    mapping["vpg_histogram_down" ]=10
    mapping["apg_histogram_up"]=5
    mapping["apg_histogram_down" ]=10
    mapping["ppg_fft_peaks_heights"]=5
    mapping["ppg_fft_peaks_neighbor_avgs"]=5
    mapping["vpg_fft_peaks_heights"]=5
    mapping["vpg_fft_peaks_neighbor_avgs"]=5
    mapping["apg_fft_peaks_heights"]=5
    mapping["apg_fft_peaks_neighbor_avgs"]=5
    mapping["usdc"]=6
    mapping["dsdc"]=9

    add_cols = []
    for item in mapping:
        for itr in range(mapping[item]):
            add_cols.append(item+"_{}".format(itr))                    
    return add_cols

def generate_anno(db_path, allmeta, fs=125, ma_window_size=100, is_flat_threshold=0.5):
    # Names of additional hand-engineered features
    add_features = ["hr","p2p","ppg_histogram_up","ppg_histogram_down","vpg_histogram_up","vpg_histogram_down",
                    "apg_histogram_up","apg_histogram_down","ppg_fft_peaks_heights","ppg_fft_peaks_neighbor_avgs","vpg_fft_peaks_heights",
                    "vpg_fft_peaks_neighbor_avgs","apg_fft_peaks_heights","apg_fft_peaks_neighbor_avgs","usdc","dsdc"]
    add_feature_cols =["hr","p2p"] + get_add_feature_cols()

    # Define SQI for each signal
    psqi = SQI("smas_out")
    esqi = SQI("moving_average")

    # Note valid and invalid subjects
    valid, invalid = mimicv1_get_valid_records(db_path)

    # For each subject in allmeta
    for s in allmeta:
        meta = allmeta[s]
        subject_df = pd.DataFrame(columns=["subject_id","record_id","chunk_id","ptt","sbp_mean","sbp_std","dbp_mean","dbp_std","rec_date","rec_time","ecg_sqi","ppg_sqi","signal_path"] + add_feature_cols)

        for rec in meta["records"]:       
            # Check if the record is valid
            if rec not in valid: continue
            print("Processing Subject {}, Record {}".format(s,rec))    
            
            # Get the 10 minutes chunks
            rec_path = db_path + "{}/*00*.hea".format(rec)
            paths = [x[:-4] for x in glob(rec_path)]

            # Load the data for each of the paths
            for path in tqdm(paths[:10]):
                r = wfdb.rdrecord(path)
                
                # Load the signal 
                ppg = r.p_signal[:, r.sig_name.index("PLETH")]
                ecg = r.p_signal[:, r.sig_name.index("II")]
                abp = r.p_signal[:, r.sig_name.index("ABP")]
                
                base_date = r.base_date.strftime("%Y-%m-%d")
                base_time = r.base_time.strftime("%H-%M-%S")
                
                # Fill all NaN values with zeroes
                ppg[np.isnan(ppg)] = 0
                ecg[np.isnan(ecg)] = 0
                
                assert ppg.shape[0] == ecg.shape[0] == abp.shape[0]
                
                # Grab the 30 seconds data
                chunk_size = (30 * fs)
                N_chunk = ppg.shape[0] // chunk_size   
                
                for itr in range(2):
                    start = itr * chunk_size
                    end = start + chunk_size
                    
                    # Get the chunk data
                    c_ppg = PPG(ppg[start:end], fs=fs)
                    c_ecg = ECG(ecg[start:end], fs=fs)
                    c_abp = abp[start:end]
                    
                    # Compute the SQI
                    ppg_sqi = psqi.score(c_ppg.data, signal_type="PPG", fs=fs, ma_window_size=ma_window_size, is_flat_threshold=is_flat_threshold)
                    ecg_sqi = esqi.score(c_ecg.data, signal_type="ECG", fs=fs, ma_window_size=ma_window_size, is_flat_threshold=is_flat_threshold)

                    # Try computing for the PTT
                    try:
                        ppg_peaks = c_ppg.valleys()
                        ecg_peaks = c_ecg.peaks()
                        ecg_peaks, ppg_peaks = align_peaks_ecg_ppg(ecg_peaks, ppg_peaks)
                        ptt = get_ptt(ecg_peaks, ppg_peaks, fs=fs)
                    except:
                        ptt = None                    
                    
                    # Compute SBP and DBP
                    try:
                        sbp_mean, dbp_mean, sbp_std,dbp_std = get_bp_labels(c_abp, fs=fs)
                    except:
                        sbp_mean, dbp_mean, sbp_std, dbp_std = (0,0,0,0)

                    # Save the chunk
                    output_signal = {"ppg":c_ppg.data, "ecg":c_ecg.data, "abp":c_abp}
                    
                    
                    output_signal_path = db_path.replace("raw","processed") + "{}-{}-{:0>5}_{}_{}.pkl".format(s, rec, itr, base_date, base_time)   
                    os.makedirs(db_path.replace("raw","processed"), exist_ok=True)                 
                    pkl.dump(output_signal, open(output_signal_path, "wb"))
                                        
                    row = {
                        "subject_id":s,
                        "record_id":rec,
                        "chunk_id":itr,
                        "ptt":ptt,
                        "sbp_mean":sbp_mean,
                        "sbp_std":sbp_std,
                        
                        "dbp_mean":dbp_mean,
                        "dbp_std":dbp_std,
                        
                        "rec_date": base_date,
                        "rec_time": base_time,
                        
                        "ecg_sqi": ecg_sqi,
                        "ppg_sqi": ppg_sqi,
                        "signal_path": output_signal_path,
                    }

                    
                    try:
                        # PPG Feature extraction
                        he_features, _, _ = c_ppg.features_extractor()
                        for f in add_features:
                            if type(he_features[f]) == type(np.array([])):
                                for ctr in range(len(he_features[f])):
                                    row[f+"_{}".format(ctr)] = he_features[f][ctr]
                            else:
                                row[f] = he_features[f]
                    except:
                        # Unable to extract additional features
                        for f in add_feature_cols:
                            row[f] = 0
                    
                    subject_df = subject_df.append(row, ignore_index=True)
                
        if subject_df.shape[0] > 0:
            save_path = db_path.replace("raw","anno")+"/{:0>3}.csv".format(s)
            os.makedirs(db_path.replace("raw","anno"), exist_ok=True)  
            subject_df.to_csv(save_path, index=False)


def main(args):
    # Load mimicv1's metadata (scrapped from https://physionet.org/files/mimicdb/1.0.0/mimic-index.shtml)   
    allmeta = json.load(open(args.db_path + "mimic-1.0.0-meta.json","r"))
    
    # Generate annotations as .csv
    generate_anno(args.db_path+"raw/", allmeta)


if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())
