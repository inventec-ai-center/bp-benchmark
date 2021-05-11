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
from core.signal_processing.parser import load_mimicv1_csv
from core.signal_processing.clean import filtering, SQI
from core.signal_processing.extract import PPG, ECG, get_ptt, align_peaks_ecg_ppg
from core.signal_processing.utils import global_norm, global_denorm, get_bp_labels, waveform_norm

import argparse

DEFAULT_CONF_PATH = "core/config/default_config.json"


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", type=str, default=DEFAULT_CONF_PATH,
                        help="path to the config file")
    parser.add_argument("--db_path", default="../../datasets/origin/") 
    parser.add_argument("--output_path", default="../../processed_data/processed_origin/")
    parser.add_argument('--n_sample', type=int, default=10)
    parser.add_argument('--n_std', type=float, default=1)
    parser.add_argument('--tolerance', type=float, default=0.8)


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

def generate_anno(db_path, processed_path, allmeta, fs=125, ma_window_size=100, is_flat_threshold=0.5):
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
                    
                    
                    # output_signal_path = db_path.replace("raw","processed") + "{}-{}-{:0>5}_{}_{}.pkl".format(s, rec, itr, base_date, base_time)   
                    output_signal_path = f"{processed_path}/processed/" + "{}-{}-{:0>5}_{}_{}.pkl".format(s, rec, itr, base_date, base_time) 
                    # os.makedirs(db_path.replace("raw","processed"), exist_ok=True) 
                    os.makedirs(f"{processed_path}/processed/", exist_ok=True)                 
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
            # save_path = db_path.replace("raw","anno")+"/{:0>3}.csv".format(s)
            # os.makedirs(db_path.replace("raw","anno"), exist_ok=True)  
            save_path = f"{processed_path}/anno/" + "{:0>3}.csv".format(s)
            os.makedirs(f"{processed_path}/anno/", exist_ok=True)  
            subject_df.to_csv(save_path, index=False)

def extract_signal_features(signal_path, n_sample, n_std, tolerance):
    '''
    signal_path is a string pointing to a pickle-dictionary object containing ecg, ppg, and abp signals
    '''
    data = pd.read_pickle(signal_path)
    output = {}
    
    output["ecg"] = waveform_norm(data["ecg"])
    output["ppg"] = waveform_norm(data["ppg"])
    output["abp"] = data["abp"]
    
    ppg = PPG(data["ppg"], fs=125)
    ecg = ECG(data["ecg"], fs=125)
    
    # Derivative features
    output["apg"] = waveform_norm(ppg.apg())
    output["vpg"] = waveform_norm(ppg.vpg())
    
    # Clean signals
    cycles = ppg.clean_cycles(n_sample=n_sample, n_std=n_std, tolerance=tolerance)
    output["cycles"] = cycles
    
    # For the cycle stats, we can use larger n_std and set the tolerance to 1 since we are going to use PCA
    # Prominent noises are treated as features, otherwise they are ignore by PCA
    cycle_stats = ppg.cycles_stat_representation(n_pca_components=5, n_std=2, tolerance=1.0)
    output["cycle_stats"] = cycle_stats
    
    return output


def main(args):
    # Load mimicv1's metadata (scrapped from https://physionet.org/files/mimicdb/1.0.0/mimic-index.shtml)   
    allmeta = json.load(open(args.db_path + "mimic-1.0.0-meta.json","r"))
    json.dump(allmeta, open(args.output_path + "mimic-1.0.0-meta.json", "w"))
    # Generate annotations as .csv
    generate_anno(args.db_path+"raw/", args.output_path, allmeta)


    df = load_mimicv1_csv(args.output_path)
    signal_paths = df.signal_path.values    
    
    for path in tqdm(signal_paths):
        output = extract_signal_features(path, args.n_sample, args.n_std, args.tolerance) 
        os.makedirs(f"{args.output_path}/extracted/", exist_ok=True)        
        pkl.dump(output, open(f"{args.output_path}/extracted/" + os.path.basename(path), "wb"))




if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())
