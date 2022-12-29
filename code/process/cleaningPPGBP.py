
import pandas as pd
import numpy as np

import os
import argparse
from omegaconf import OmegaConf

from joblib import Parallel, delayed

from core.lib.preprocessing import normalize_data, mean_filter_normalize, identify_out_pk_vly, rm_baseline_wander

from core.lib.features_extraction import extract_cycle_check, extract_feat_cycle, extract_feat_original

def _print_step(s,cl_log):
    print("--- {} ---".format(s))
    cl_log.write("--- {} ---\n".format(s))

def _compute_quality_idx(df):
    df['ppg_p2p'] = df['ppg_pks'].map(np.diff).map(np.median)
    df['ppg_v2v'] = df['ppg_vlys'].map(np.diff).map(np.median)
    
    df['ppg_pks_bpm'] = args.fs/df['ppg_p2p'] * 60
    df['ppg_vlys_bpm'] = args.fs/df['ppg_v2v'] * 60
    
    return df

def _print_n_samples(df, cl_log, sent = "data size: "):
    rec = df.trial.map(lambda s: s[:s.rfind('_')])
    stats = [df.patient.unique().shape[0], rec.unique().shape[0], df.shape[0]]
    print(sent+" {}, {}, {}".format(*stats))
    cl_log.write(sent+" {}, {}, {} \n".format(*stats))

def _filter_ppg(df, args):
    """ Compute and save filter signal in dataframe.
    """
    if args.ppg_filter.enable:
        df['fsignal']=Parallel( n_jobs=args.parallel.n_jobs, 
                                verbose=args.parallel.verbose)(
                                delayed(mean_filter_normalize)( sig, fs=args.fs,
                                                                lowcut=args.ppg_filter.lowcut,
                                                                highcut=args.ppg_filter.highcut,
                                                                order=1) for sig in df.signal)
    else:
        df['fsignal']=Parallel( n_jobs=args.parallel.n_jobs, 
                                verbose=args.parallel.verbose)(
                                delayed(normalize_data)(sig) for sig in df.signal)
    return df

def _extract_c(sig, fs, pk_th=0.6, remove_start_end=False):
    """ Wrapper for extract_cycle_check function.
    """
    try:
        cs, pks_norm, flag1, flag2, pks, vlys = extract_cycle_check(sig, fs, pk_th, remove_start_end)
    except:
        cs, pks_norm, flag1, flag2, pks, vlys = [], [], True, True, [], []
    return cs, pks_norm, flag1, flag2, pks, vlys

#---------- Step Functions ----------#

def _abnormal_BP(df, args, cl_log):

    up_sbp = args.bp_filter.up_sbp
    lo_dbp = args.bp_filter.lo_dbp
    lo_diff = args.bp_filter.lo_diff

    #Limit BP labels
    df['bp_dif'] = df['SP'] - df['DP']
    BP_max = (df.SP <= up_sbp)
    BP_min = (df.DP >= lo_dbp)
    BP_dif = (df.bp_dif >= lo_diff)

    cl_log.write(" - removed by SP range: {} \n".format((~BP_max).sum()))
    cl_log.write(" - removed by DP range: {} \n".format((~BP_min).sum()))
    cl_log.write(" - removed by BP-diff range: {} \n".format((~BP_dif).sum()))

    df = df[BP_max & BP_min & BP_dif].reset_index(drop=True)

    return df

def _extract_ppg_cycles(df, args, cl_log):
    """ Compute and save the ppg cycles in dataframe.
        This function updates filtered signal. 
    """
    df = _filter_ppg(df, args)

    c_ppg = Parallel(n_jobs=args.parallel.n_jobs, verbose=args.parallel.verbose)(delayed(_extract_c)(sig, fs=args.fs, remove_start_end=args.remove_start_end) for sig in df.fsignal)

    for i, label in enumerate(['cs','pks_norm','ppg_f1','ppg_f2','ppg_pks','ppg_vlys']):
        df[label] = [val[i] for val in c_ppg]

    not_computed = ((df['ppg_pks'].map(len) < 1) | (df['ppg_vlys'].map(len) < 2) | df['ppg_f2'])
    cl_log.write(" - removed by not computed: {} \n".format((not_computed).sum()))
    df = df[~not_computed].reset_index(drop=True)

    return df

def _limitation_bpm(df, args, cl_log):
    lo_th_p2p = args.cycle_len.lo_bpm
    up_th_p2p = args.cycle_len.up_bpm
    
    pre = 'ppg'

    removed_abp_p2p = (df[pre+'_pks_bpm'] < lo_th_p2p) | (df[pre+'_pks_bpm'] > up_th_p2p) # | (df[pre+'_pks_bpm'].isna())
    removed_p2p = removed_abp_p2p

    cl_log.write("- removed by {} p2p distance (BPM): {} \n".format(pre,removed_abp_p2p.sum()))
    
    removed_abp_v2v = (df[pre+'_vlys_bpm'] < lo_th_p2p) | (df[pre+'_vlys_bpm'] > up_th_p2p) | (df[pre+'_vlys_bpm'].isna())
    removed_v2v = removed_abp_v2v

    cl_log.write(" - removed by {} v2v distance (BPM): {} \n".format(pre,removed_abp_v2v.sum()))
    
    df = df[~removed_p2p & ~removed_v2v].reset_index(drop=True)

    return df

def _compute_features_df(df, args):
    """ Compute features and output dataframe with features, and dataframe with cleaned raw signals.
    """

    assert df.ppg_f2.sum() == 0
    ## --------- Generate second set of features ---------
    heads_feats = Parallel(n_jobs=args.parallel.n_jobs, verbose=args.parallel.verbose)(delayed(extract_feat_cycle)(row.cs, row.pks_norm, fs=args.fs) for i,row in df.iterrows())

    list_2 = []
    for i,(h,f) in enumerate(heads_feats):
        if len(h)==0 or len(f) == 0:
            list_2.append(i)

    list_series = [pd.Series(f, index=h, dtype='float64') for h,f in heads_feats]
    res_second = pd.DataFrame(list_series)

    ## --------- Generate first set of features ---------
    heads_feats_first = Parallel(n_jobs=args.parallel.n_jobs, verbose=args.parallel.verbose)(delayed(extract_feat_original)(fsig, fs=args.fs, filtered = args.ppg_filter.enable, remove_start_end=args.remove_start_end) for fsig in df['fsignal'])

    list_2_1 = []
    for i,(h,f) in enumerate(heads_feats_first):
        if len(h)==0 or len(f) == 0:
            list_2_1.append(i)
            
    list_series = [pd.Series(f, index=h, dtype='float64') for h,f in heads_feats_first]
    res = pd.DataFrame(list_series)

    template = pd.read_csv(args.path.feats_template)['columns'].values

    cols = []
    for c in template:
        if c in res.columns:
            cols.append(c)

    res_first=res[cols].copy()

    res_feats = pd.concat([res_first,res_second], axis=1)
    data_feats = df[['patient','trial','SP','DP']].copy()
    data_feats = pd.concat([data_feats, res_feats],axis=1)

    data_feats = data_feats.drop(index=list_2+list_2_1).reset_index(drop=True)
    df = df.drop(index=list_2+list_2_1).reset_index(drop=True)

    keep_mask = ~data_feats.isna()['bd']

    data_feats = data_feats[keep_mask].reset_index(drop=True)
    df = df[keep_mask].reset_index(drop=True)

    # Remove longer signals
    len_=df.signal.map(len)
    idx_ = len_[len_!=len_.iloc[0]].index
    for i in idx_:
        df.at[i,'signal'] = df.loc[i,'signal'][:len_.iloc[0]]

    return data_feats, df


def main(args):

    os.makedirs(os.path.dirname(args.path.log), exist_ok=True)
    cl_log = open(args.path.log, "w") ## Read logging file

    ## Create the dirs for the ouput data if do not exist
    os.makedirs(os.path.dirname(args.path.save_name_feats), exist_ok=True)
    os.makedirs(os.path.dirname(args.path.save_name), exist_ok=True)

    df = pd.read_pickle(args.path.data) ## Read dataset
    original_columns = df.columns ## Save original columns

    _print_n_samples(df, cl_log, sent = "Original data: ")


    #---------- Filter BP values ----------#
    _print_step("Abnormal BP values",cl_log)
    df=_abnormal_BP(df, args, cl_log)
    _print_n_samples(df, cl_log)


    #---------- Peaks and valleys computation ----------#
    _print_step("Peaks and valleys computation",cl_log)
    df = _extract_ppg_cycles(df, args, cl_log)
    _print_n_samples(df, cl_log)
    

    #---------- Baseline Wander (BW) removal ----------#
    _print_step("Baseline Wander (BW) removal",cl_log)
    ### remove baseline wander
    rm_signal = Parallel(n_jobs=args.parallel.n_jobs, verbose=args.parallel.verbose)(delayed(rm_baseline_wander)(row.signal, vlys=row.ppg_vlys) for i, row in df.iterrows())
    df['signal']=[s[0] for s in rm_signal]
    _print_n_samples(df, cl_log)


    #---------- Refinement ----------#

    _print_step("Refinement",cl_log)

    _print_step("BPM limitation (PPG)",cl_log)
    df = _extract_ppg_cycles(df, args, cl_log)
    df = _compute_quality_idx(df)
    df = _limitation_bpm(df, args, cl_log)
    _print_n_samples(df, cl_log)


    #---------- Feature generation ----------#
    _print_step("Segment removal by feature generation",cl_log)
    df = df.drop(index=df[df.trial=='203_3'].index[0]).reset_index(drop=True) ## Missed bad quality signal

    data_feats, data_raw = _compute_features_df(df, args)
    data_raw=data_raw[original_columns]

    _print_n_samples(data_feats, cl_log)


    #---------- SQI removal ----------#
    _print_step("Segment removal by skewness SQI",cl_log)

    keep_mask = data_feats['SQI_skew'] > args.lo_sqi
    data_feats = data_feats[keep_mask].reset_index(drop=True)
    data_raw = data_raw[keep_mask].reset_index(drop=True)

    _print_n_samples(data_feats, cl_log)


    #---------- Save data ----------#
    cl_log.close()
    data_feats.to_pickle(args.path.save_name_feats)
    data_raw.to_pickle(args.path.save_name)

if __name__=="__main__":

    #---------- Read config file ----------#
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Path for the config file", required=True)
    args_m = parser.parse_args()

    ## Read config file
    if os.path.exists(args_m.config_file) == False:         
        raise RuntimeError("config_file {} does not exist".format(args_m.config_file))
    args = OmegaConf.load(args_m.config_file)
    
    main(args)
