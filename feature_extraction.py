import librosa
import pandas as pd 
import numpy as np 
from typing import List
from sklearn.model_selection import train_test_split
from optparse import OptionParser
import os 
import tqdm
import pdb

def load_data(df: pd.DataFrame):
    """ Load wav files from given folder using librosa """
    loaded = []
    for i, r in df.iterrows():
        y, sr = librosa.load(r["filepaths"])
        loaded.append((y, sr, r["label"]))
    return loaded 

def compute_mfcc(time_series: List):
    mfccs = []
    labels = []
    for t in time_series:
        y = t[0]
        sr = t[1]
        label = t[2]
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        mfccs.append(mfcc)
        labels.append(label)
    return mfccs, labels

def compute_delta():
    pass 

def compute_plp():
    pass 
    
def create_df(mfccs: List, labels: List):
    df = pd.DataFrame(np.transpose(mfccs[0]))
    df["label"] = [labels[0]] * len(df)
   
    for i in range(1, len(mfccs)):
        df_temp = pd.DataFrame(np.transpose(mfccs[i]))
        df_temp["label"] = [labels[i]] * len(df_temp)
        df = pd.concat([df, df_temp])
    return df 

def split_data(df: pd.DataFrame):
    X_train, X_t = train_test_split(df, test_size=0.3)
    X_dev, X_test = train_test_split(X_t, test_size=0.33)

    return X_train, X_dev, X_test
# def split_data(df: pd.DataFrame):
#     X_train, X_t, y_train, y_t = train_test_split(df.loc[:,df.columns!='label'], df['label'], test_size=0.3)
#     X_dev, X_test, y_dev, y_test = train_test_split(X_t, y_t, test_size=0.33)

#     train = pd.concat([X_train, y_train], axis=1)
#     dev = pd.concat([X_dev, y_dev], axis=1)
#     test = pd.concat([X_test, y_test], axis=1)

#     return train, dev, test


def build_datasets(train, dev, test):
    train_df = pd.concat(train)
    dev_df = pd.concat(dev)
    test_df = pd.concat(test)

    return train_df, dev_df, test_df

if __name__ == "__main__":
    parser = OptionParser(__doc__)

    parser.add_option("--feature_type",
                    dest="feature_type",
                    default="mfcc",
                    help="--feature_type=[mfcc|plp] to select feature extraction method; default is mfcc.")
    options, args = parser.parse_args()
    os.makedirs(options.feature_type, exist_ok=True)
    data_list = "/home2/debnaths/streamlined/data/"
    all_train = []
    all_dev = []
    all_test = []

    for folder in os.listdir(data_list):
        print(folder)
        filepaths = []
        lang = []
        for f in os.listdir(data_list+folder):
            filepaths.append(data_list+folder+"/"+f)
            lang.append(folder)
        
        df = pd.DataFrame()
        df["filepaths"] = filepaths
        df["label"] = lang 

        train, dev, test = split_data(df)
        all_train.append(train)
        all_dev.append(dev)
        all_test.append(test)
    
    train_files, dev_files, test_files = build_datasets(all_train, all_dev, all_test)
    
    data_loaded_train = load_data(train_files)
    data_loaded_dev = load_data(dev_files)
    data_loaded_test = load_data(test_files)

    if options.feature_type == "mfcc":
        feats_train, labels_train = compute_mfcc(data_loaded_train)
        feats_dev, labels_dev = compute_mfcc(data_loaded_dev)
        feats_test, labels_test = compute_mfcc(data_loaded_test)
    else:
        feats = compute_plp(data_loaded)
    
    train_df = create_df(feats_train, labels_train)
    train_df.to_pickle(options.feature_type + "/train_df.p")
    dev_df = create_df(feats_dev, labels_dev)
    dev_df.to_pickle(options.feature_type + "/dev_df.p")
    test_df = create_df(feats_test, labels_test)
    test_df.to_pickle(options.feature_type + "/test_df.p")
    pdb.set_trace()