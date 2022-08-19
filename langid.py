import pdb 
from optparse import OptionParser
from feature_extraction import *
from sklearn.metrics import classification_report
from rf import * 
import logging
from typing import List, Dict 

logging.basicConfig(level = logging.INFO)

class LangID:

    def __init__(self, feature_type, model, train, test):
        self.feature_type = feature_type
        self.model = model 
        self.train = train 
        self.test = test 
        self.logger = logging

    def get_features(self):
        self.logger.info("=== Loading features ===")
        self.train_df = pd.read_pickle("./" + self.feature_type + "/" + self.train)
        self.test_df = pd.read_pickle("./" + self.feature_type + "/" + self.train)

    
    def train_model(self):
        self.logger.info("=== training model ===")
        if self.model == "rf":
            self.clf = train_model(self.train_df, self.test_df)
        else:
            # FFN
            pass 
    
    def eval(self):
        self.logger.info("=== Running evaluation ===")
        y_test = self.test_df.loc[:,self.test_df.columns!='label']
        y_pred_test = self.clf.predict(y_test)
        return classification_report(y_test, y_pred_test)


if __name__ == "__main__":

    parser = OptionParser(__doc__)

    parser.add_option("--feature_type",
                    dest="feature_type",
                    default="mfcc",
                    help="--feature_type=[mfcc|plp] to select feature extraction method; default is mfcc.")
    
    parser.add_option("--model",
                    dest="model",
                    default="rf",
                    help="--model=[rf|ann] to select model to train; default is svm.")
    
    parser.add_option("--train",
                    dest="train",
                    default="train_df.p",
                    help="--train pickle filename of training data.")
    
    parser.add_option("--test",
                    dest="test",
                    default="dev_df.p",
                    help="--test pickle filename of testing data.")                
    
    options, args = parser.parse_args()

    pipeline = LangID(options.feature_type, options.model, options.train, options.test)
    pipeline.get_features()
    pipeline.train_model()
    report = pipeline.eval()
    