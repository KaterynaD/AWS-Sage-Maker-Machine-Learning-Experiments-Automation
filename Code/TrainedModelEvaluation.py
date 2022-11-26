import argparse
import os
import sys
import subprocess
import pathlib
import pickle
import tarfile
import joblib
import numpy as np
import pandas as pd
import xgboost


#Evaluation metric
from sklearn.metrics import roc_auc_score
#To estimate models performance we need a custom gini function
def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)

def create_fmap(ModelName,featureset):
    fmap_filename='%s.fmap'%ModelName
    outfile = open(fmap_filename, 'w')
    for i, feat in enumerate(featureset):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()
    return fmap_filename

if __name__=='__main__':
    
    #installing XGBFir
    XGBFirFlg = False
    try:
        xgbfir_installed = subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgbfir'])
        if xgbfir_installed == 0:
            import xgbfir
            XGBFirFlg = True
            print('Successfully installed XGBfir')
        else:
            print('XGBfir was not installed')
    except:
        print('XGBfir was not installed')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--split_year', type=int)       
    parser.add_argument('--model', type=str)
    parser.add_argument('--featureset', type=str)     
    parser.add_argument('--target', type=str)
    args, _ = parser.parse_known_args()    
    print('Received arguments {}'.format(args))
    
    featureset=args.featureset.split(',')
    target_column=args.target
    #prediction will be added into the dataset in column "model_name"
    model_name=args.model
    model_path = '/opt/ml/processing/input/model/model.tar.gz'
    input_data_path = os.path.join('/opt/ml/processing/input', args.data_file)
    metrics_data_path = '/opt/ml/processing/output_metrics/metrics.csv'
    importance_data_path = '/opt/ml/processing/output_importance/importance.csv'
    interactions_data_path = '/opt/ml/processing/output_importance/interactions_%s.xlsx'%model_name
    prediction_data_path = '/opt/ml/processing/output_prediction/prediction.csv' 
    
    print('Reading model from file %s'%model_path)
    with tarfile.open(model_path) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=".")
    
    model = pickle.load(open('xgboost-model','rb'))

    print('Reading dataset from %s'%input_data_path)
    dataset = pd.read_csv(input_data_path, error_bad_lines=False, index_col=False)
    
    print('Creating dataset for prediction')
    test_dataset = pd.DataFrame()
    for f in featureset:
        print(f)
        test_dataset[f]=dataset.eval(f)
    dataset[target_column]=dataset.eval(target_column) 
    
    print('Creating DMatrix from dataset for prediction')
    X_test = xgboost.DMatrix(test_dataset.values)
    
    print('Prediction')
    predictions = model.predict(X_test)
    dataset[model_name]=predictions
    dataset[[model_name]].to_csv(prediction_data_path, header=True, index=False)
    
    print('Evaluation')
    test_roc_auc=roc_auc_score(dataset[(dataset.cal_year == args.split_year)][target_column], dataset[(dataset.cal_year == args.split_year)][model_name])
    train_roc_auc=roc_auc_score(dataset[(dataset.cal_year < args.split_year)][target_column], dataset[(dataset.cal_year < args.split_year)][model_name])
    
    test_gini=gini(dataset[(dataset.cal_year == args.split_year)][target_column],dataset[(dataset.cal_year == args.split_year)][model_name])/gini(dataset[(dataset.cal_year == args.split_year)][target_column],dataset[(dataset.cal_year == args.split_year)][target_column])
    train_gini=gini(dataset[(dataset.cal_year < args.split_year)][target_column],dataset[(dataset.cal_year < args.split_year)][model_name])/gini(dataset[(dataset.cal_year < args.split_year)][target_column],dataset[(dataset.cal_year < args.split_year)][target_column])
    
    TestingDataResults = pd.DataFrame(list(zip([model_name],[train_roc_auc],[test_roc_auc],[train_gini],[test_gini])), 
               columns =['Model','Train ROC-AUC','Test ROC-AUC','Train gini','Test gini'])
    
    TestingDataResults.to_csv(metrics_data_path, header=True, index=False)
    
    print('Feature Importance')
    
    fmap_filename=create_fmap(model_name,featureset)
    feat_imp = pd.Series(model.get_score(fmap=fmap_filename,importance_type='weight')).to_frame()
    feat_imp.columns=['Weight']
    feat_imp = feat_imp.join(pd.Series(model.get_score(fmap=fmap_filename,importance_type='gain')).to_frame())
    feat_imp.columns=['Weight','Gain']
    feat_imp = feat_imp.join(pd.Series(model.get_score(fmap=fmap_filename,importance_type='cover')).to_frame())
    feat_imp.columns=['Weight','Gain','Cover']
    feat_imp['FeatureName'] = feat_imp.index
    feat_imp['Model'] = model_name
    
    feat_imp.to_csv(importance_data_path, header=True, index=False)
    
    if XGBFirFlg:
        print('Feature Interaction')
        xgbfir.saveXgbFI(model, feature_names=featureset,  TopK = 500,  MaxTrees = 500, MaxInteractionDepth = 2, OutputXlsxFile = interactions_data_path)
