
#Training and Validation dataset for SageMaker are the same structure: no headers, the first column is a target and the rest are features


import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--split_year', type=int)   
    parser.add_argument('--num_folds', type=int)      
    parser.add_argument('--target', type=str)      
    parser.add_argument('--model', type=str)
    parser.add_argument('--featureset', type=str)    
    args, _ = parser.parse_known_args()    
    print('Received arguments {}'.format(args))
    
    featureset=args.featureset.split(',')
    target_column=args.target
    input_data_path = os.path.join('/opt/ml/processing/input', args.data_file)
    train_data_output_path = '/opt/ml/processing/output/training_data'  
    validation_data_output_path = '/opt/ml/processing/output/validation_data'
  
    
   
    
    print('Reading input data from {}'.format(input_data_path))
    dataset = pd.read_csv(input_data_path, error_bad_lines=False, index_col=False)
    
    
    dataset=dataset[(dataset.cal_year < args.split_year)][featureset + [target_column]]
    
    X = dataset[featureset]
    y = dataset[target_column]

    #StratifiedKFold
    kfold =args.num_folds 
    skf = StratifiedKFold(n_splits=kfold, random_state=42, shuffle=True)
    
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(' fold: {}  of  {} : '.format(i+1, kfold))
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index] 
        
        train_data_output_path_fold = os.path.join(train_data_output_path,  'fold_%s_training_%s.csv'%(i,args.model))    
        validation_data_output_path_fold = os.path.join(validation_data_output_path, 'fold_%s_validation_%s.csv'%(i,args.model))       
        
        training_dataset=pd.DataFrame({'hasclaim':y_train}).join(X_train)
        training_dataset.to_csv(train_data_output_path_fold, header=False, index=False)
                                                   
        validation_dataset=pd.DataFrame({'hasclaim':y_valid}).join(X_valid)   
        validation_dataset.to_csv(validation_data_output_path_fold, header=False, index=False)    
