
#Training and Validation dataset for SageMaker are the same structure: no headers, the first column is a target and the rest are features


import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--split_year', type=int)     
    parser.add_argument('--val_size', type=float)     
    parser.add_argument('--target', type=str)      
    parser.add_argument('--model', type=str)  
    parser.add_argument('--featureset', type=str)    
    args, _ = parser.parse_known_args()    
    print('Received arguments {}'.format(args))
    
    featureset=args.featureset.split(',')
    target_column=args.target
    input_data_path = os.path.join('/opt/ml/processing/input', args.data_file)
    train_data_output_path = os.path.join('/opt/ml/processing/output/training_data', 'training_%s.csv'%args.model)    
    validation_data_output_path = os.path.join('/opt/ml/processing/output/validation_data', 'validation_%s.csv'%args.model)
  
    
   
    
    print('Reading input data from {}'.format(input_data_path))
    dataset = pd.read_csv(input_data_path, error_bad_lines=False, index_col=False)
    
    
    dataset=dataset[(dataset.cal_year < args.split_year)]
    
    X = pd.DataFrame()
    for f in featureset:
        X[f]=dataset.eval(f)
    
    y=dataset.eval(target_column)
    


    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.val_size, random_state=42)
    
    training_dataset=pd.DataFrame({'hasclaim':y_train}).join(X_train)
    training_dataset.to_csv(train_data_output_path, header=False, index=False)


    
    validation_dataset=pd.DataFrame({'hasclaim':y_val}).join(X_val)   
    validation_dataset.to_csv(validation_data_output_path, header=False, index=False)
    
 
    
