
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
    parser.add_argument('--model', type=str)
    parser.add_argument('--featureset', type=str)  
    parser.add_argument('--training_dataset_sizes', type=str) 
    args, _ = parser.parse_known_args()   
    
    print('Received arguments {}'.format(args))
    
    featureset=args.featureset.split(',')
    target_column='hasclaim'
    training_dataset_sizes=args.training_dataset_sizes.split(',')
    input_data_path = os.path.join('/opt/ml/processing/input', args.data_file)
    train_data_output_path = '/opt/ml/processing/output/training_data'  
    validation_data_output_path = '/opt/ml/processing/output/validation_data'

    
   
    
    print('Reading input data from {}'.format(input_data_path))
    dataset = pd.read_csv(input_data_path, error_bad_lines=False, index_col=False)
    
    
    dataset=dataset[(dataset.cal_year < args.split_year)][featureset + [target_column]]
    
    X = dataset[featureset]
    y = dataset[target_column]
    
    for s in training_dataset_sizes:
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1 - float(s), random_state=42)
       
        
        train_data_output_path_batch = os.path.join(train_data_output_path,  'batch_%s_training_%s.csv'%(s,args.model))    
        validation_data_output_path_batch = os.path.join(validation_data_output_path, 'batch_%s_validation_%s.csv'%(s,args.model))
     
        
        training_dataset=pd.DataFrame({'hasclaim':y_train}).join(X_train)
        training_dataset.to_csv(train_data_output_path_batch, header=False, index=False)
                                                   
        validation_dataset=pd.DataFrame({'hasclaim':y_val}).join(X_val)   
        validation_dataset.to_csv(validation_data_output_path_batch, header=False, index=False)    
