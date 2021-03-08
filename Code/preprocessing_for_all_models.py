
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
    parser.add_argument('--config_file', type=str)     
    args, _ = parser.parse_known_args()    
    print('Received arguments {}'.format(args))
    
   
    target_column=args.target
    input_data_path = os.path.join('/opt/ml/processing/input', args.data_file)
    config_data_path = os.path.join('/opt/ml/processing/config', args.config_file)
    
    print('Reading input data from {}'.format(input_data_path))
    dataset = pd.read_csv(input_data_path, error_bad_lines=False, index_col=False)
    dataset_test=dataset[(dataset.cal_year == args.split_year)]
    dataset=dataset[(dataset.cal_year < args.split_year)]    
    

    print('Reading config data from {}'.format(config_data_path))
    models = pd.read_csv(config_data_path, error_bad_lines=False, index_col=False)    
    

    #iterating thru config file with models and featureset
    for index, row in models.iterrows():
        model=row['Model']
        print (index, ': Creating datasets for model %s'%model)
        featureset=row[1:51].tolist()
        featureset=[x for x in featureset if str(x) != 'nan']
        print(','.join(featureset))
        
        #creating dataset for a model according to configured dataset
        X = pd.DataFrame()
        X_test = pd.DataFrame()        
        for f in featureset:
            X[f]=dataset.eval(f)
            X_test[f]=dataset_test.eval(f)            
        y=dataset.eval(target_column)
        y_test=dataset_test.eval(target_column) 
        
        print('Testing data...')
        test_data_output_path = '/opt/ml/processing/output/testing_data/%s/'%model              
        if not os.path.exists(test_data_output_path):
            os.makedirs(test_data_output_path)       
        test_data_output_path = os.path.join(test_data_output_path,  'testing_%s.csv'%(model))     
        test_dataset=pd.DataFrame({target_column:y_test}).join(X_test)
        test_dataset.to_csv(test_data_output_path, header=False, index=False)

        
        
        #separating training and validation 
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.val_size, random_state=42)
        
        print('Train data...')        
        if not os.path.exists('/opt/ml/processing/output/training_data/%s'%model):
            os.makedirs('/opt/ml/processing/output/training_data/%s'%model)
        train_data_output_path = os.path.join('/opt/ml/processing/output/training_data/%s'%model, 'training_%s.csv'%model) 
        
        training_dataset=pd.DataFrame({target_column:y_train}).join(X_train)
        training_dataset.to_csv(train_data_output_path, header=False, index=False)

        print('Validation data...')                
        if not os.path.exists('/opt/ml/processing/output/validation_data/%s'%model):
            os.makedirs('/opt/ml/processing/output/validation_data/%s'%model)
        validation_data_output_path = os.path.join('/opt/ml/processing/output/validation_data/%s'%model, 'validation_%s.csv'%model)        
        
        validation_dataset=pd.DataFrame({target_column:y_val}).join(X_val)   
        validation_dataset.to_csv(validation_data_output_path, header=False, index=False)
