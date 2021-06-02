
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
    
           
    #StratifiedKFold
    kfold =args.num_folds 
    skf = StratifiedKFold(n_splits=kfold, random_state=42, shuffle=True)
    
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

        #Testing data starts from y_test because they are read in XGBoost processing script to DMatrix amd first column is separated anyway
        #Without the column the script can not predict
        print('Testing data...')
        test_data_output_path = '/opt/ml/processing/output/testing_data/%s/'%model              
        if not os.path.exists(test_data_output_path):
            os.makedirs(test_data_output_path)       
        test_data_output_path = os.path.join(test_data_output_path,  'testing_%s.csv'%(model))  
        test_dataset=pd.DataFrame({target_column:y_test}).join(X_test)
        test_dataset.to_csv(test_data_output_path, header=False, index=False)
        
        #The rest of the data will be used in cv-fold as a whole and seprated to training/validation insode cv        
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            print(' fold: {}  of  {} : '.format(i+1, kfold))
            X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[test_index] 

            train_data_output_path = '/opt/ml/processing/output/training_data/%s/'%model              
            if not os.path.exists(train_data_output_path):
                os.makedirs(train_data_output_path)
            train_data_output_path_fold = os.path.join(train_data_output_path,  'fold_%s_training_%s.csv'%(model,i))
            
            validation_data_output_path = '/opt/ml/processing/output/validation_data/%s/'%model  
            if not os.path.exists(validation_data_output_path):
                os.makedirs(validation_data_output_path)            
            validation_data_output_path_fold = os.path.join(validation_data_output_path, 'fold_%s_validation_%s.csv'%(model,i))       
        
            training_dataset=pd.DataFrame({target_column:y_train}).join(X_train)
            training_dataset.to_csv(train_data_output_path_fold, header=False, index=False)
                                                   
            validation_dataset=pd.DataFrame({target_column:y_valid}).join(X_valid)   
            validation_dataset.to_csv(validation_data_output_path_fold, header=False, index=False)    
