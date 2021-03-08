
#no headers


import argparse
import os
import pandas as pd
import numpy as np


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--config_file', type=str) 
    parser.add_argument('--split_to_N_parts', type=int, default=1)
    args, _ = parser.parse_known_args()    
    print('Received arguments {}'.format(args))
    
    split_to_N_parts=args.split_to_N_parts
    input_data_path = os.path.join('/opt/ml/processing/input', args.data_file)
    config_data_path = os.path.join('/opt/ml/processing/config', args.config_file)


    
    print('Reading input data from {}'.format(input_data_path))
    dataset = pd.read_csv(input_data_path, error_bad_lines=False, index_col=False)
    

    print('Reading config data from {}'.format(config_data_path))
    models = pd.read_csv(config_data_path, error_bad_lines=False, index_col=False)      

  
    #iterating thru config file with models and featureset
    for index, row in models.iterrows():
        model=row['Model']
        print (index, ': Creating featuresets for model %s'%model)
        featureset=row[1:51].tolist()
        featureset=[x for x in featureset if str(x) != 'nan']    

        X = pd.DataFrame()
        for f in featureset:
            X[f]=dataset.eval(f)        
        
        if not os.path.exists('/opt/ml/processing/output/%s'%model):
            os.makedirs('/opt/ml/processing/output/%s'%model)
        output_data_path = os.path.join('/opt/ml/processing/output/%s'%model, 'data.csv')
        if split_to_N_parts>1:
            parts = np.array_split(X, split_to_N_parts)
            for i,p in enumerate(parts):
                output_data_path = os.path.join('/opt/ml/processing/output/%s'%model, 'data_%s.csv'%i)
                p.to_csv(output_data_path,header=False,index=False)
        else:           
            X.to_csv(output_data_path, header=False, index=False)
        

    

    
    
 
    
