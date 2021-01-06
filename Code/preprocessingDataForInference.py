


import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--featureset', type=str)    
    args, _ = parser.parse_known_args()    
    print('Received arguments {}'.format(args))
    
    featureset=args.featureset.split(',')
    input_data_path = os.path.join('/opt/ml/processing/input', args.data_file)
    output_data_path = os.path.join('/opt/ml/processing/output', 'data_%s.csv'%args.model)    

  
    
   
    
    print('Reading input data from {}'.format(input_data_path))
    dataset = pd.read_csv(input_data_path, error_bad_lines=False, index_col=False)
    
    
    test_dataset = pd.DataFrame()
    for f in featureset:
        test_dataset[f]=dataset.eval(f)
    
       
    test_dataset.to_csv(output_data_path, header=False, index=False)
    
 
    
