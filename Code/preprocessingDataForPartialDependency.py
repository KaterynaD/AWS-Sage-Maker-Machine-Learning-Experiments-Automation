
#The code creates a separate dataset for each feature with all possible combination of feature values and the rest of the data
#dataset for SageMaker are the same structure: no headers, the first column is a target and the rest are features


import argparse
import os
import pandas as pd
import numpy as np


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--model_complete_featureset', type=str)      
    parser.add_argument('--featureset', type=str)    
    parser.add_argument('--featuretypes', type=str)     
    args, _ = parser.parse_known_args()    
    print('Received arguments {}'.format(args))
    
    model_complete_featureset=args.model_complete_featureset.split(',')
    featureset=args.featureset.split(',')
    featuretypes=args.featuretypes.split(',')
    input_data_path = os.path.join('/opt/ml/processing/input', args.data_file)

    
    print('Reading input data from {}'.format(input_data_path))
    dataset = pd.read_csv(input_data_path, error_bad_lines=False, index_col=False)
    
     
   
    for feature,ftype in zip(featureset,featuretypes):
        print(feature,ftype)
        dataset_feature = pd.DataFrame()    
        #dataset_temp = dataset[model_complete_featureset].copy()  
        dataset_temp = pd.DataFrame()
        for f in model_complete_featureset:
            dataset_temp[f]=dataset.eval(f)
        if ftype=='Continuous':
            # continuous
            grid = sorted(np.linspace(np.percentile(dataset_temp[feature], 0.1),
                       np.percentile(dataset_temp[feature], 99.5),
                          50))
        else:
            #categorical
            grid = sorted(dataset_temp[feature].unique())        
 
        for i, val in enumerate(grid):
            dataset_temp[feature] = val
            dataset_feature=dataset_feature.append(dataset_temp)
        #save in parts if large dataset
        if ftype=='Continuous':
            parts = np.array_split(dataset_feature, 5)
            
            for i,p in enumerate(parts):
                output_data_path = os.path.join('/opt/ml/processing/output', '%s_%s.csv'%(feature,i))
                p.to_csv(output_data_path,header=False,index=False)
        else:   
            output_data_path = os.path.join('/opt/ml/processing/output', '%s.csv'%feature)
            dataset_feature.to_csv(output_data_path,header=False,index=False)
        
