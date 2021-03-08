
#The code creates a separate dataset for each feature with all possible combination of feature values and the rest of the data
#dataset for SageMaker are the same structure: no headers, the first column is a target and the rest are features


import argparse
import os
import pandas as pd
import numpy as np


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--config_file', type=str)   
    parser.add_argument('--featureset', type=str)    
    parser.add_argument('--featuretypes', type=str) 
    parser.add_argument('--split_to_N_parts', type=int, default=1)
    args, _ = parser.parse_known_args()    
    print('Received arguments {}'.format(args))
    

    featureset=args.featureset.split(',')
    featuretypes=args.featuretypes.split(',')
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
        model_complete_featureset=row[1:51].tolist()
        model_complete_featureset=[x for x in model_complete_featureset if str(x) != 'nan']
        #specific folder for each model data
        if not os.path.exists('/opt/ml/processing/output/%s'%model):
            os.makedirs('/opt/ml/processing/output/%s'%model)
        #iterating thru features for pd
        for feature,ftype in zip(featureset,featuretypes):
            print(feature,ftype)
            dataset_feature = pd.DataFrame()    
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
                parts = np.array_split(dataset_feature, split_to_N_parts)
            
                for i,p in enumerate(parts):
                    output_data_path = os.path.join('/opt/ml/processing/output/%s'%model, '%s_%s.csv'%(feature,i))
                    p.to_csv(output_data_path,header=False,index=False)
            else:   
                output_data_path = os.path.join('/opt/ml/processing/output/%s'%model, '%s.csv'%feature)
                dataset_feature.to_csv(output_data_path,header=False,index=False)
        
