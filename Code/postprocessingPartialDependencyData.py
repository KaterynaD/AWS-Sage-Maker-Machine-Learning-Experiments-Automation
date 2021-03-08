
#The code joins InputData files for each feature and inference from each model and then average by each feature value


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


    output_data_path = os.path.join('/opt/ml/processing/output', 'data.csv', )
    
    print('Reading input data from {}'.format(input_data_path))
    dataset = pd.read_csv(input_data_path, error_bad_lines=False, index_col=False)
    

    print('Reading config data from {}'.format(config_data_path))
    models = pd.read_csv(config_data_path, error_bad_lines=False, index_col=False)   

    #final dataset - average pd by each feature value
    all_fm_pd = pd.DataFrame()

        
    for f,ftype in zip(featureset,featuretypes):
        print('Processing %s'%f)
        #iterating thru config file with models and featureset
        df_all_models=pd.DataFrame()
        for index, row in models.iterrows():
            model=row['Model']
            print (index, ': Creating featuresets for model %s'%model)
            model_complete_featureset=row[1:51].tolist()
            model_complete_featureset=[x for x in model_complete_featureset if str(x) != 'nan']
        
            if ftype=='Continuous':
                feature_InputData_dataset=pd.DataFrame()
                #Continious data can be splitted to split_to_N_parts files
                for j in range(0,split_to_N_parts):
                    feature_InputData_path=os.path.join('/opt/ml/processing/input/InputData/%s'%model, '%s_%s.csv'%(f,j))
                    feature_InputData_dataset_j = pd.read_csv(feature_InputData_path, names=model_complete_featureset, error_bad_lines=False, index_col=False)
                    feature_InputData_dataset=feature_InputData_dataset.append(feature_InputData_dataset_j)          
            else:
                feature_InputData_path=os.path.join('/opt/ml/processing/input/InputData/%s'%model, '%s.csv'%f)
                feature_InputData_dataset = pd.read_csv(feature_InputData_path, names=model_complete_featureset, error_bad_lines=False, index_col=False)
            
            fm_pd = pd.DataFrame()           
            print('Reading predicted data from model %s'%model)
            if ftype=='Continuous':
                pdf_dataset=pd.DataFrame()
                #Continious data can be splitted to split_to_N_parts files
                for j in range(0,split_to_N_parts):
                    model_predicted_data_path =os.path.join('/opt/ml/processing/input/PartialDependency/%s'%model.replace('_','-'),'%s_%s.csv.out'%(f,j))
                    pdf_dataset_i = pd.read_csv(model_predicted_data_path, names=['pd'], error_bad_lines=False, index_col=False)
                    pdf_dataset=pdf_dataset.append(pdf_dataset_i)
            else:
                model_predicted_data_path =os.path.join('/opt/ml/processing/input/PartialDependency/%s'%model.replace('_','-'),'%s.csv.out'%f)                
                pdf_dataset = pd.read_csv(model_predicted_data_path, names=['pd'], error_bad_lines=False, index_col=False)
            #model feature partial dependency columns name
            pd_column_name='%s_pd'%model
            feature_InputData_dataset[pd_column_name]= pdf_dataset['pd'].values
            
            #average
            fm_s = feature_InputData_dataset.groupby(f)[pd_column_name].mean()
            fm_pd_model=pd.DataFrame({'value':fm_s.index, pd_column_name:fm_s.values})           
            fm_pd_model['feature']=f
            fm_pd_model = fm_pd_model[['feature','value',pd_column_name]]

            fm_pd=pd.concat([fm_pd,fm_pd_model],axis=1)          
            #add text value for categorical encd columns
            #assuming there is encoded (_encd ended) and original values in the dataset
            fm_pd['value2']=fm_pd['value'].astype(str)
            if '_encd' in f and f.replace('_encd','') in dataset.columns:
                #unique combindation of codes and original values from the main dataset into list and then dictionary
                dataset['dummy']= dataset[f.replace('_encd','')] +'-'+ dataset[f].astype(str)
                unique_comb_l=dataset['dummy'].unique().tolist()
                unique_comb_value=[i.split('-', 1)[0] for i in unique_comb_l]
                unique_comb_key=[i.split('-', 1)[1] for i in unique_comb_l]
                unique_comb_dict = dict(zip(unique_comb_key, unique_comb_value))
                #replace value2 in the feature values and partial dependencies
                fm_pd['value2'].replace(unique_comb_dict, inplace=True)
            if len(df_all_models)==0:
                df_all_models=fm_pd
            else:
                df_all_models = pd.merge(df_all_models,fm_pd, on=['feature','value','value2'], how='outer')    

        all_fm_pd=all_fm_pd.append(df_all_models)
    #saving final output    
    all_fm_pd.to_csv(output_data_path,header=True ,index=False)    
