import argparse
import os
import sys
import subprocess
import pathlib
import pickle
import tarfile
import joblib
import numpy as np
import pandas as pd
import xgboost


#Evaluation metric
from sklearn.metrics import roc_auc_score
#To estimate models performance we need a custom gini function
def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)

def create_fmap(ModelName,featureset):
    fmap_filename='%s.fmap'%ModelName
    outfile = open(fmap_filename, 'w')
    for i, feat in enumerate(featureset):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()
    return fmap_filename

if __name__=='__main__':
    
    #installing XGBFir
    XGBFirFlg = False
    try:
        xgbfir_installed = subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgbfir'])
        if xgbfir_installed == 0:
            import xgbfir
            XGBFirFlg = True
            print('Successfully installed XGBfir')
        else:
            print('XGBfir was not installed')
    except:
        print('XGBfir was not installed')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--split_year', type=int)       
    parser.add_argument('--model', type=str)
    parser.add_argument('--featureset', type=str)     
    parser.add_argument('--target', type=str)
    args, _ = parser.parse_known_args()    
    print('Received arguments {}'.format(args))
    
    featureset=args.featureset.split(',')
    target_column=args.target
    #prediction will be added into the dataset in column "model_name-fold"
    model_name=args.model
    models_path = '/opt/ml/processing/input/model/'
    models_file='output.tar.gz'
    input_data_path = os.path.join('/opt/ml/processing/input', args.data_file)
    auc_metrics_data_path = '/opt/ml/processing/output_metrics/auc_metrics.csv'
    gini_metrics_data_path = '/opt/ml/processing/output_metrics/gini_metrics.csv'    
    importance_data_path = '/opt/ml/processing/output_importance/importance.csv'    
    prediction_data_path = '/opt/ml/processing/output_prediction/prediction.csv' 
    
    print('Extracting models from file %s'%os.path.join(models_path, models_file))
    with tarfile.open(os.path.join(models_path, models_file)) as tar:
        tar.extractall(models_path)
    
    

    print('Reading dataset from %s'%input_data_path)
    dataset = pd.read_csv(input_data_path, error_bad_lines=False, index_col=False)
    
    print('Creating dataset for prediction')
    test_dataset = pd.DataFrame()
    for f in featureset:
        print(f)
        test_dataset[f]=dataset.eval(f)
    dataset[target_column]=dataset.eval(target_column) 
    
    print('Creating DMatrix from dataset for prediction')
    X_test = xgboost.DMatrix(test_dataset.values)
    
    #loop thru available models in output.tar.gz which are xgboost-model-fold
    #assuming maximum models available is 10 from 0 to 9 in the last position of the file name
    Prediction_df = pd.DataFrame()
    Scores_df = pd.DataFrame()
    FI_df = pd.DataFrame() 
    for filename in os.listdir(models_path):
        if filename.startswith('xgboost-model-fold'):
            print('Processing model %s...'%filename)
            ind=int(filename[-1])
            print('1. Extracting model')
            model = pickle.load(open(os.path.join(models_path, filename),'rb'))
            
            print('2. Prediction')
            predictions = model.predict(X_test)
            prediction_column_name='%s-%s'%(model_name,ind)
            dataset[prediction_column_name]=predictions
            Prediction_df=pd.concat([Prediction_df,dataset['%s-%s'%(args.model,ind)]],axis=1)
            
    
            print('3. Evaluation')
            test_roc_auc=roc_auc_score(dataset[(dataset.cal_year == args.split_year)][target_column], dataset[(dataset.cal_year == args.split_year)][prediction_column_name])
            train_roc_auc=roc_auc_score(dataset[(dataset.cal_year < args.split_year)][target_column], dataset[(dataset.cal_year < args.split_year)][prediction_column_name])
    
            test_gini=gini(dataset[(dataset.cal_year == args.split_year)][target_column],dataset[(dataset.cal_year == args.split_year)][prediction_column_name])/gini(dataset[(dataset.cal_year == args.split_year)][target_column],dataset[(dataset.cal_year == args.split_year)][target_column])
            train_gini=gini(dataset[(dataset.cal_year < args.split_year)][target_column],dataset[(dataset.cal_year < args.split_year)][prediction_column_name])/gini(dataset[(dataset.cal_year < args.split_year)][target_column],dataset[(dataset.cal_year < args.split_year)][target_column])
    
            TestingDataResults = pd.DataFrame(list(zip([model_name],[ind],[train_roc_auc],[test_roc_auc],[train_gini],[test_gini])), 
               columns =['Model','fold','Train ROC-AUC','Test ROC-AUC','Train gini','Test gini'])
      
            Scores_df=pd.concat([Scores_df,TestingDataResults])
            
    
            print('4. Feature Importance')
    
            fmap_filename=create_fmap(model_name,featureset)
            feat_imp = pd.Series(model.get_score(fmap=fmap_filename,importance_type='weight')).to_frame()
            feat_imp.columns=['Weight']
            feat_imp = feat_imp.join(pd.Series(model.get_score(fmap=fmap_filename,importance_type='gain')).to_frame())
            feat_imp.columns=['Weight','Gain']
            feat_imp = feat_imp.join(pd.Series(model.get_score(fmap=fmap_filename,importance_type='cover')).to_frame())
            feat_imp.columns=['Weight','Gain','Cover']
            feat_imp['FeatureName'] = feat_imp.index
            feat_imp['Model'] = model_name
            feat_imp['fold'] = ind
            
            FI_df=pd.concat([FI_df,feat_imp])
  
            if XGBFirFlg:
                print('Feature Interaction')
                interactions_data_path = '/opt/ml/processing/output_importance/interactions_%s_%s.xlsx'%(model_name,ind)
                xgbfir.saveXgbFI(model, feature_names=featureset,  TopK = 500,  MaxTrees = 500, MaxInteractionDepth = 2, OutputXlsxFile = interactions_data_path)           
    
    print('Averaging results')
    #FI
    num_folds=FI_df['fold'].max()+1
    #number of columns with folds scores depends on the number of folds (num_folds) We do not know in advance how many of them exist in the results
    folds_train_columns=[]
    folds_test_columns=[]
    folds_gain_columns=[]
    folds_weight_columns=[]
    folds_cover_columns=[]
    for i in range(0,int(num_folds),1):
        folds_train_columns.append('train-%s-fold'%i)
        folds_test_columns.append('test-%s-fold'%i)
        folds_gain_columns.append('gain-%s'%i)
        folds_weight_columns.append('weight-%s'%i)
        folds_cover_columns.append('cover-%s'%i)
    print('1. Feature importance')
    FI_gain=FI_df[['Model','fold','FeatureName','Gain']]
    FI_gain=FI_gain.sort_values(['Model','fold'], ascending=[False,True])
    FI_gain = pd.pivot_table(FI_gain, index=['Model','FeatureName'], columns=['fold'])
    FI_gain.reset_index( drop=False, inplace=True )
    FI_gain.columns = FI_gain.columns.droplevel(0)
    FI_gain.columns =['Model','feature']+folds_gain_columns
    FI_gain['gain-mean']=FI_gain[folds_gain_columns].mean(axis=1)
    FI_gain['gainc-std']=FI_gain[folds_gain_columns].std(axis=1)
    FI_gain['gain-sem']=FI_gain[folds_gain_columns].sem(axis=1)
    #
    FI_weight=FI_df[['Model','fold','FeatureName','Weight']]
    FI_weight=FI_weight.sort_values(['Model','fold'], ascending=[False,True])
    FI_weight = pd.pivot_table(FI_weight, index=['Model','FeatureName'], columns=['fold'])
    FI_weight.reset_index( drop=False, inplace=True )
    FI_weight.columns = FI_weight.columns.droplevel(0)
    FI_weight.columns =['Model','feature']+folds_weight_columns
    FI_weight['weight-mean']=FI_weight[folds_weight_columns].mean(axis=1)
    FI_weight['weightc-std']=FI_weight[folds_weight_columns].std(axis=1)
    FI_weight['weight-sem']=FI_weight[folds_weight_columns].sem(axis=1)   
    #
    FI_cover=FI_df[['Model','fold','FeatureName','Cover']]
    FI_cover=FI_cover.sort_values(['Model','fold'], ascending=[False,True])
    FI_cover = pd.pivot_table(FI_cover, index=['Model','FeatureName'], columns=['fold'])
    FI_cover.reset_index( drop=False, inplace=True )
    FI_cover.columns = FI_cover.columns.droplevel(0)
    FI_cover.columns =['Model','feature']+folds_cover_columns
    FI_cover['cover-mean']=FI_cover[folds_cover_columns].mean(axis=1)
    FI_cover['coverc-std']=FI_cover[folds_cover_columns].std(axis=1)
    FI_cover['cover-sem']=FI_cover[folds_cover_columns].sem(axis=1) 
    FI_df=pd.merge(FI_gain, FI_weight, on=['Model','feature'], how='inner')
    FI_df=pd.merge(FI_df, FI_cover, on=['Model','feature'], how='inner')
    
    print('Saving importance...')
    FI_df.to_csv(importance_data_path, header=True, index=False)
    
    print('2. Prediction')
    Prediction_df[model_name]=Prediction_df.mean(axis=1)
    dataset[model_name]=Prediction_df[model_name]
    
    print('Saving predictions...')
    Prediction_df.to_csv(prediction_data_path, header=True, index=False)
    
    print('2. Scores')
    
    
    train_ROCAUC = Scores_df[['Model','fold','Train ROC-AUC']].copy()
    train_ROCAUC.columns=['Model','fold','train:auc']
    train_ROCAUC=train_ROCAUC.sort_values(['Model','fold'], ascending=[False,True])
    train_ROCAUC = pd.pivot_table(train_ROCAUC, index=['Model'], columns=['fold'])
    train_ROCAUC.reset_index( drop=False, inplace=True )
    train_ROCAUC.columns = train_ROCAUC.columns.droplevel(0)
    train_ROCAUC.columns =['Model']+folds_train_columns
    train_ROCAUC['train-auc-mean']=train_ROCAUC[folds_train_columns].mean(axis=1)
    train_ROCAUC['train-auc-std']=train_ROCAUC[folds_train_columns].std(axis=1)
    train_ROCAUC['train-auc-sem']=train_ROCAUC[folds_train_columns].sem(axis=1)

    test_ROCAUC = Scores_df[['Model','fold','Test ROC-AUC']].copy()
    test_ROCAUC.columns=['Model','fold','test:auc']
    test_ROCAUC=test_ROCAUC.sort_values(['Model','fold'], ascending=[False,True])
    test_ROCAUC = pd.pivot_table(test_ROCAUC, index=['Model'], columns=['fold'])
    test_ROCAUC.reset_index( drop=False, inplace=True )
    test_ROCAUC.columns = test_ROCAUC.columns.droplevel(0)
    test_ROCAUC.columns =['Model']+folds_test_columns
    test_ROCAUC['test-auc-mean']=test_ROCAUC[folds_test_columns].mean(axis=1)
    test_ROCAUC['test-auc-std']=test_ROCAUC[folds_test_columns].std(axis=1)
    test_ROCAUC['test-auc-sem']=test_ROCAUC[folds_test_columns].sem(axis=1)
    
    AUC_Scores_df = pd.merge(train_ROCAUC, test_ROCAUC, on=['Model'], how='inner')
    
    train_gini = Scores_df[['Model','fold','Train gini']].copy()
    train_gini.columns=['Model','fold','train:gini']
    train_gini=train_gini.sort_values(['Model','fold'], ascending=[False,True])
    train_gini = pd.pivot_table(train_gini, index=['Model'], columns=['fold'])
    train_gini.reset_index( drop=False, inplace=True )
    train_gini.columns = train_gini.columns.droplevel(0)
    train_gini.columns =['Model']+folds_train_columns
    train_gini['train-gini-mean']=train_gini[folds_train_columns].mean(axis=1)
    train_gini['train-gini-std']=train_gini[folds_train_columns].std(axis=1)
    train_gini['train-gini-sem']=train_gini[folds_train_columns].sem(axis=1)
    
    
    test_gini = Scores_df[['Model','fold','Test gini']].copy()
    test_gini.columns=['Model','fold','test:gini']
    test_gini=test_gini.sort_values(['Model','fold'], ascending=[False,True])
    test_gini = pd.pivot_table(test_gini, index=['Model'], columns=['fold'])
    test_gini.reset_index( drop=False, inplace=True )
    test_gini.columns = test_gini.columns.droplevel(0)
    test_gini.columns =['Model']+folds_test_columns
    test_gini['test-gini-mean']=test_gini[folds_test_columns].mean(axis=1)
    test_gini['test-gini-std']=test_gini[folds_test_columns].std(axis=1)
    test_gini['test-gini-sem']=test_gini[folds_test_columns].sem(axis=1)
    
    gini_Scores_df = pd.merge(train_gini, test_gini, on=['Model'], how='inner')
    
    print('3. Scores on folds average prediction')
    
    test_roc_auc=roc_auc_score(dataset[(dataset.cal_year == args.split_year)][target_column], dataset[(dataset.cal_year == args.split_year)][model_name])
    train_roc_auc=roc_auc_score(dataset[(dataset.cal_year < args.split_year)][target_column], dataset[(dataset.cal_year < args.split_year)][model_name])
    
    test_gini=gini(dataset[(dataset.cal_year == args.split_year)][target_column],dataset[(dataset.cal_year == args.split_year)][model_name])/gini(dataset[(dataset.cal_year == args.split_year)][target_column],dataset[(dataset.cal_year == args.split_year)][target_column])
    train_gini=gini(dataset[(dataset.cal_year < args.split_year)][target_column],dataset[(dataset.cal_year < args.split_year)][model_name])/gini(dataset[(dataset.cal_year < args.split_year)][target_column],dataset[(dataset.cal_year < args.split_year)][target_column])
    
    TestingDataResults = pd.DataFrame(list(zip([model_name],[-1],[train_roc_auc],[test_roc_auc],[train_gini],[test_gini])), 
               columns =['Model','fold','Train ROC-AUC','Test ROC-AUC','Train gini','Test gini'])
      
    FinalDataResults = pd.DataFrame(list(zip([train_roc_auc],[test_roc_auc],[train_gini],[test_gini])), 
               columns =['final-train-auc','final-test-auc','final-train-gini','final-test-gini'])
    
    AUC_Scores_df=pd.concat([AUC_Scores_df,FinalDataResults[['final-train-auc','final-test-auc']]],axis=1)
    gini_Scores_df=pd.concat([gini_Scores_df,FinalDataResults[['final-train-gini','final-test-gini']]],axis=1)
    
    print('Saving scores...')
    AUC_Scores_df.to_csv(auc_metrics_data_path, header=True, index=False)
    gini_Scores_df.to_csv(gini_metrics_data_path, header=True, index=False)
