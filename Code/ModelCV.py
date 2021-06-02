#ModelCV.py uses standard XGBoost evaluation metric(AUC) in CV




import argparse
import json
import logging
import os
import pandas as pd
import pickle as pkl

from sagemaker_containers import entry_point
from sagemaker_xgboost_container.data_utils import get_dmatrix
from sagemaker_xgboost_container import distributed

import xgboost as xgb

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score

import re

def cv_misc_callback(oof_train_scores:list, oof_valid_scores:list, best_models:list, NeedModelsFlg='N', maximize=True):
    """
    It's called inside XGB CV to catch individual folds scores
    """    
    state = {}
    def init(env):
        if maximize:
            state['best_score'] = -np.inf
        else:
            state['best_score'] = np.inf   
    def callback(env):
        #fold best model if flag
        if NeedModelsFlg=='Y':
            if not state:
                init(env)
            best_score = state['best_score']
            score = env.evaluation_result_list[-1][1]
            if (maximize and score > best_score) or (not maximize and score < best_score):
                for i, cvpack in enumerate(env.cvfolds): 
                    best_models[i]=cvpack.bst
                state['best_score'] = score    
        #all iterations folds scores
        folds_train_scores = []
        folds_valid_scores = []
        for i, cvpack in enumerate(env.cvfolds):
            scores = cvpack.eval(iteration=0,feval=None)
            scores_l = re.split(': |\t',scores)
            train_score=scores_l[1].rpartition(':')[2]
            valid_score=scores_l[2].rpartition(':')[2]
            folds_train_scores.append(train_score)
            folds_valid_scores.append(valid_score)
        oof_train_scores.append(folds_train_scores)
        oof_valid_scores.append(folds_valid_scores)
    callback.before_iteration = False
    return callback

def _xgb_cv(params, dtrain,  num_boost_round, nfold, early_stopping_rounds, model_dir, output_data_dir, GetFIFlg,GetTestScoreFlg,GetTestPredFlg,is_master):
    """Run xgb cv on arguments given with rabit initialized.

    This is our rabit execution function.

    :param args_dict: Argument dictionary used to run xgb.train().
    :param is_master: True if current node is master host in distributed training,
                        or is running single node training job.
                        Note that rabit_run will include this argument.
    """
    oof_train_scores = []
    oof_valid_scores = []
    best_models=[None]*nfold
    NeedModelsFlg = 'Y' if 'Y' in (GetFIFlg,GetTestScoreFlg,GetTestPredFlg) else 'N'
    cv_results=xgb.cv(params,  dtrain,  num_boost_round=num_boost_round,
                 nfold=nfold, stratified=True, shuffle=True,early_stopping_rounds=early_stopping_rounds, seed=42
                      ,callbacks=[cv_misc_callback(oof_train_scores, oof_valid_scores,best_models,NeedModelsFlg,True), xgb.callback.print_evaluation(period=20)]
                     )



    #scores to dataframe
    df_oof_train_scores = pd.DataFrame.from_records(oof_train_scores).apply(pd.to_numeric)
    df_oof_valid_scores = pd.DataFrame.from_records(oof_valid_scores).apply(pd.to_numeric)


    
    #only folds scores columns names
    columns = df_oof_train_scores.columns.tolist()

    
    #mean and std, sem 
    df_oof_train_scores['std'] = df_oof_train_scores[columns].std(axis=1)
    df_oof_valid_scores['std'] = df_oof_valid_scores[columns].std(axis=1)
    df_oof_train_scores['sem'] = df_oof_train_scores[columns].sem(axis=1)
    df_oof_valid_scores['sem'] = df_oof_valid_scores[columns].sem(axis=1)    
    df_oof_train_scores['mean'] = df_oof_train_scores[columns].mean(axis=1)
    df_oof_valid_scores['mean'] = df_oof_valid_scores[columns].mean(axis=1)
    
    #best models feature importance 
    if GetFIFlg=='Y':
        oof_fi_weight_best = {}
        oof_fi_gain_best = {}
        oof_fi_cover_best = {}
        for i in range(0,nfold):
            oof_fi_weight_best[i]=best_models[i].get_score(importance_type='weight')
            oof_fi_gain_best[i]= best_models[i].get_score(importance_type='gain')
            oof_fi_cover_best[i]= best_models[i].get_score(importance_type='cover')
    
        #converting to dataframe
        df_oof_fi_weight_best = pd.DataFrame(oof_fi_weight_best).apply(pd.to_numeric)
        df_oof_fi_gain_best = pd.DataFrame(oof_fi_gain_best).apply(pd.to_numeric)
        df_oof_fi_cover_best = pd.DataFrame(oof_fi_cover_best).apply(pd.to_numeric)
    
    
        #mean and std, sem 
        df_oof_fi_weight_best['std'] = df_oof_fi_weight_best[columns].std(axis=1)
        df_oof_fi_gain_best['std'] = df_oof_fi_gain_best[columns].std(axis=1)
        df_oof_fi_cover_best['std'] = df_oof_fi_cover_best[columns].std(axis=1)
    
        df_oof_fi_weight_best['sem'] = df_oof_fi_weight_best[columns].sem(axis=1)
        df_oof_fi_gain_best['sem'] = df_oof_fi_gain_best[columns].sem(axis=1)
        df_oof_fi_cover_best['sem'] = df_oof_fi_cover_best[columns].sem(axis=1)
    
        df_oof_fi_weight_best['mean'] = df_oof_fi_weight_best[columns].mean(axis=1)
        df_oof_fi_gain_best['mean'] = df_oof_fi_gain_best[columns].mean(axis=1)
        df_oof_fi_cover_best['mean'] = df_oof_fi_cover_best[columns].mean(axis=1)
    
    
        #feature codes from index to column
        df_oof_fi_weight_best.reset_index(level=0, inplace=True)
        df_oof_fi_weight_best.columns=['feature'] + columns + ['std','sem','mean']
        df_oof_fi_gain_best.reset_index(level=0, inplace=True)
        df_oof_fi_gain_best.columns=['feature'] + columns + ['std','sem','mean']
        df_oof_fi_cover_best.reset_index(level=0, inplace=True)
        df_oof_fi_cover_best.columns=['feature'] + columns + ['std','sem','mean']
    
    if 'Y' in (GetTestScoreFlg,GetTestPredFlg):
        #Prediction on test data  from folds best models...
        df_prediction=pd.DataFrame()
        df_prediction['actual']=dtest.get_label()
        for i in range(0,nfold):
            df_prediction[i]=best_models[i].predict(dtest)
   
        #Test scores from test prediction   
        df_scores = pd.DataFrame()
        for i in range(0,nfold):
            df_scores[i]=[roc_auc_score(df_prediction['actual'], df_prediction[i])]

        df_scores['std'] = df_scores[columns].std(axis=1)
        df_scores['sem'] = df_scores[columns].sem(axis=1)
        df_scores['mean'] = df_scores[columns].mean(axis=1)

    if is_master:
        if not os.path.exists(output_data_dir):
            os.makedirs(output_data_dir)
            
        if NeedModelsFlg == 'Y':
            model_location = model_dir + '/xgboost-model'
            pkl.dump(best_models[0], open(model_location, 'wb'))
            print('Stored best model from 1st fold at {}'.format(model_location))
            logging.info('Stored best model from 1st fold at {}'.format(model_location))        
               
            print('Stored best models from all folds at {}'.format(output_data_dir))
            logging.info('Stored best models from all folds at {}'.format(output_data_dir))
        
            for i in range(0,nfold):
                model_location = output_data_dir + '/xgboost-model-fold'+str(i)
                pkl.dump(best_models[i], open(model_location, 'wb'))
        
        if  GetTestPredFlg=='Y':    
            predictions_location = os.path.join(output_data_dir, 'test_predictions.csv')
            print('Saving test predictions at {}'.format(predictions_location))
            logging.info('Saving test predictions at {}'.format(predictions_location))            
            df_prediction.to_csv(predictions_location, header=True, index=False)
        
        if  GetTestScoreFlg=='Y':
            oof_test_scores_location = os.path.join(output_data_dir, 'oof_test_scores.csv')
            print('Saving oof_test_scores at {}'.format(oof_test_scores_location))
            logging.info('Saving oof_test_scores at {}'.format(oof_test_scores_location))
            df_scores.to_csv(oof_test_scores_location, header=True, index=False)
        
        cv_result_location = os.path.join(output_data_dir, 'cv_results.csv')
        print('Saving cv results at {}'.format(cv_result_location))
        logging.info('Saving cv results at {}'.format(cv_result_location))
        cv_results.to_csv(cv_result_location, header=True, index=False)
        
        oof_train_scores_location = os.path.join(output_data_dir, 'oof_train_scores.csv')
        print('Saving oof_train_scores at {}'.format(oof_train_scores_location))
        logging.info('Saving oof_train_scores at {}'.format(oof_train_scores_location))
        df_oof_train_scores.to_csv(oof_train_scores_location, header=True, index=False)  
        
        oof_valid_scores_location = os.path.join(output_data_dir, 'oof_valid_scores.csv')
        print('Saving oof_valid_scores at {}'.format(oof_valid_scores_location))
        logging.info('Saving oof_valid_scores at {}'.format(oof_valid_scores_location))
        df_oof_valid_scores.to_csv(oof_valid_scores_location, header=True, index=False)
        
        if  GetFIFlg=='Y':
            oof_fi_weight_best_location = os.path.join(output_data_dir, 'oof_fi_weight_best.csv')
            print('Saving oof_fi_weight_best at {}'.format(oof_fi_weight_best_location))
            logging.info('Saving oof_fi_weight_best at {}'.format(oof_fi_weight_best_location))
            df_oof_fi_weight_best.to_csv(oof_fi_weight_best_location, header=True, index=False)  
        
            oof_fi_gain_best_location = os.path.join(output_data_dir, 'oof_fi_gain_best.csv')
            print('Saving oof_fi_gain_best at {}'.format(oof_fi_gain_best_location))
            logging.info('Saving oof_fi_gain_best at {}'.format(oof_fi_gain_best_location))
            df_oof_fi_gain_best.to_csv(oof_fi_gain_best_location, header=True, index=False)        
        
            oof_fi_cover_best_location = os.path.join(output_data_dir, 'oof_fi_cover_best.csv')
            print('Saving oof_fi_cover_best at {}'.format(oof_fi_cover_best_location))
            logging.info('Saving oof_fi_cover_best at {}'.format(oof_fi_cover_best_location))
            df_oof_fi_cover_best.to_csv(oof_fi_cover_best_location, header=True, index=False)  
           
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here.
    parser.add_argument('--max_depth', type=int,)
    parser.add_argument('--eta', type=float)
    parser.add_argument('--objective', type=str)
    parser.add_argument('--num_round', type=int)
    parser.add_argument('--nfold', type=int)
    parser.add_argument('--early_stopping_rounds', type=int)
    parser.add_argument('--booster', type=str)
    parser.add_argument('--eval_metric', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scale_pos_weight', type=float)
    parser.add_argument('--colsample_bylevel', type=float)
    parser.add_argument('--colsample_bytree', type=float)
    parser.add_argument('--subsample', type=float)
    parser.add_argument('--max_delta_step', type=int)
    parser.add_argument('--reg_lambda', type=float)
    parser.add_argument('--reg_alpha', type=float)
            
            

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    
    parser.add_argument('--output_data_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--sm_hosts', type=str, default=os.environ.get('SM_HOSTS'))
    parser.add_argument('--sm_current_host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    
    parser.add_argument('--GetFIFlg', type=str, default='N')
    parser.add_argument('--GetTestScoreFlg', type=str, default='N')
    parser.add_argument('--GetTestPredFlg', type=str, default='N')
                
                

    args, _ = parser.parse_known_args()

    # Get SageMaker host information from runtime environment variables
    sm_hosts = json.loads(args.sm_hosts)
    sm_current_host = args.sm_current_host

    dtrain = get_dmatrix(args.train, 'csv')
    
    dtest = get_dmatrix(args.test, 'csv')

    if not(dtest):
        if ((args.GetTestScoreFlg=='Y') | (args.GetTestPredFlg=='Y')):
            raise Exception('Please provide test data in a test channel for prediction and scores or set GetTestScoreFlg and GetTestPredFlg to N')

    train_hp = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'objective': args.objective,
        'booster': args.booster,
        'seed': args.seed,
        'eval_metric':args.eval_metric,
        'scale_pos_weight':args.scale_pos_weight,
        'colsample_bylevel': args.colsample_bylevel,
        'colsample_bytree': args.colsample_bytree,
        'subsample': args.subsample,
        'max_delta_step':args.max_delta_step,
        'reg_lambda': args.reg_lambda,
        'reg_alpha': args.reg_alpha
        }

    xgb_train_args = dict(
        params=train_hp,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        nfold=args.nfold, 
        early_stopping_rounds=args.early_stopping_rounds,
        model_dir=args.model_dir,
        output_data_dir=args.output_data_dir,
        GetFIFlg=args.GetFIFlg,
        GetTestScoreFlg=args.GetTestScoreFlg,
        GetTestPredFlg=args.GetTestPredFlg
    )

    if len(sm_hosts) > 1:
        # Wait until all hosts are able to find each other
        entry_point._wait_hostname_resolution()

        # Execute training function after initializing rabit.
        distributed.rabit_run(
            exec_fun=_xgb_cv,
            args=xgb_train_args,
            include_in_training=(dtrain is not None),
            hosts=sm_hosts,
            current_host=sm_current_host,
            update_rabit_args=True
        )
    else:
        # If single node training, call training method directly.
        if dtrain:
            xgb_train_args['is_master'] = True
            _xgb_cv(**xgb_train_args)
        else:
            raise ValueError("Training channel must have data to train model.")
