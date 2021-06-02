#ModelTraining_Gini_EvalMetric.py uses XGBoost training with custom evaluation metric - gini. Use custom image_uri and metric defnitions.

#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.
from __future__ import print_function

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

def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)
def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', gini(y, pred) / gini(y, y)

def _xgb_train(params, dtrain, evals, num_boost_round, early_stopping_rounds, model_dir, output_data_dir, GetFIFlg,GetTestScoreFlg,GetTestPredFlg, is_master):
    """Run xgb train on arguments given with rabit initialized.

    This is our rabit execution function.

    :param args_dict: Argument dictionary used to run xgb.train().
    :param is_master: True if current node is master host in distributed training,
                        or is running single node training job.
                        Note that rabit_run will include this argument.
    """
    progress = dict()
    booster = xgb.train(params=params,
                        dtrain=dtrain,
                        evals=evals,
                        feval=gini_xgb,
                        maximize=True,
                        num_boost_round=num_boost_round,
                        early_stopping_rounds=early_stopping_rounds,
                        evals_result=progress,
                        verbose_eval=100)
    
    print('Eval results')    
    train_error=progress['train']['gini']
    eval_error=progress['validation']['gini']
    results_pd=pd.DataFrame({'train':train_error,'valid':eval_error},columns=['train','valid'])
    
    
    #feature importance
    if GetFIFlg=='Y':
        fi_weight =booster.get_score(importance_type='weight')
        fi_gain = booster.get_score(importance_type='gain')
        fi_cover= booster.get_score(importance_type='cover')
        fi_weight_pd = pd.DataFrame(fi_weight.items(),columns=['feature','weight'])
        fi_gain_pd = pd.DataFrame(fi_gain.items(),columns=['feature','gain'])
        fi_cover_pd = pd.DataFrame(fi_cover.items(),columns=['feature','cover'])
        fi_pd=pd.merge(fi_gain_pd, fi_weight_pd, on='feature', how='inner')
        fi_pd=pd.merge(fi_pd, fi_cover_pd, on='feature', how='inner')

    #Prediction on test data ...
    if 'Y' in (GetTestScoreFlg,GetTestPredFlg):
        df_prediction=pd.DataFrame()
        df_prediction['actual']=dtest.get_label()
        df_prediction['pred']=booster.predict(dtest)
   
        #Test scores from test prediction   
        df_score = pd.DataFrame()
        df_score['gini-test']=[gini(df_prediction['actual'], df_prediction['pred'])/gini(df_prediction['actual'],df_prediction['actual'])]
        
    
    if is_master:
        model_location = model_dir + '/xgboost-model'
        pkl.dump(booster, open(model_location, 'wb'))
        logging.info("Stored trained model at {}".format(model_location))
        
        if not os.path.exists(output_data_dir):
            os.makedirs(output_data_dir)

        result_location = os.path.join(output_data_dir, 'eval_results.csv')
        print('Saving eval results at {}'.format(result_location))
        logging.info('Saving eval results at {}'.format(result_location))
        results_pd.to_csv(result_location, header=True, index=False)
        
        if GetFIFlg=='Y':
            fi_location = os.path.join(output_data_dir, 'fi.csv')
            print('Saving feature importance at {}'.format(fi_location))
            logging.info('Saving feature importance at {}'.format(fi_location))
            fi_pd.to_csv(fi_location, header=True, index=False)
        
        if GetTestPredFlg=='Y':
            predictions_location = os.path.join(output_data_dir, 'test_predictions.csv')
            print('Saving test predictions at {}'.format(predictions_location))
            logging.info('Saving test predictions at {}'.format(predictions_location))
            df_prediction.to_csv(predictions_location, header=True, index=False)
            
        if GetTestScoreFlg=='Y':        
            test_score_location = os.path.join(output_data_dir, 'test_score.csv')
            print('Saving test score  at {}'.format(test_score_location))
            logging.info('Saving test score  at {}'.format(test_score_location))        
            df_score.to_csv(test_score_location, header=True, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here.
    parser.add_argument('--max_depth', type=int,)
    parser.add_argument('--eta', type=float)
    parser.add_argument('--objective', type=str)
    parser.add_argument('--num_round', type=int)

    parser.add_argument('--early_stopping_rounds', type=int)
    parser.add_argument('--booster', type=str)
    #parser.add_argument('--eval_metric', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scale_pos_weight', type=float)
    parser.add_argument('--colsample_bylevel', type=float)
    parser.add_argument('--colsample_bytree', type=float)
    parser.add_argument('--subsample', type=float)
    parser.add_argument('--max_delta_step', type=int)
    
    
    parser.add_argument('--GetFIFlg', type=str, default='N')
    parser.add_argument('--GetTestScoreFlg', type=str, default='N')
    parser.add_argument('--GetTestPredFlg', type=str, default='N')            
            

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    
    parser.add_argument('--output_data_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--sm_hosts', type=str, default=os.environ.get('SM_HOSTS'))
    parser.add_argument('--sm_current_host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    args, _ = parser.parse_known_args()

    # Get SageMaker host information from runtime environment variables
    sm_hosts = json.loads(args.sm_hosts)
    sm_current_host = args.sm_current_host

    dtrain = get_dmatrix(args.train, 'csv')
    dval = get_dmatrix(args.validation, 'csv')
    watchlist = [(dtrain, 'train'), (dval, 'validation')] if dval is not None else [(dtrain, 'train')]

      
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
        #'eval_metric':args.eval_metric,
        'disable_default_eval_metric': '1',
        'scale_pos_weight':args.scale_pos_weight,
        'colsample_bylevel': args.colsample_bylevel,
        'colsample_bytree': args.colsample_bytree,
        'subsample': args.subsample,
        'max_delta_step':args.max_delta_step
        }

    xgb_train_args = dict(
        params=train_hp,
        dtrain=dtrain,
        evals=watchlist,
        num_boost_round=args.num_round,
        early_stopping_rounds=args.early_stopping_rounds,
        model_dir=args.model_dir,
        output_data_dir=args.output_data_dir,
        GetFIFlg=args.GetFIFlg,
        GetTestScoreFlg=args.GetTestScoreFlg,
        GetTestPredFlg=args.GetTestPredFlg)

    if len(sm_hosts) > 1:
        # Wait until all hosts are able to find each other
        entry_point._wait_hostname_resolution()

        # Execute training function after initializing rabit.
        distributed.rabit_run(
            exec_fun=_xgb_train,
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
            _xgb_train(**xgb_train_args)
        else:
            raise ValueError("Training channel must have data to train model.")

#not clear what's this for multi-node training?
def model_fn(model_dir):
    """Deserialize and return fitted model.

    Note that this should have the same name as the serialized model in the _xgb_train method
    """
    model_file = 'xgboost-model'
    booster = pkl.load(open(os.path.join(model_dir, model_file), 'rb'))
    return booster
