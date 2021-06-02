import time
import pandas as pd
import numpy as np
import openpyxl 
import sagemaker
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from smexperiments.trial_component import TrialComponent

def cleanup_experiment(Experiment_name): 
    try:
        experiment = Experiment.load(experiment_name=Experiment_name)
        for trial_summary in experiment.list_trials():
            trial = Trial.load(trial_name=trial_summary.trial_name)
            for trial_component_summary in trial.list_trial_components():
                tc = TrialComponent.load(
                    trial_component_name=trial_component_summary.trial_component_name)
                trial.remove_trial_component(tc)
                try:
                    # comment out to keep trial components
                    tc.delete()
                except:
                    # tc is associated with another trial
                    continue
                # to prevent throttling
                time.sleep(.5)
            trial.delete()
            experiment_name = experiment.experiment_name
        experiment.delete()        
    except Exception as ex:
        if 'ResourceNotFound' in str(ex):
            print('%s is a new experiment. Nothing to delete'%Experiment_name)
    

def cleanup_trial(Experiment_name, Trial_name):
    experiment = Experiment.load(experiment_name=Experiment_name)
    for trial_summary in experiment.list_trials():
            trial = Trial.load(trial_name=trial_summary.trial_name)
            #print(trial_summary.trial_name)
            if trial_summary.trial_name==Trial_name:
                for trial_component_summary in trial.list_trial_components():
                    tc = TrialComponent.load(trial_component_name=trial_component_summary.trial_component_name)
                    print(trial_component_summary.trial_component_name)
                    trial.remove_trial_component(tc)
                    try:
                        # comment out to keep trial components
                        tc.delete()
                    except:
                        # tc is associated with another trial
                        continue
                    # to prevent throttling
                    time.sleep(.5)
                trial.delete()
                

# create the experiment if it doesn't exist
def create_experiment(Experiment_name,Experiment_description = None ):
    try:
        experiment = Experiment.load(experiment_name=Experiment_name)
    except Exception as ex:
        if "ResourceNotFound" in str(ex):
            experiment = Experiment.create(experiment_name = Experiment_name,
                                    description = Experiment_description)
            
# create the trial if it doesn't exist
def create_trial(Experiment_name, Trial_name):
    try:
        trial = Trial.load(trial_name=Trial_name)
    except Exception as ex:
        if "ResourceNotFound" in str(ex):
            trial = Trial.create(experiment_name=Experiment_name, trial_name=Trial_name)
            
#Waiting till the end of all jobs
#If there are None the waiting cycle will not start
#Processing jobs should take ~10-15 min
#Waiting till complete
def wait_processing_jobs(processors,check_every_sec,print_every_n_output, wait_min):
    n = 0
    #If there are not complete  jobs in ~10 minutes, skip
    t = 0
    minutes_to_wait=wait_min*60/check_every_sec
    ProcessorsFlg=len(processors)>0
    while (True & ProcessorsFlg):
        statuses = list()
        n = n + 1
        for p in processors:
            name=p.jobs[-1].describe()['ProcessingJobName']
            status=p.jobs[-1].describe()['ProcessingJobStatus']
            if n==print_every_n_output:
                print('Processing job %s status: %s'%(name,status))
            statuses.append(status)
        if 'InProgress' in statuses:
            if n==print_every_n_output:
                print('Continue waiting...')
                n = 0
        else:
            if set(statuses)=={'Completed'}:
                print('All Processing Jobs are Completed')
            else:
                print('Something went wrong.')
            break 
        t = t+1
        if t>minutes_to_wait:
            raise Exception('Something went wrong. Processing jobs are still running.')
        time.sleep(check_every_sec)
        
#Waiting till the end of all jobs
#If there are None the waiting cycle will not start
#Processing jobs should take ~10-15 min
#Waiting till complete
def wait_transform_jobs(processors,tranform_jobs,check_every_sec,print_every_n_output,wait_min):
    n = 0
    #If there are not complete  jobs in ~10 minutes, skip
    t = 0
    minutes_to_wait=wait_min*60/check_every_sec
    ProcessorsFlg=len(processors)>0
    while (True & ProcessorsFlg):
        statuses = list()
        n = n + 1
        for p,name in zip(processors,tranform_jobs):
            status=p.sagemaker_session.describe_transform_job(name)['TransformJobStatus']
            if n==print_every_n_output:
                print('Transforming job %s status: %s'%(name,status))
            statuses.append(status)
        if 'InProgress' in statuses:
            if n==print_every_n_output:
                print('Continue waiting...')
                n = 0
        else:
            if set(statuses)=={'Completed'}:
                print('All Transforming Jobs are Completed')
            else:
                print('Something went wrong.')
            break 
        t = t+1
        if t>minutes_to_wait:
            raise Exception('Something went wrong. Transforming jobs are still running.')
        time.sleep(check_every_sec)

#Waiting till the end of all jobs
#If there are None the waiting cycle will not start
#Processing jobs should take ~10-15 min
#Waiting till complete
def wait_training_jobs(processors,check_every_sec,print_every_n_output, wait_min):
    n = 0
    #If there are not complete  jobs in ~10 minutes, skip
    t = 0
    minutes_to_wait=wait_min*60/check_every_sec
    ProcessorsFlg=len(processors)>0
    while (True & ProcessorsFlg):
        statuses = list()
        n = n + 1
        for p in processors:
            name=p.jobs[-1].describe()['TrainingJobName']
            status=p.jobs[-1].describe()['TrainingJobStatus']
            if n==print_every_n_output:
                print('Training job %s status: %s'%(name,status))
            statuses.append(status)
        if 'InProgress' in statuses:
            if n==print_every_n_output:
                print('Continue waiting...')
                n = 0
        else:
            if set(statuses)=={'Completed'}:
                print('All Training Jobs are Completed')
            else:
                print('Something went wrong.')
            break 
        t = t+1
        if t>minutes_to_wait:
            raise Exception('Something went wrong. Training jobs are still running.')
        time.sleep(check_every_sec)
        
        
#Saving into log (Excel file)
def SaveToExperimentLog(Experiments_file, LogEntry, data):
    book = openpyxl.load_workbook(Experiments_file)
    writer = pd.ExcelWriter(Experiments_file, engine='openpyxl') 
    writer.book = book

    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

    data.to_excel(writer, LogEntry[0:29],index=False)

    writer.save()
    writer.close()
    
#Saving charts to log (Excel file)
def SaveChartToExperimentLog(Experiments_file, LogEntry, Start_Position, Step, lst_img_filenames):
    book = openpyxl.load_workbook(Experiments_file)
    ws = book[LogEntry[0:29]]
    p = Start_Position + 10
    for f in lst_img_filenames:
        img = openpyxl.drawing.image.Image(f)
        ws.add_image(img)
        position = p
        img.anchor = 'A%s'%p
        p = p + Step

    book.save(Experiments_file)


import numpy as np
import math
import scipy.stats as stats

# Nadeau and Bengio corrected paired t-test
# https://link.springer.com/content/pdf/10.1023/A:1024068626366.pdf
# https://www.cs.waikato.ac.nz/~eibe/pubs/bouckaert_and_frank.pdf

def corrected_paired_ttest(data1, data2, n_training_size_folds, n_test_size_folds, alpha):
    #corrected paired t-test
    diff=[np.abs(y - x) for y, x in zip(data1, data2)]
    n = len(diff)
    m = np.mean(diff)
    #it's important to provide ddof=1 (delta degrees of freedom) in numpy var to calculate variance with degre of freedom n - 1.
    v = np.var(diff,ddof=1)
    t = m/math.sqrt(v*(1/n + n_test_size_folds/n_training_size_folds))
    
    #degree of freedom
    df = n - 1
    
    #Critical value for Two-tailed test from t distribution table:
    critical_value=stats.t.ppf(q=1-alpha/2, df=df)
    
    #p-value - probability of getting a more extreme value - for two-sided test
    pvalue = 2*(1-stats.t.cdf(t, df))
    
    return t, critical_value, pvalue

def corrected_confidence_interval(data1, data2, n_training_size_folds, n_test_size_folds, confidence=0.95):
    diff=[np.abs(y - x) for y, x in zip(data1, data2)]
    n = len(diff)
    m = np.mean(diff)
    v = np.var(diff, ddof=1) 
    df = n - 1  
    t = stats.t.ppf((1 + confidence)/2, df) 

    lower = m - t * math.sqrt(v*(1/n + n_test_size_folds/n_training_size_folds))
    upper = m + t * math.sqrt(v*(1/n + n_test_size_folds/n_training_size_folds))
    return lower, upper