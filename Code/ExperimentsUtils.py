import time
import pandas as pd
import numpy as np
from openpyxl import load_workbook
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
    book = load_workbook(Experiments_file)
    writer = pd.ExcelWriter(Experiments_file, engine='openpyxl') 
    writer.book = book

    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

    data.to_excel(writer, LogEntry[0:29],index=False)

    writer.save()
    writer.close()