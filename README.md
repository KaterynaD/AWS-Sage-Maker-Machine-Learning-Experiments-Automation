# AWS-Sage-Maker-Machine-Learning-Experiments-Automation
Machine Learning experiments automation with the help of AWS Sage Maker (SageMaker Open Source XGBoost Classification and Insurance Property data).

## Why it's needed:
The same code can be run for different experiments over and over again with changes in configuration. Finally I have disconnected or lost experiment configuration (data file used, feature set, target, model parameters) and the model score, feature importance, partial dependency data, model files.
Saving all together in an experiment log (I used to and like Excel files and it's enough for me as an individual researcher) helps to organize my work.

Experiments are configured in an Excel file. The code reads configuration and runs AWS Sage Maker jobs in parallel.
Jobs results are read from AWS Sage Maker experiments or directly from S3 files and added to Excel files (local experiments logs). 
Finally total time of experiments is extracted and cost can be calculated.
Type of experiments:
1. Parameters and/or Features (original from a dataset or calculated) sets impact on a model score
-  Quick research without cross validaton "01.AWS SageMaker XGBoost Classification Training Model Experiment (Features and Parameters research).ipynb"
-  Deep models comparison with cross validation and performing t-test "02.AWS SageMaker XGBoost Classification Training Models with cross validation Experiment (Features and Parameters research)" Works well with limited number of comparing models 2-3. With 10+ models 10 folds failed to exrtact results.
-  Deep models comparison using native XGBoost CV "03.AWS SageMaker XGBoost Classification Training Models using native XGBoost CV for cross validation Experiment (Features and Parameters research)". Works for 40+ models and 10 folds. t-test is conducted at the end. XGBoost CV is extended to extract a best model and folds metrics. This additional information is stored in output.tar.gz

Training and cross validation can use a standard XGBoost evaluation metric (ROC-AUC) or custom (gini).
Each experiment run returns train/valid errors to estimate overfitting. There is an option to extract feature importance if needed and perform prediction on test data. Standard erros of the mean and standard deviation are returned and analyzed along with cross validation metric (standard ROC-AUC or custom gini) means.

2. Partial Dependency from AWS SageMaker XGBoost model -  "06. AWS Sagemaker XGBoost Partial Dependency.ipynb"
3. Model evaluation and feature importance
-  Download standard SageMaker XGBoost model from model.tar.gz, run prediction, calculate ROC-AUV and gini metrics, extract standard XGBoost feature importance and feature interactions using XGBFIR "04. AWS Sagemaker Trained Models Evaluation and Feature Importance.ipynb"
-  Extended native XGBoost CV returns all folds best models in output.tar.gz. "05. AWS Sagemaker XGBoost CV Best Models Evaluation and Feature Importance.ipynb" extracts all models from non-standard XGBoost location (output.tar.gz) , run prediction, calculate ROC-AUV and gini metrics, extract standard XGBoost feature importance and feature interactions using XGBFIR. The output result is an average with standard error and deviation based on all models.
4. Inference with Shap values "07. AWS Sagemaker XGBoost Batch Transform with ShapValues.ipynb"
5. "08. AWS Sagemaker Experiments Cost.ipynb" calculates costs of an experiment based on the registered jobs and used instances
6. "09.AWS Hyperparameter Tuning job for XGBoost Classification.ipynb" uses training individual models or cross validation (XGBoost CV) with standard or custom evaluation metris.
