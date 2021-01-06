# AWS-Sage-Maker-Machine-Learning-Experiments-Automation
Machine Learning experiments automation with the help of AWS Sage Maker using XGBoost Classification and Insurance Property data

## Why it's needed:
The same code can be run for different experiments over and over again with changes in configuration. Finally I have disconnected or lost experiment configuration (data file used, feature set,
target, model parameters) and the model score, feature importance, partial dependency data, model files.
Saving all together in an experiment log (I used to and like Excel files and it's enough for me as an individual researcher) helps to organize my work.

Experiments are configured in an Excel file. The code reads configuration and runs AWS Sage Maker jobs in parallel with experiment registry.
Jobs results are read from AWS Sage Maker experiments and added to Excel files (local experiments logs). 
Finally total time of experiments is extracted and cost can be calculated.
Type of experiments:
1. Features (original from a dataset or calculated) impact on a model score
-  02.Create training and validation datasets - Features Research.ipynb
-  03.Training Models - Features Research.ipynb
2. t-test to estimate a feature impact on a model score
-  "04. Create training and validation datasets for t-Test.ipynb"
-  05.Training Models and t-Test.ipynb
3. Overfitting test
-  06. Create training and validation datasets for overfitting test.ipynb
-  07.Training Models and overfitting test.ipynb
4. Best parameters research
-  08.Training Models - Parameters Research.ipynb
-  09. Hyperparameter Tuning job.ipynb
5. Partial Dependency
-  10.Create dataset for PartialDependency.ipynb
-  11. PartialDependency Inference.ipynb
-  12. PartialDependency Post Processing.ipynb
6. Model evaluation and feature importance
-  13. Models Evaluation and Feature Importance
7. Shap values
-  14. Create dataset for batch transform Need Testing.ipynb
-  15. Batch Transform - Prediction and ShapValues Need Testing.ipynb
8. 16. Experiments Cost.ipynb

ToDo:
1. Automate the flow using AWS Step functions or something similar. For now all parts should be run manually
2. Read more experiment configuration from a file instead of hard coding (parameters)
3. Not all intermediate results need to be saved in a final experiment log locally.
4. Add interactions extraction in Model evaluation and feature importance.
