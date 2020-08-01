# Prediction-of-Patients-with-Parkinsons
Performing a wide array of statistical analysis including Uni-variate &amp; Multivariate analysis, Principal Component Analysis for the identification of key features. Followed by Logistic Regression &amp; Random Forest Classifiers to predict patients exhibiting Parkinson's disease.

Dataset: 
The dataset is composed of a range of biomedical voice measurement samples from a cohort of patients with Parkinson's disease (PD). 
Each feature in the dataset featured a particular voice measure and each observation corresponds to one of the 195 voice recordings from the individual patients.

Aim:
To successfully be able to discriminate healthy people from those with PD (indicated by the status column with 1/0 values)

Approach:
1. Performed extensive Exploratory data analysis to get an overview of all the numerical features indicating various voice measures. 
2. Implemented PCA to identify the key predictors and to reduce the dimensionality of the features.
3. Having identified the important features, trained a Random Forest and Logistic Regression classifier to predict the results on test data

Results:
Achieved 85% accuracy and 77% Recall with Logistic Regression on the test dataset.
