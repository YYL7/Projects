# Data Mining Project

# 1. Introduction

This Data Mining project will be using dataset from the UCI machine learning lab, representing 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. The data contains 50 attributes such as race, gender, age, HbA1c test result, diabetic medications, etc. Our task is to predict, as accurately as possible, whether a person will be readmitted within 30 days after being discharged from the hospital.

# 2. Data Preprocessing 
The original data set contains 101,766 instances with 50 features.

2.1 Feature Correlation 

We droped the entire columns of features where they have more than half of the records missing. 

Features that have too many (more than 95%) same value are considered not informative, so we drop them as well. 

2.2 Missing Values 

We remove the instances with feature “race” == ‘?’ or “gender” == ‘Unknown/ Invalid’ since they only contain a small part (less than 2%) of the entire data set. 

2.3 Other Processing Steps 

Converted to binary values (0 and 1): “readmitted”, “diabetesMed”, “change”, “glipizide”, “metformin”, “A1Cresult”, “gender” 

Replaced by numerical values： “ages” 

For “diag_1”, we sort the column to nine different groups includes: Circulatory,  Injury, Respiratory, Digestive, Diabetes, Musculoskeletal, Genitourinary, Neoplasms and Others(groups less than 3.5% of instances) by using icd9 codes. 

2.4 Data Summary

After performing the above operation, we were left with 69,667 instances with 21 features for further analysis. 

Lables: readmitted (0/1)

Lable 1: Readimited within 30 days. Total number of lable 1 instance is 6,167.

Lable 0: Not get readmitted within 30 days or get readmitted after 30 days. Total number of lable 0 instance is 63,500.

# 3. Algorithms

For the following models, all values are treated as numeric value, for all categorical features such as “diag_1” , “admission_type_id” etc. we use one-hot encoding to convert them to binary features.  And we also normalized each numeric features. There are 41 features total for our evaluation, given by the PCC(Pearson correlation coefficient). 

3.1 KNN

3.1.1 Neighbor Selection

We look for the best k neighbor value from odd numbers in the range of (1, 100). The judgement is based on the recall score of label 1, because the label 1 occupies one tenth of the whole dataset, which means more predicted values are label 0 rather than label 1. The recall score of label 1 could precisely show the effectiveness of this model, compared to that of label 0.  

Through the application of Cross Validation, the model got an average recall score each time. After sorting by the recall scores, our team get the best neighbor equal to 53. 

3.1.2 Features Selection

For KNN model, Our team prefer to choose filter method, which is based on Pearson Correlation Coefficient,  to select features used. We rank the most related features. Due to the incredible amount of data, our team decide to select at most 20 features to run the model. The training data starts from only one feature that is the top one from the rank. Then adding one features every time, the model shows rates including accurate rate, recall score rate with label 1 and 0, precision rate and f1 scores. 

From the plot, we can see that the first twelve elements reached almost best performance with comparably running time. Hence, the KNN model used the first 11 most related features from the whole dataset.

The recall score of label 1 is 0.5427118528745769. The recall score of label 0 is 0.595380947004105. The accuracy is 0.5725093969721744. 

3.2 SVM

We built our SVM model using top 3, 5, 8, 10, 12, 15 features among the PCC ranks.The maximum recall for TP(1) is achieved with 0.5311 when the model includes top 10 features among the rank. 

3.3 Logistic Regression

We used feature selection based on PCC to rank the features’ correlation to the features and collect the best features to make the future prediction. And we chose the top 23 features among the PCC ranks. 

The overall precision and recall scores were decent but not great. A 61% recall means that the model was able to recall 61% of all the right answers and a 62% precision shows that the model needed to take many guesses before it can get to the right answers.  

# 4. Conclusion and Future Work 

For this project, we created three models, including KNN, SVM and Logistic Regression. The Logistic Regression model was considered to be the best model in regard of running time performance and prediction result with overall recall score of 0.61, representing that the model was able to recall 61% of all the right answers. In the future, we would like to try ensemble models or Ada-boosting in order to get more stable results and a new way of balancing the data such as bagging. 

