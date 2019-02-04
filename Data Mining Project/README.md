# Data Mining Project

# 1. Introduction

This Data Mining project will be using dataset from the UCI machine learn- ing lab, representing 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. The data contains 50+ attributes such as race, gender, age, HbA1c test result, diabetic medications, etc. Our task is to predict, as accurately as possible, whether a person will be readmitted within 30 days after being discharged from the hospital.

# 2. Data Preprocessing 
The original data set contains 101766 instances with 50 features.

2.1 Feature Correlation 


2.2 Missing Values 

We remove the instances with feature “race” == ‘?’ or “gender” == ‘Unknown/ Invalid’ since they only contain a small part (less than 2%) of the entire data set. 

After performing the above operation, we were left with 69667 instances with 21 features for further analysis. 

2.3.3 Other Processing Steps 

Converted to binary values (0 and 1): “readmitted”, “diabetesMed”, “change”, “glipizide”, “metformin”, “A1Cresult”, “gender” 

Replaced by numerical values： “ages” 

For “diag_1”, we sort the column to nine different groups includes: Circulatory,  Injury, Respiratory, Digestive, Diabetes, Musculoskeletal, Genitourinary, Neoplasms and Others(groups less than 3.5% of instances) by using icd9 codes. 

# 3. Algorithms


# 4. Result
For this project, we created four testing models, including KNN, SVM, Random Forest and tested them on two processed data sets, each of which underwent a different imputation method. For our optimal model we chose Random Forest which provided the highest recall score. Logistic Regression also came pretty close to the optimal model in terms of accuracy.  
 

