# Applied Regression Analysis


# 1. Introduction
This project is to use R to build a logistic regression model and predict which pairs of daters want to have a second date after the speed dating experiment.

The dataset, which contains information on speed dating experiments, conducted on graduate and professional students. Each person in the experiment met with 10-20 randomly selected people of the opposite sex (only heterosexual pairings) for four minutes. After each speed date, each participant ﬁlled out a questionnaire about the other person. A second date is planned only if both people within the matched pair want to see each other again.

From the contingency table, we can see that we have total 63 pairs couple decide to go on a second date, with percentage of 22.83%.

# 2. Data Preprocessing
The original data set contains 276 instances with 22 features.

Make a new column in the data set and call it "second.date". Values in this column with 0 if there will be no second date, with 1 if there will be a second date.

2.1 Missing Values

After checking each rows' missing value we learn that most observation with missing value has 1 to 2 missing values. So we will keep them.   

2.2 Other Processing Steps

Replaced by numerical values： “PartnerYesM”, “PartnerYesF”, "AgeM","AgeF", "RaceM", "RaceF", "SincereM", "SincereF", "IntelligentM", "IntelligentF", "FunM", "FunF", "AmbitiousM, and "AmbitiousF".

Many of the numerical variables are on rating scales from 1 to 10, but the varibales of PartnerYesM, FunM, SharedInterestsM and SharedInterestsF have 0 value, so I diecided to replace all 0 of those varibles to 1, which is convinient for the following analysis.

2.3 Data Summary

After performing the above operation, we were left with 276 instances with 23 features for further analysis.

Lables: second.date (A second date is planned only if both people within the matched pair want to see each other again.)

Lable 1: Having second date. Total number of lable 1 instance is 63.

Lable 0: Having no second date. Total number of lable 0 instance is 213.

# 3. Data Visualization

3.1 ScatterPlot with all numerical variables

For variables of "like","Attractive","fun" and "SharedInterests", we can see that majority of the male and female who rate each other lower than 4 all end up not having second date.

For varialbe like "ambitious" and "Sincere", females with scores around 6 to 8 for each variable, will all end up having a second date.

Varible such as "Age" do not show much correlation with second date, which means that the factor may not affect much when they decide to plan a second date sor not.

Apperently, if a man or woman could get higher rating for each variables, except for age and race, he or she probably have a big chance for a second date. But, we are still not clear which variables may have major influence when it comes to second date. So, I will use logistic regression to help to find the critical variables and help people know more about their dating based on assumptions and hypothesis.

3.2 Mosaic Plot with Race

There are five race categoires:Asian, Black, Caucasian, Latino and Other. Both genders indicate that majority of the people will not go on a second date no matter what race they are.It shows that race is not a main factor that affecet the decision making when it comes to second date.

# 4. Logistic Regression Model

4.1 Logistic Regression Model

Firstly, I built logistic regression model with all numerical variables. Then I dropped variable of race, based on previous mosaic plot, we know that race doesn't affet much. After that, I dropped those p value more than 0.5, which is high, including variables of LikeF,AttractiveM,SincereM,SincereF,IntelligentM,IntelligentF and SharedInterestsM, and SharedInterestsF. Lastly, I dropped AgeF,FunM  and AmbitiousM for the same reason of high p value. So our final model was built by the remaining variables, including LikeM ,PartnerYesM,PartnerYesF,AttractiveF,FunF and AmbitiousF.

The final model:

P(have a second date|LikeM ,PartnerYesM,PartnerYesF,AttractiveF,FunF, AmbitiousF)=
  e^(-10.5161+0.4940(LikeM)+0.3416(PartnerYesM)+0.2693(PartnerYesF)+0.3486(FunF)- 0.3047(AmbitiousF))/
 (1+e^(-10.5161+0.4940(LikeM)+0.3416(PartnerYesM)+0.2693(PartnerYesF)+0.3486(FunF)- 0.3047(AmbitiousF)))



4.2 Log-Likelihood Test for Overall Model

H0: beta_1 = beta_2=...=0;

H1: at least one beta != 0

The p-value is 2.764745e-18, which is small enough to reject the null hypothesis.



4.3 Z-test

H0: the beta slope(variable) = 0;

Hα: the beta slope(variable) != 0

All the p-value is small enough to reject the null hypothesis, and these six variables are statistically significant.



4.4 AUC - ROC curve

AUC - ROC curve is a performance measurement for classification problem at various thresholds settings. ROC is a probability curve and AUC represents degree or measure of separability. It tells how much model is capable of distinguishing between classes. Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s.

For our model, we had AUC of 0.8724, which was good to distinguishing between classes, since majority of the data included.

# 4. Conclusion 
The logistic regression model provided low relative root mean squared error 1.281. With the including variables in the final model, the percentage of dates ended with both people wanting a second date is increasing to "24.31%", comparing to the Q1 with 22.83%. 
 
