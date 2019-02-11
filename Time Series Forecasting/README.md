# Time Series Forecasting - LSTM

# Introduction

Time Series is a sequence of well-defined data points measured at consistent time intervals over a period of time.

Time Series Forecasting is the use of a model to predict future values based on previously observed values.

LSTM (Long short-term memory), a special kind of RNN, capable of learning long-term dependencies. We can use Keras pakage in python to do Time Series Prediction with LSTM Recurrent Neural Networks.

The goal is to explore and predict the global land average temperature by time series forecasting with LSTM model.

# Data Summary
Used dataset from the Berkeley Earth Surface Temperature Study, combining temperature reports by each month since the year of 1750. Originally, we have 3,129 instances with missing value, which later were replaced by the mean value of 8.37. 

From the time series plot among the actual values, we can see that the plot flucuate and then become stable. It seemed that the world get warmer cause we barely have any extreme temperatures but stable temperatures. 

From the plot of Yearly Average Temperatures. we can also see that the yearly average temperatures flucuate by periods and then actually increase by years. 

# Data Preprocessing 
Before we difine the LSTM model, we should prepare the dataset for the time series LSTM model.

Firstly, I splitted the data into training data with 70% of the data, and testing data with 30% of the data;

Secondly, I scaled the average temperature from 0 to 1, which should be inversed back after building the modle cause we need to calculate the RMSE;

Lastly, I reshaped the training data and testing data by size of looking back, from which we could predict the current temperature by looking back how many periods we difined. And for this dataset, I decided to set the size of looking back as 60 months, or 5 years.

# Algorithm
After prepared the dataset, I used Keras in Python to build one-layer LSTM model with epco of 100. 

# Result
LSTM model provided low relative root mean-squared error 1.43 for the testing data, which means the model could give good prediction, enabling forecasting based on historical imagery.
