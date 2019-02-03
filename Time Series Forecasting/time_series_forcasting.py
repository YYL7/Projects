# import packages
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

# load the dataset
data_matrix = pd.read_csv('GlobalTemperatures.csv')[['dt','LandAverageTemperature']]
data_matrix.columns = ['Date','Temp']
data_matrix['Date'] = pd.to_datetime(data_matrix['Date'])
data_matrix.reset_index(drop=True, inplace=True)
data_matrix.set_index('Date', inplace=True)
print(data_matrix.shape)
data_matrix.head()
data_matrix.tail()

# mean value
data_matrix.mean()

# replace missing value with mean value
data_matrix.fillna(data_matrix.mean(), inplace=True)

# check missing value 
data_matrix.isnull().sum()

# Data Visiualization
# Create a time series plot.
plt.figure(figsize = (15, 5))
plt.plot(data_matrix)
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.title('Temperature Variation')
#plt.legend()
plt.show()

# second way to plot - using seaborn
#plt.figure(figsize=(35,10))
#sns.lineplot(x=data.index, y=data['Temp'])
#plt.title('Temperature Variation')
#plt.show()

# Yearly Aerage Temperatures - calculate mean by year 
data1=data_matrix.copy()
data1['year'] = data1.index.year

year_avg = pd.pivot_table(data1, values='Temp', index='year', aggfunc='mean')
year_avg['10 Years MA'] = year_avg['Temp'].rolling(5).mean()
year_avg[['Temp','5 Years MA']].plot(figsize=(20,6))
plt.title('Yearly Aerage Temperatures')
plt.xlabel('Months')
plt.ylabel('Temperature')
plt.xticks()
plt.show()

# create data and reshape it
data_np = data_matrix.transpose().as_matrix()
shape = data_matrix.shape
data_np = data_np.reshape((shape[0] * shape[1], 1))

# Split the whole data into train(70%) and test data (30%)
dates = pd.date_range(start='1750-01', freq='MS', periods=len(data_matrix.index))
dates

data = pd.DataFrame({'Temp': data_np[:,0]})
data.set_index(dates, inplace=True)
 
train, test = data_matrix[0:int(len(data.index)*0.7)], data_matrix[-int(len(data.index)*0.3):]
print("Number of entries (training set, test set): " + str((len(train), len(test))))
# Number of entries (training set, test set): (2226, 954)

# Scale the data - (0 to 1)
scaler = MinMaxScaler(feature_range=(0, 1))

train_data = scaler.fit_transform(train)
test_data = scaler.fit_transform(test)

# compare
train_data[:5]
scaler.inverse_transform(train_data[:5])

# shape of the train/test data
print("Shape of train data: " + str(train_data.shape))
print("Shape of test data: " + str(test_data.shape))

# Create dataset
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []

    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    
    return np.array(dataX), np.array(dataY)
    
look_back = 60 # predict the value by looking back 60 months(5 years)
trainX, trainY = create_dataset(train_data, look_back)
testX, testY = create_dataset(test_data, look_back)

print("Shape of train input: " + str(trainX.shape))
print("Shape of train labels: " + str(trainY.shape))
print("Shape of test input: " + str(testX.shape))
print("Shape of test labels: " + str(testY.shape))

# reshape data
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

print("Shape of train input: " + str(trainX.shape))
print("Shape of train labels: " + str(trainY.shape))
print("Shape of test input: " + str(testX.shape))
print("Shape of test labels: " + str(testY.shape))

# LSTM Model
def fit_model(trainX, trainY, look_back = 1):
    model = Sequential()
    
    model.add(LSTM(4, input_shape = (1, look_back)))
    model.add(Dense(1))
    model.compile(loss = "mean_squared_error", 
                  optimizer = "adam")
    model.fit(trainX, 
              trainY, 
              epochs = 100, 
              batch_size = 1, 
              verbose = 2)
    
    return(model)

# Fit the first model.
model1 = fit_model(trainX, trainY, look_back)

# predicted value
trainPredict = model1.predict(trainX)
testPredict = model1.predict(testX)

testPredict[:5]

# Inverse transform the data
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Calculate Root Mean Square Error(RMSE) for train and test predictions
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# training
# Generate Dates for index
dates = pd.date_range(start='1751-01', freq='MS', periods=trainY.shape[1])

# Create Dataframes for actual values and predicted values of train data
trainActual = pd.DataFrame({'Temp': trainY[0]})
trainActual.index = dates

trainPredictdf = pd.DataFrame({'Temp': trainPredict[:,0]})
trainPredictdf.index = dates

# plot - training
plt.figure(figsize=(20,5))
plt.plot(trainActual, color='blue', label='Actual Temperature')
plt.plot(trainPredictdf, color='red', label='Predicted Temperature')
plt.title('Train data: Temperature')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.legend(loc='best')

# testing
# Generate Dates for index
testDates = pd.date_range(start=trainActual.index[-1]+60, freq='MS', periods=test.shape[0] - look_back - 1)
testDates

testActual = pd.DataFrame({'Temp': testY[0]})
testActual.index = testDates

testPredictdf = pd.DataFrame({'Temp': testPredict[:,0]})
testPredictdf.index = testDates

plt.figure(figsize=(20,5))
plt.plot(testActual, color='orange', label='Actual Temperature')
plt.plot(testPredictdf, color='red', label='Predicted Temperature')
plt.title('Test data: Temperature')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.legend(loc='best')
