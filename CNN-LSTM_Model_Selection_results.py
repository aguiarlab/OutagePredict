#@title Grid search 5-fold C
#Grid Search 5-fold cross validation via sklearn
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from keras.layers import Dropout
from pandas import DataFrame
from numpy import concatenate
from scipy.stats import boxcox

# split a multivariate sequence into samples - Reference: Brownee's ML-Mystery
def split_sequences(sequences, n_steps):
  X, y = list(), list()
  for i in range(len(sequences)):
      # find the end of this pattern
      end_ix = i + n_steps
      # check if we are beyond the dataset
      if end_ix > len(sequences):
          break
      # gather input and output parts of the pattern
      seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
      X.append(seq_x)
      y.append(seq_y)
  return np.array(X), np.array(y)

# Normalize input variables: train and test series
def scaleNormInput(X_train):
  # Reshape to 2D
  X_train = X_train.astype('float32')
  dwX_train = X_train.reshape((X_train.shape[0],X_train.shape[1] * X_train.shape[2]))
  dwX_train = pd.DataFrame(dwX_train)
  scaler = MinMaxScaler()
  # Fit train set
  scaled1 = scaler.fit_transform(dwX_train)
  train_X = DataFrame(scaled1)
  train_X = train_X.values
  # Reshape to 3D
  train = train_X.reshape((train_X.shape[0],n_steps,n_features))
  return(train)

# Normalize output variable: y_train only
def scaleNormOutput(y_train):
  y_train = y_train.astype('float32')
  dwY_train = pd.DataFrame(y_train)
  scaler = MinMaxScaler()
  # Fit train set
  scaled3 = scaler.fit_transform(dwY_train)
  train_y = DataFrame(scaled3)
  train_y = train_y.values
  # Reshape to 3D
  train = train_y.reshape((train_y.shape[0],1))
  return(train)

def Rolling_Percentile(mydata1):
  index_steps = 36
  # convert into input/output
  X1, y1 = split_sequences(mydata1, index_steps)
  #n_features = X.shape[2]
  # try some rolling hr mean
  dwAvg = 24
  ##recodeVal = 25
  y1 = pd.DataFrame(y1)
  y1 = y1.rolling(dwAvg).mean()
  # replace NAN with 0's
  y1 = np.array(y1)
  y1 = np.nan_to_num(y1)  
  return X1,y1

# Function to create model, required for KerasRegressor
def create_model(optimizer='rmsprop', init='glorot_uniform',dropout_rate=0.1,loss='huber'):  
  from keras.layers import Input
  from keras.models import Model
  from keras.layers import Dense, Lambda, Flatten, Activation, Concatenate, CuDNNLSTM
  ##from keras.layers.recurrent import LSTM
  from keras.layers import Conv1D
  from keras.layers import MaxPooling1D
  from keras.layers import Dropout
  from matplotlib import pyplot
  from matplotlib import pyplot
  import tensorflow as tf

  tf.keras.backend.clear_session()
  tf.random.set_seed(1)
  np.random.seed(1)

  inputs1=Input(shape=(n_steps,n_features))
  x = Conv1D(filters=32,kernel_size=2,padding='causal',activation="relu")(inputs1)
  x = Conv1D(filters=32,kernel_size=2,padding='causal',activation="relu")(x)
  x = Conv1D(filters=32,kernel_size=2,padding='causal',activation="relu")(x)
  x = Conv1D(filters=32,kernel_size=2,padding='causal',activation="relu")(x)
  x = MaxPooling1D(pool_size=2)(x)
  x = Dropout(dropout_rate)(x)
  x = Flatten()(x)
  ##ls = LSTM(50, kernel_initializer=init, activation='relu', dropout=dropout_rate)(inputs1)
  ls = CuDNNLSTM(50,kernel_initializer=init)(inputs1)
  ls = Activation('relu')(ls)
  ls = Dropout(dropout_rate)(ls)
  merged = Concatenate()([x, ls])
  merged = Dense(10,activation='linear',)(merged)
  merged = Dense(1,activation='linear')(merged)
  merged = Lambda(lambda x: x * 10)(merged)
  model=Model(inputs1,merged)
  model.compile(loss=loss,optimizer=optimizer)
  return model

#Load data
mydata = pd.read_csv('nassauC.csv',header=0)
dataIn = mydata[mydata.columns[1:71]]
# Merged Differenced Columns
dateT = mydata[mydata.columns[0]]
dateT = DataFrame(dateT)
dateT['dateTime'] = dateT
dateT.drop('Unnamed: 0', axis = 1, inplace = True)
df_data = pd.concat((dateT, dataIn), axis=1)
proData = df_data
# drop first datetime column
proData = proData.drop(proData.columns[0], axis = 1)
proData = proData.astype('float32')
proData = proData.interpolate(method ='linear', limit_direction ='both', limit = 1500, axis=0)
##print(proData.head())

box_power =proData['Total Outages']
# power transform
transformed, lmbda = boxcox(box_power)
proData['Total Outages'] = transformed
finData = proData
print(finData.head())

# move target variable to the last column to confirm with surpervise learning function
colNewn = finData[finData.columns[5]]
labelNewn = finData.columns[5]
finData.drop(finData.columns[5], axis=1, inplace =True)
finData.insert(6,labelNewn, colNewn)
print(finData.head())

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
n_steps = 36
n_features = 6

finData = finData.values

X = finData[:,:-1] # Input
y = finData[:,-1] # Output

# split into train and test sets
train_size = int(len(finData) * 0.80)
test_size = len(finData) - train_size
train_x, test_x = X[0:train_size,:], X[train_size:len(X),:]
train_y, test_y = y[0:train_size], y[train_size:len(y)]

y_train = train_y.reshape((train_y.shape[0],1))
X_train = concatenate((train_x,y_train), axis=1)
y_test = test_y.reshape((test_y.shape[0],1))
X_test = concatenate((test_x,y_test), axis=1)


train_val_size = int(len(X_train) * 0.1)
train_val = X_train[-train_val_size:]
g = len(train_val)
gt = 36 + g
train_val = X_train[-gt:]
train_data = train_val

# split into input (X) and output (Y) variables
X , y = Rolling_Percentile(train_data)
print(X.shape, y.shape)
trainFold,train_y_Fold = X, y
X_tr = scaleNormInput(trainFold)
y_tr = scaleNormOutput(train_y_Fold)
print(X_tr.shape, y_tr.shape)

# create model
model = KerasRegressor(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
# evaluate using 10-fold cross validation
kfold = TimeSeriesSplit(n_splits=5)

# grid search epochs, batch size and optimizer
losses = ['mse', 'huber', 'mae']
optimizers = ['rmsprop','adam','sgd']
#init = ['glorot_uniform', 'normal', 'uniform']
init = ['glorot_uniform','normal']
epochs = [20,60,100]
batches = [16,36,64,256]
dropout_rate = [0.1,0.5]
param_grid = dict(optimizer=optimizers,epochs=epochs,batch_size=batches,init=init,
                  dropout_rate=dropout_rate,loss = losses)
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=kfold,n_jobs=1)
grid_result = grid.fit(X_tr, y_tr)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
scores = []
for mean, stdev, param in zip(means, stds, params):
  scores.append(means)
  print("%f (%f) with: %r" % (mean, stdev, param))
