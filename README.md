# Stock-Price-Prediction
In [1]:
#importing libraries
import math 
import numpy as np
import pandas_datareader as web
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense, LSTM
plt.style.use('fivethirtyeight')
/usr/local/lib/python3.6/dist-packages/pandas_datareader/compat/__init__.py:7: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
  from pandas.util.testing import assert_frame_equal
In [2]:
mar= web.DataReader('SBIN.NS', data_source = 'yahoo', start = '2019-03-01', end = '2020-08-10')
mar

plt.figure(figsize=(20,10))
plt.title("SBI Share Closing Price Prends")
plt.plot(mar['Close'])
plt.xlabel('Date', fontsize=20)
plt.ylabel('Closing Prices', fontsize=20)
plt.show()

In [3]:
df = mar['Close']
arr= df.to_numpy()
arr = arr.reshape(-1,1)
leng = len(arr)*0.8
test_len = math.ceil(leng)
test_len
Out[3]:
282
In [4]:
#Preprocessing
scaler= MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(arr)
data
Out[4]:
array([[0.55111719],
       [0.56691498],
       [0.57436248],
       [0.59083729],
       [0.58835483],
       [0.61611378],
       [0.61408258],
       [0.64229292],
       [0.63665085],
       [0.6659896 ],
       [0.66847219],
       [0.68697806],
       [0.6903633 ],
       [0.66440979],
       [0.64612955],
       [0.68900926],
       [0.71293157],
       [0.7589709 ],
       [0.7758971 ],
       [0.80410744],
       [0.76574138],
       [0.77160908],
       [0.75016922],
       [0.73098619],
       [0.73978787],
       [0.72241028],
       [0.74046489],
       [0.7420447 ],
       [0.7436245 ],
       [0.74430153],
       [0.72195888],
       [0.71857364],
       [0.69690816],
       [0.72173326],
       [0.70119618],
       [0.72963215],
       [0.71812239],
       [0.70999773],
       [0.71834801],
       [0.71270595],
       [0.69781094],
       [0.66531258],
       [0.67005186],
       [0.70954633],
       [0.70435566],
       [0.73933648],
       [0.72782672],
       [0.74430153],
       [0.76009931],
       [0.87497187],
       [0.84269913],
       [0.85872268],
       [0.86368773],
       [0.92304225],
       [0.95170398],
       [0.94425634],
       [0.89280072],
       [0.91491759],
       [0.91017831],
       [0.92349365],
       [0.90972692],
       [0.83976528],
       [0.86301057],
       [0.87316629],
       [0.88580461],
       [0.87181225],
       [0.88309639],
       [0.87090947],
       [0.84405331],
       [0.85398326],
       [0.84856696],
       [0.87700293],
       [0.89618596],
       [0.91333792],
       [0.92845856],
       [0.93568043],
       [0.95373505],
       [0.94967278],
       [0.95102683],
       [0.96434216],
       [0.97178966],
       [0.97743173],
       [0.99210111],
       [0.92281649],
       [0.94177389],
       [0.91785158],
       [0.95847447],
       [0.9602799 ],
       [0.94425634],
       [0.96366514],
       [1.        ],
       [0.96050553],
       [0.9259761 ],
       [0.90273081],
       [0.86368773],
       [0.8519522 ],
       [0.85962533],
       [0.86549316],
       [0.87090947],
       [0.79756259],
       [0.81855119],
       [0.75062061],
       [0.71135191],
       [0.67433989],
       [0.67953056],
       [0.62762354],
       [0.64770936],
       [0.63416839],
       [0.59805916],
       [0.62694652],
       [0.63213719],
       [0.61385695],
       [0.59963897],
       [0.57120287],
       [0.531257  ],
       [0.5427669 ],
       [0.58384118],
       [0.60866628],
       [0.60505527],
       [0.5581133 ],
       [0.55517945],
       [0.53057998],
       [0.56082152],
       [0.55269686],
       [0.55563084],
       [0.57391109],
       [0.60663508],
       [0.6147596 ],
       [0.6357482 ],
       [0.60415262],
       [0.55563084],
       [0.58474383],
       [0.5560821 ],
       [0.68088474],
       [0.73527422],
       [0.684947  ],
       [0.58406681],
       [0.59128868],
       [0.58835483],
       [0.54141272],
       [0.474385  ],
       [0.46626041],
       [0.44730308],
       [0.44346651],
       [0.49695334],
       [0.46626041],
       [0.46806591],
       [0.47212817],
       [0.4856692 ],
       [0.47483632],
       [0.51726478],
       [0.53622205],
       [0.54005868],
       [0.56240132],
       [0.50394945],
       [0.5899345 ],
       [0.58587224],
       [0.62762354],
       [0.72918076],
       [0.73437143],
       [0.73775667],
       [0.75987369],
       [0.75242605],
       [0.75513427],
       [0.74542994],
       [0.75558566],
       [0.70390427],
       [0.7002934 ],
       [0.77206047],
       [0.78650422],
       [0.81042653],
       [0.80320466],
       [0.81471456],
       [0.80546148],
       [0.83615441],
       [0.83344619],
       [0.86978105],
       [0.89573456],
       [0.86210792],
       [0.84698716],
       [0.83683143],
       [0.86210792],
       [0.83660581],
       [0.76348456],
       [0.74858955],
       [0.73369441],
       [0.73098619],
       [0.77183484],
       [0.82013086],
       [0.81697138],
       [0.82171066],
       [0.79485451],
       [0.80027081],
       [0.84405331],
       [0.81945384],
       [0.81516595],
       [0.80794408],
       [0.84134509],
       [0.82848115],
       [0.8255473 ],
       [0.82870691],
       [0.85059802],
       [0.82532167],
       [0.7589709 ],
       [0.75626268],
       [0.76258177],
       [0.80952388],
       [0.81877682],
       [0.81200634],
       [0.79959379],
       [0.78266758],
       [0.77837956],
       [0.75445725],
       [0.73640263],
       [0.73504859],
       [0.74610696],
       [0.7779283 ],
       [0.7817648 ],
       [0.74633272],
       [0.74136768],
       [0.74746114],
       [0.72150763],
       [0.75648845],
       [0.66463556],
       [0.70142181],
       [0.72286167],
       [0.77228624],
       [0.76596701],
       [0.75671407],
       [0.78311898],
       [0.76438734],
       [0.79711133],
       [0.76077634],
       [0.73730542],
       [0.75242605],
       [0.76506436],
       [0.79801398],
       [0.77679989],
       [0.79417735],
       [0.80049657],
       [0.77228624],
       [0.68675244],
       [0.61633941],
       [0.62739791],
       [0.6068607 ],
       [0.62130445],
       [0.54005868],
       [0.46310086],
       [0.42541189],
       [0.27871814],
       [0.41141954],
       [0.32723992],
       [0.2902279 ],
       [0.2904536 ],
       [0.23832088],
       [0.26630559],
       [0.13879486],
       [0.14601667],
       [0.17625814],
       [0.18912208],
       [0.20356576],
       [0.16271717],
       [0.20762809],
       [0.16113744],
       [0.11126154],
       [0.16046035],
       [0.14511395],
       [0.16655381],
       [0.14737078],
       [0.1421801 ],
       [0.16993905],
       [0.1913789 ],
       [0.18799366],
       [0.15301284],
       [0.17084176],
       [0.16181446],
       [0.13044457],
       [0.13563525],
       [0.15098171],
       [0.17716092],
       [0.17896636],
       [0.12638231],
       [0.08824188],
       [0.09140149],
       [0.08982169],
       [0.07131568],
       [0.06589931],
       [0.07244409],
       [0.10494246],
       [0.07718344],
       [0.07018727],
       [0.02008575],
       [0.00880161],
       [0.01150976],
       [0.00496498],
       [0.        ],
       [0.00248245],
       [0.03498082],
       [0.03317532],
       [0.04716767],
       [0.08666214],
       [0.08756486],
       [0.10855332],
       [0.10471676],
       [0.1667795 ],
       [0.16226585],
       [0.15165873],
       [0.16632811],
       [0.11870905],
       [0.12773635],
       [0.10313695],
       [0.09952602],
       [0.09975171],
       [0.12999318],
       [0.15188443],
       [0.16632811],
       [0.18776797],
       [0.15233582],
       [0.15526967],
       [0.15233582],
       [0.12818775],
       [0.12457681],
       [0.15323854],
       [0.15617239],
       [0.15278715],
       [0.16790792],
       [0.17039044],
       [0.18528544],
       [0.21778381],
       [0.20198602],
       [0.18889638],
       [0.15888061],
       [0.14872489],
       [0.15978333],
       [0.16858494],
       [0.18009477],
       [0.19656958],
       [0.18573684],
       [0.21394717],
       [0.18551114],
       [0.16384559],
       [0.17422701],
       [0.1821259 ],
       [0.16113744],
       [0.18325431],
       [0.18686525],
       [0.1839314 ],
       [0.18325431],
       [0.18099749],
       [0.17964338],
       [0.19386143],
       [0.1995035 ]])
In [5]:
#Create training data set
train_data = data[0:test_len, : ]
#Rest of the data will be used to back test
#splitting the data for training

xtrain = []
ytrain=[]

for i in range(60,len(train_data)):
  xtrain.append(train_data[i-60:i , 0])
  ytrain.append(train_data[i,0])

#Creating numpy arrays for LSTM model
xtrain , ytrain = np.array(xtrain), np.array(ytrain)

xtrain = xtrain.reshape(xtrain.shape[0],xtrain.shape[1],1)

xtrain.shape
Out[5]:
(222, 60, 1)
In [6]:
#Building LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences= True, input_shape=(xtrain.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
In [7]:
#Compiling
model.compile(optimizer='adam', loss= 'mean_squared_error')
In [8]:
#Training data
model.fit(xtrain, ytrain, batch_size=1, epochs= 1)
222/222 [==============================] - 4s 20ms/step - loss: 0.0182
Out[8]:
<tensorflow.python.keras.callbacks.History at 0x7ff7719847b8>
In [9]:
#Creating test data set
testdata= data[test_len - 60: , : ]
#Creating test datasets
xtest = []
ytest = data[test_len: , :]
for i in range(60, len(testdata)):
  xtest.append(testdata[i-60:i,0])
In [10]:
#Converting to numpy arrays
xtest = np.array(xtest)
xtest.shape
xtest = xtest.reshape(xtest.shape[0], xtest.shape[1],1)
xtest.shape
Out[10]:
(70, 60, 1)
In [11]:
#Get the predicted values
fprice = model.predict(xtest)
fprice = scaler.inverse_transform(fprice)
In [12]:
#Model testing
rmse = np.sqrt(np.mean((fprice - ytest)**2))
rmse
Out[12]:
187.50075364696124
In [13]:
#Plotting 
train = mar[:test_len]
valid = mar[test_len:]
valid['Predictions'] = fprice

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Price', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close' , 'Predictions']])
plt.legend(['Train' , 'Val' , 'Predictions'], loc = 'lower right')
plt.show()
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:4: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  after removing the cwd from sys.path.
