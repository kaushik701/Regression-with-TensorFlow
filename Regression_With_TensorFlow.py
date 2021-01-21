#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from utils import *
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping,LambdaCallback

%matplotlib inline
print('Libraries Imported')
# %%
df = pd.read_csv('data.csv',names=column_names)
df.head()
df.isna().sum()
# %%
df = df.iloc[:,1:]
df_norm = (df-df.mean())/df.std()
df_norm.head()
# %%
y_mean = df['price'].mean()
y_std = df['price'].std()
def convert_label_value(pred):
    return int(pred*y_std + y_mean)
print(convert_label_value(0.350088))
# %%
X = df_norm.iloc[:,:6]
X.head()
Y = df_norm.iloc[:,-1]
Y.head()
# %%
X_arr = X.values
Y_arr = Y.values
print(X_arr.shape)
print(Y_arr.shape)
# %%
X_train,X_test,y_train,y_test = train_test_split(X_arr,Y_arr,test_size=0.05,shuffle=True,random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
#%%
model = Sequential([
    Dense(10,input_shape=(6,),activation='relu'),
    Dense(20,activation='relu'),
    Dense(5,activation='relu'),
    Dense(1)
])
model.compile(loss='mse',optimizer='adadelta')
model.summary()
# %%
early_stopping = EarlyStopping(monitor='val_loss',patience=5)
preds_on_untrained = model.predict(X_test)
history = model.fit(
    X_train,y_train,validation_data=(X_test,y_test),epochs=1000,callbacks=[early_stopping]
)
# %%
plot_loss(history)
preds_on_trained = model.predict(X_test)
compare_predictions(preds_on_untrained,preds_on_trained,y_test)
# %%
price_on_untrained = [convert_label_value(y) for y in preds_on_untrained]
price_on_trained = [convert_label_value(y) for y in preds_on_trained]
price_y_test = [convert_label_value(y) for y in y_test]
compare_predictions(price_on_untrained,price_on_trained,price_y_test)