#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd
from sklearn.neural_network import MLPRegressor


# In[2]:


#reading
df=pd.read_csv("dabeerinput.csv")


# In[3]:


df


# In[4]:


X_train = df.iloc[:,1:-1]
print(X_train.head(5))
Y_train = df.iloc[:,-1:]
print(Y_train.head(5))


# In[5]:


print(type(X_train))
print(type(Y_train))


# In[6]:


import time
model_fit_time = time.time()
#model = LinearRegression()
#model = DecisionTreeRegressor(random_state=0)
model=MLPRegressor(hidden_layer_sizes=(300), activation='relu', solver='adam', max_iter=200, learning_rate = 'adaptive', random_state=None, shuffle=False, verbose=False)
trained_model = model.fit(X_train,Y_train)
print("Model: ",model)
print("--- Total time for model_fit_time: %s seconds ---" % (time.time() - model_fit_time))


# In[43]:


#creating list one each for one column
for i in df.columns[1:-1]:
    locals()["list_"+str(i)]=df[i].to_list()


# In[42]:


df


# In[44]:


#predicting feature using ARIMA
from statsmodels.tsa.arima.model import ARIMA

df_X2 = pd.DataFrame()
for i in df.columns[1:-1]:
    # fit model
    locals()["new_list_"+str(i)]=[]
    locals()["new_list_"+str(i)].extend(locals()["list_"+str(i)])

    for iter in range(0, len(locals()["list_"+str(i)]), 1):
        arima_f1 = ARIMA(locals()["new_list_"+str(i)], order=(1, 1, 1))
        arima_f1_fit = arima_f1.fit()
        # make prediction
        print(len(locals()["new_list_"+str(i)]))
        yhat = arima_f1_fit.predict(len(locals()["new_list_"+str(i)]), len(locals()["new_list_"+str(i)]))
#         print(yhat)
#         print(type(yhat))
#         print(len(yhat))
        locals()["new_list_"+str(i)].extend(yhat)

    print(locals()["new_list_"+str(i)])
    #generating X dataframe with train and test
    df_X2[str(i)] = locals()["new_list_"+str(i)]

df_X2.to_csv('trainX_testX.csv', index=False)


# In[53]:


X_test = df_X2.loc[8:, :]
X_test.reset_index(inplace=True, drop=True)
X_test


# In[54]:


predicted_Y=trained_model.predict(X_test)


# In[55]:


predicted_Y


# In[61]:


testY_predictions=pd.Series(predicted_Y)

testY_predictions.to_csv('testY_predictions.csv', index=False)


# In[ ]:





# In[ ]:




