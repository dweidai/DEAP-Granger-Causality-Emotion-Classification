#!/usr/bin/env python
# coding: utf-8

# # Suppress Printing

# In[1]:


import sys, os
old = sys.stdout
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = old


# # Load Back Projected ICA Data

# In[2]:


import scipy.io as sio


# In[18]:


enablePrint()
filepath = '~/ICA_Data/EEGData01.mat'
mat_contents = sio.loadmat(filepath)
ica = mat_contents['data']
temp_trial = ica[:,:,1]
trial = ica.shape[2]
print("{}: {}".format("Total number of trials is", trial))
channel = temp_trial.shape[0]
print("{}: {}".format("Total number of channels in each trial is", channel))
timepoint = temp_trial.shape[1]
print("{}: {}".format("Total number of time points in per channel per trial is", timepoint))


# # Testing between 1 trial 2 channels with GC

# In[19]:


import statsmodels.tsa.stattools as stm
import numpy as np


# In[20]:


#we just gonna pick temp_trial
print(temp_trial.shape)
hz = 128
#a 3 second pre-trial baseline removed
temp_trial = temp_trial[:,128*3:]
print(temp_trial.shape)


# In[21]:


a = np.asarray(temp_trial[0,:])
b = np.asarray(temp_trial[1,:])
x = np.vstack((a, b)).T
print(x.shape)


# In[22]:


from statsmodels.tsa.ar_model import AR
model = AR(a)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
maxlag = model_fit.k_ar


# In[23]:


addconst = True
verbose = True


# In[24]:


result = stm.grangercausalitytests(x, maxlag, addconst, verbose)
optimal_lag = -1
F_test = -1.0
for key in result.keys():
    _F_test_ = result[key][0]['params_ftest'][0]
    if _F_test_ > F_test:
        F_test = _F_test_
        optimal_lag = key


# In[25]:


print("{} {}".format("We are going to look into the GC with Optimal Lag of", optimal_lag))


# We consider the p-value of the test as a measure for Granger causality: rejection of ℋ0 (p < 0.03) signifies Granger causality, acceptance means non-causality.

# The causality relations drawn from systems with very small values of |det(ΛˆI)| are not meaningful

# In[26]:


if (result[optimal_lag][0]['params_ftest'][1] < 0.03):
    print(result[optimal_lag][0]['params_ftest'][0])


# # Compute one Multivariant Granger Causality Matrix MGCM

# In[27]:


from matplotlib import pyplot
import math
import time
print("testing")


# In[28]:
'''

temp_trial = temp_trial[:,1::8] #TODO
time_start = time.clock()
MGCM = np.zeros((channel,channel))
for i in range(channel):
    for j in range(channel):
        if i == j:
            blockPrint()
            print("{}:{}".format(i,j))
            MGCM[i,j] = 0
        blockPrint()
        x = np.vstack((np.asarray(temp_trial[i,:]), np.asarray(temp_trial[j,:]))).T
        #model = AR(a)
        #model_fit = model.fit()
        #maxlag = model_fit.k_ar
        #if maxlag > 5:
        maxlag = 3
        result = stm.grangercausalitytests(x, maxlag, addconst = True, verbose = True)
        optimal_lag = 2
        F_test = -1.0
        for key in result.keys():
            _F_test_ = result[key][0]['params_ftest'][0]
            if _F_test_ > F_test:
                F_test = _F_test_
                optimal_lag = key
        enablePrint()
        #print(optimal_lag)
        blockPrint()
        if (result[optimal_lag][0]['params_ftest'][1] < 0.03):
            MGCM[i,j] = math.log(result[optimal_lag][0]['params_ftest'][0])
        else:
            MGCM[i,j] = 0
blockPrint()
diag = np.max(MGCM)
for i in range(channel):
    for j in range(channel):
        if i == j:
            MGCM[i,j] = 1
        else:
            MGCM[i,j] = MGCM[i,j]/diag


# In[29]:


enablePrint()
pyplot.matshow(MGCM)
pyplot.show()
pyplot.imsave("test.png", MGCM)
time_elapsed = (time.clock() - time_start)
print("{}: {}".format("Time Used", time_elapsed))


# In[30]:


print(np.mean(MGCM))
print(np.median(MGCM))
print(np.max(MGCM))
print(np.min(MGCM))


# In[31]:


print("{}:{}".format("Hours needed", time_elapsed*40*32/60/60))


# In[80]:


#GoogleColab takes 34 seconds
#Hours needed:33.8410016
#Hours needed:2.871406577777816 no optimal maxlag =5
#Hours needed:4.871406577777816 no optimal maxlag =5
'''

# # Split data computing MGCM for LSTM

# In[47]:


temp_trial = ica[:,:,1]
trial = ica.shape[2]
print("{}: {}".format("Total number of trials is", trial))
channel = temp_trial.shape[0]
print("{}: {}".format("Total number of channels in each trial is", channel))
timepoint = temp_trial.shape[1]
print("{}: {}".format("Total number of time points in per channel per trial is", timepoint))


# In[48]:


from matplotlib import pyplot
import math
import time


# In[49]:


hz = 128
#a 3 second pre-trial baseline removed
lstm_trial = temp_trial[:,128*3:]
print(temp_trial.shape)
print(lstm_trial.shape)


# In[50]:


current_trial = lstm_trial[:,0*hz:(0+1)*hz]
print(current_trial.shape)


# In[51]:


from statsmodels.tsa.ar_model import AR
a = np.asarray(current_trial[0,:])
b = np.asarray(current_trial[17,:])
x = np.vstack((a, b)).T
print(x.shape)

model = AR(a)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
maxlag = model_fit.k_ar
if maxlag > 3:
    maxlag =3


# In[52]:


blockPrint()
result = stm.grangercausalitytests(x, maxlag, addconst, verbose)
optimal_lag = -1
F_test = -1.0
for key in result.keys():
    _F_test_ = result[key][0]['params_ftest'][0]
    if _F_test_ > F_test:
        F_test = _F_test_
        optimal_lag = key
enablePrint()
print("{} {}".format("We are going to look into the GC with Optimal Lag of", optimal_lag))


# Here split the dataset second by second and compute the MGCM for each 128 datapoints. Since 7680/128 = 60 seconds, the sanity check is good

# In[53]:


temp_list = []
MGCM = np.zeros((channel,channel))
#load for lstm


# In[54]:


time_start = time.clock()
print("Start")
for k in range(60):
    enablePrint()
    print(k)
    blockPrint()
    current_trial = lstm_trial[:, k*hz:(k+1)*hz]
    for i in range(channel):
        for j in range(channel):
            if i == j:
                print("{} -> {}:{}".format(k,i,j))
                MGCM[i,j] = 1
            blockPrint()
            x = np.vstack((np.asarray(current_trial[i,:]), np.asarray(current_trial[j,:]))).T
            #model = AR(a)
            #model_fit = model.fit()
            maxlag = 3 #model_fit.k_ar
            #if maxlag > 3:
            #    maxlag = 3
            result = stm.grangercausalitytests(x, maxlag, addconst = True, verbose = True)
            optimal_lag = -1
            F_test = -1.0
            for key in result.keys():
                _F_test_ = result[key][0]['params_ftest'][0]
                if _F_test_ > F_test:
                    F_test = _F_test_
                    optimal_lag = key
            if (result[optimal_lag][0]['params_ftest'][1] < 0.03):
                MGCM[i,j] = math.log(result[optimal_lag][0]['params_ftest'][0])
            else:
                MGCM[i,j] = 0
    enablePrint()
    diag = np.max(MGCM)
    #print(diag)
    for i in range(channel):
        for j in range(channel):
            if i == j:
                MGCM[i,j] = 1
            else:
                MGCM[i,j] = MGCM[i,j]/diag
    enablePrint()
    temp_list.append(MGCM)
    #pyplot.matshow(MGCM)
    #pyplot.show()
time_elapsed = (time.clock() - time_start)
print("{}: {}".format("Time Used", time_elapsed))


# In[55]:


print("{} {}".format("Hours needed", time_elapsed*40*32/60/60))


# In[56]:


print(len(temp_list)) #should be 60


# In[57]:


#Original one need 393 hours
#changed to 172 hours
#141 hours current
#137 hours without printing


# # Data Generating for CNN

# In[1]:


import sys, os
old = sys.stdout
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = old


# In[2]:


import scipy.io as sio
import statsmodels.tsa.stattools as stm
import numpy as np
from statsmodels.tsa.ar_model import AR
from matplotlib import pyplot
import math
import time


# In[3]:


filename = []
directory = '/Users/apple/Desktop/eeglab14_1_2b/ICA_Data/'
for file in os.listdir(directory):
    if file.endswith(".mat"):
        filename.append(file)
filename.sort()
filename = filename[31::]
print(filename)


# In[4]:


result =np.zeros((32,32))
blockPrint()
print(type(result))


# In[5]:


for eegdata in filename:
    filepath = directory + eegdata
    mat_contents = sio.loadmat(filepath)
    ica = mat_contents['data']
    temp_trial = ica[:,:,1]
    trial = ica.shape[2]
    print("{}: {}".format("Total number of trials is", trial))
    channel = temp_trial.shape[0]
    print("{}: {}".format("Total number of channels in each trial is", channel))
    timepoint = temp_trial.shape[1]
    print("{}: {}".format("Total number of time points in per channel per trial is", timepoint))
    for k in range(trial):
        hz = 128
        #a 3 second pre-trial baseline removed
        temp_trial = ica[:,:,k]
        temp_trial = temp_trial[:,hz*3:]
        MGCM = np.zeros((channel,channel))
        for i in range(channel):
            for j in range(channel):
                if i == j:
                    enablePrint()
                    MGCM[i,j] = 0
                else:
                    blockPrint()
                x = np.vstack((np.asarray(temp_trial[i,:]), np.asarray(temp_trial[j,:]))).T
                maxlag = 3
                result = stm.grangercausalitytests(x, maxlag, addconst = True, verbose = True)
                optimal_lag = 2
                F_test = -1.0
                for key in result.keys():
                    _F_test_ = result[key][0]['params_ftest'][0]
                    if _F_test_ > F_test:
                        F_test = _F_test_
                        optimal_lag = key
                if (result[optimal_lag][0]['params_ftest'][1] < 0.03):
                    MGCM[i,j] = math.log(result[optimal_lag][0]['params_ftest'][0])
                else:
                    MGCM[i,j] = 0
        diag = np.max(MGCM)
        for i in range(channel):
            for j in range(channel):
                if i == j:
                    MGCM[i,j] = 1
                else:
                    MGCM[i,j] = MGCM[i,j]/diag
        pyplot.matshow(MGCM)
        pyplot.show()
        np.append(result,MGCM)
        eegdata = os.path.splitext(eegdata)[0]
        imgname = "img/{0}_{1}.png".format(eegdata, str(k).zfill(2))
        pyplot.imsave(imgname, MGCM)


# In[51]:


#print(type(MGCM))

#np.append(result,MGCM)


# In[ ]:




