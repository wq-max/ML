import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import pywt
import datetime
import time
import random
#################################################################################################################################
data = pd.read_csv('d:/data/jq_data.XSGE_1d.csv')
data.replace(0,np.NAN,inplace=True)
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)
ma1=data['close'].rolling(5).mean()
ma=data['close'].rolling(30).mean()
wave_close=(ma1-ma)/ma1*1000
Fs=1.0/60 # 采样频率：最高可识别频率0.5*(1/60)
L=90
time=L

# T0=datetime.datetime.now()
# 采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
x = np.linspace(0, time, L)
T=datetime.datetime.now().time().microsecond
print(T)
# 设置需要采样的信号，频率分量有180，390和600
# y = 7 * np.sin(2 * np.pi * 180 * x) + 2.8 * np.sin(2 * np.pi * 390 * x) + 5.1 * np.sin(2 * np.pi * 600 * x)
t0=datetime.datetime.now()
count=0
for i in range(L,len(data),1000):
    y=np.array(wave_close.iloc[i-L:i])
    cA5,cD5,cD4,cD3,cD2,cD1=pywt.wavedec(y,wavelet='db4',mode='antireflect',level=5)
    count=count+1
t1=datetime.datetime.now()
print('count, wavede time: ',count, t1-t0)

ycD1= (cD1) # 取绝对值
xcD1 = np.arange(len(cD1))  # 频率
ycD2= (cD2) # 取绝对值
xcD2 = np.arange(len(cD2))  # 频率
ycD3= (cD3) # 取绝对值
xcD3 = np.arange(len(cD3))  # 频率
ycD4= (cD4) # 取绝对值
xcD4 = np.arange(len(cD4))  # 频率
ycD5= (cD5) # 取绝对值
xcD5 = np.arange(len(cD5))  # 频率
ycA5= (cA5) # 取绝对值
xcA5 = np.arange(len(cA5))  # 频率

print(len(cA5),len(cD5),len(cD4),len(cD3),len(cD2),len(cD1),\
      len(cD1)+len(cD2)+len(cD3)+len(cD4)+len(cD5)+len(cA5))
plt.subplot(711)
plt.plot(x, y)
plt.title('Original wave')
plt.subplot(712)
plt.plot(xcA5, ycA5, 'r')
plt.title('DWT-ca5 ', fontsize=12, color='#7A378B')  # 注意这里的颜色可以查询颜色代码表
plt.subplot(713)
plt.plot(xcD5, ycD5, 'r')
plt.title('DWT-cd5 ', fontsize=12, color='#7A378B')  # 注意这里的颜色可以查询颜色代码表
plt.subplot(714)
plt.plot(xcD4, ycD4, 'r')
plt.title('DWT-cd4', fontsize=12, color='#7A378B')  # 注意这里的颜色可以查询颜色代码表
plt.subplot(715)
plt.plot(xcD3, ycD3, 'r')
plt.title('DWT-cd3', fontsize=12, color='#7A378B')  # 注意这里的颜色可以查询颜色代码表
plt.subplot(716)
plt.plot(xcD2, ycD2, 'r')
plt.title('DWT-cd2', fontsize=12, color='#7A378B')  # 注意这里的颜色可以查询颜色代码表
plt.subplot(717)
plt.plot(xcD1, ycD1, 'r')
plt.title('DWT-cd1', fontsize=12, color='#7A378B')  # 注意这里的颜色可以查询颜色代码表
plt.show()