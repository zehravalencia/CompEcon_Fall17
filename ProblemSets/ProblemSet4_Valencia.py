# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 22:16:26 2017

@author: zcagi
"""

import pandas as pd
import numpy as np
import xlrd
import scipy.optimize as opt
from scipy.optimize import minimize
import scipy.stats as stats
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
from geopy.distance import vincenty

df = pd.read_excel('C:\\Users\\zcagi\\Desktop\\\compecon\\radio_merger_data.xlsx')

#population and price in millions of dollars : RESULT =  price_millions AND  population_millions
df['price_millions'] = df['price']/1000000
df['population_millions'] = df['population_target']/1000000

#preparing the data for actual and counterfactual matches
#divide data set into 2:
N1 = len(df[(df['year']<2008)])
N2 = len(df[(df['year']>2007)])

m = 1
N = N1
BT = 1


import pandas as pd
import numpy as np
from geopy.distance import vincenty
import math
import scipy.optimize as opt
from scipy.optimize import minimize

df = pd.read_excel('C:\\Users\\zcagi\\Desktop\\\compecon\\radio_merger_data.xlsx')

#population and price in millions of dollars : RESULT =  price_millions AND  population_millions
df1['price_millions'] = df1['price']/1000000
df1['population_millions'] = df1['population_target']/1000000



"creating the data array and counterfactuals"
N1 = len(df[(df['year']<2008)])
N2 = len(df[(df['year']>2007)])

m = 1
N = N1
BT = 1


radio = np.empty((0, 16))

while (m <= 2):
    while (BT <= N-1):
        K = 1
        while (K <= N-BT):
            point1 = (df.iloc[BT-1, 3], df.iloc[BT-1, 4])
            point2 = (df.iloc[BT-1, 5], df.iloc[BT-1, 6])
            point3 = (df.iloc[BT+K-1, 3], df.iloc[BT+K-1, 4])
            point4 = (df.iloc[BT+K-1, 5], df.iloc[BT+K-1, 6])

            x1bm_y1tm = df.iloc[BT-1, 9] * df.iloc[BT-1, 10]      #f(b,t)
            x2bm_y1tm = df.iloc[BT-1, 11] * df.iloc[BT-1, 10]
            hhi1 = df.iloc[BT-1, 8]
            price1 = df.iloc[BT-1, 7]
            dbtm = vincenty(point1, point2).miles
            
            x1qm_y1um = df.iloc[BT+K-1, 9] * df.iloc[BT+K-1, 10]   #f((b',t')
            x2qm_y1um = df.iloc[BT+K-1, 11] * df.iloc[BT+K-1, 10]
            hhi1 = df.iloc[BT-1, 8]
            price1 = df.iloc[BT-1, 7]
            dqu = vincenty(point3, point4).miles

            x1bm_y1um = df.iloc[BT-1, 9] * df.iloc[BT+K-1, 10]   #f(b',t)
            x2bm_y1um = df.iloc[BT-1, 11] * df.iloc[BT+K-1, 10]
            hhi_2 = df.iloc[BT+K-1, 8]
            price2 = df.iloc[BT+K-1, 7]
            dbu = vincenty(point1, point4).miles

            x1qm_y1tm = df.iloc[BT+K-1, 9] * df.iloc[BT-1, 10]     #f(b,t')
            x2qm_y1tm = df.iloc[BT+K-1, 11] * df.iloc[BT-1, 10]
            hhi2 = df.iloc[BT+K-1, 8]
            price2 = df.iloc[BT+K-1, 7]
            dqt = vincenty(point3, point2).miles

            radio_data = np.array([x1bm_y1tm, x2bm_y1tm, dbtm, x1qm_y1um, x2qm_y1um, dqu, x1bm_y1um, x2bm_y1um, dbu, x1qm_y1tm, x2qm_y1tm, dqt, price1, price2, hhi1, hhi2])
            print (m, BT, K, N)
            K = K + 1
            radio = np.append(radio, [radio_data], axis=0)

        BT = BT + 1
    N = N1 + N2 - 1

    m = m + 1


print(radio)

"Creating the score function"
def mse(params, radio):
    alpha, beta = params
    sum = 0
    i = 0
    while(i <= len(radio)-1):
        fbt = radio[i, 0] + alpha * radio[i, 1] + beta * radio[i, 2] + radio[i, 3] + alpha * radio[i, 4] + beta * radio[i, 5] - radio[i, 6] - alpha * radio[i, 7] - beta * radio[i, 8] - radio[i, 9] - alpha * radio[i, 10] - beta * radio[i, 11]


        if fbt > 0:
            sum = sum + 1

        i = i + 1
        print(sum)
    return -sum

b = (1, 1)
mse(b, radio)

"Initial Guess"
b1 = (0.4 , -0.2025)    

"Optimization routine"
fbt1 = opt.minimize(mse, b1, radio, method = 'Nelder-Mead', options={'disp': True})
print(fbt1)
###question 2
"Creating the score function"
def mse2(params, radio):
    delta, alpha, gamma, beta = params
    sum = 0
    i = 0
    while(i <= len(radio)-1):
        fbtA = gamma * radio[i, 0] + alpha * radio[i, 1] + delta * radio[i, 14] + beta * radio[i, 2] 
        fbtB = gamma * radio[i, 3] + alpha * radio[i, 4] + delta * radio[i, 14] + beta * radio[i, 5] 
        fbtC = gamma * radio[i, 6] + alpha * radio[i, 7] + delta * radio[i, 15] + beta * radio[i, 8] 
        fbtD = gamma * radio[i, 9] + alpha * radio[i, 10] + delta * radio[i, 15] + beta * radio[i, 11]
        fbtprice1 = radio[i, 12]
        fbtprice2 = radio[i, 13]
        
        if ((fbtA-fbtC) >= (fbtprice1-fbtprice2)) & ((fbtB-fbtD) >= (fbtprice2-fbtprice1)):
            sum = sum + 1

        i = i + 1
        print(sum)
    return -sum

a = (1,1)
mse(a, radio)

"Initial Guess"
b3 = (0.2, 0.4, 0.5, -0.2 )

"Optimization routine"
fbt2 = opt.minimize(mse2, b3, radio, method = 'Nelder-Mead', options={'disp': True})

print(fbt2)


