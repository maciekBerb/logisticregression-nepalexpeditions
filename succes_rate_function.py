# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:04:35 2019

@author: Maciek
"""

#Function with will calculate succes rate of expeditions per year. 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from collections import Counter



data = pd.read_excel("year_y.xlsx")
data1 = np.array(data)
years = data1[:,0]
succ = data1[:,1]

#def succes_rate(y0, y_end, interval):
y0 = 1950
y_count = 0
s_summ = 0 
s_r = 0 
a = 0
b = 0 
list1 = [l for l in range(1950, 2017)]
list2 = {}

for i in data1[:, 0]:
    if i == y0:
        y_count += 1
        s_summ += data1[a, 1]
        a += 1
    else:
        s_r = s_summ / y_count
        list2[b] = s_r
        y0 += 1
        y_count = 1
        s_summ = data1[a, 1]
        a += 1
        b += 1

ox = np.array(list1)
oy = np.array(list2)
plt.plot(ox, oy)
plt.show()

#Another function for 10 years periods.




  
  
  
  