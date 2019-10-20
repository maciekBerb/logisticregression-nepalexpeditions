# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:04:35 2019

@author: Maciek
"""

#Function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


data = pd.read_excel("year_y.xlsx")


def succes_rate(data, y_begin = 1950, y_end = 2017, interval = 1, show="succes"):
    data = np.array(data)
    modulo_scope = (y_end-y_begin)%interval
    count_in_int = 0
    succ_in_int = 0 
    analyzed_data = []
    curr_int = y_begin
    ox_values = [l for l in range(y_begin, y_end, interval)]
    oy_values = []
    interval_ended = False
    start = 0
    stop = 0
    data_len = len(data)
    
    for i in range (data_len):
        if data[i][0] == y_begin:
            start = i
            break
    
    for i in range(1,data_len):
        if data[-i][0] == y_end:
            stop = 1+ data_len-i
            break

    for i in data[start: stop]:
        if i[0] < curr_int+interval:
            interval_ended = False
            count_in_int += 1
            succ_in_int += i[1]
            if interval == 1:
                interval_ended = True
        else:
            interval_ended = True
            curr_int = i[0]
            analyzed_data.append([count_in_int, succ_in_int, succ_in_int/count_in_int])
            count_in_int = 0
            succ_in_int = 0

    if False == interval_ended and modulo_scope:
        analyzed_data.append([count_in_int, succ_in_int, succ_in_int/count_in_int])

    if show == "succes":
        [oy_values.append(i[2]) for i in analyzed_data]
    else:
        [oy_values.append(i[0]) for i in analyzed_data]
    ox_values = np.array(ox_values)
    oy_values = np.array(oy_values)
    fig = plt.figure(figsize = (12, 6))
    plt.plot(ox_values, oy_values)
    if show == "succes":
        plt.title("Fraction of succes during expeditions over selected years")
    else:
        plt.title("Amount of expeditions during years")
    plt.show()


  
  
  