# -*- coding: utf-8 -*-

# Markov Chains in Python: Beginner Tutorial
# https://www.datacamp.com/community/tutorials/markov-chains-python-tutorial

# The statespace
import PySimpleGUI as sg
import os
import pandas as pd
import sklearn
import sys
import numpy as np
import matplotlib.pyplot as plt

path = r"D:\Users\703143501\Documents\Genpact Internal\Debu_GUI"
os.chdir(path)

# Import data set

transitionName1 = pd.read_csv('transitionName.csv', encoding = 'ISO-8859-1')
transitionMatrix1 = pd.read_csv('transitionMatrix.csv', encoding = 'ISO-8859-1')

Data = pd.read_csv('Data_Amount_Lines_Jan16.csv', encoding = 'ISO-8859-1')
amount = pd.DataFrame(Data.iloc[:,1:2])
lines = pd.DataFrame(Data.iloc[:,2:3])
monthlyNDforecastData = pd.read_csv('monthlyNDforecast.csv', encoding = 'ISO-8859-1')
monthlyNDforecast = pd.DataFrame(monthlyNDforecastData.iloc[:,1:2])
#Combine all details
transitionName = pd.DataFrame(transitionName1.iloc[:,1:9])
transitionMatrix = pd.DataFrame(transitionMatrix1.iloc[:,1:9])

#states = transitionMatrix.columns.values
states = list(transitionMatrix)
print(states)
states = ["ND", "1-30", "31-60", "61-90", "91-180", "180-360", "360+", "Closed"]

# Possible sequences of events
transitionName = [["_".join([str(states[i]), str(states[j])]) for j in range(len(states))] for i in range(len(states))]

# Probabilities matrix (transition matrix)
transitionMatrix = np.array(
                    [[0.22,	0.22,	0.00,	0.00,	0.00,	0.00,	0.00,	0.56],
                    [0.00,	0.00,	0.49,	0.01,	0.00,	0.00,	0.00,	0.50],
                    [0.00,	0.00,	0.00,	0.75,	0.01,	0.00,	0.00,	0.24],
                    [0.00,	0.00,	0.00,	0.00,	0.85,	0.00,	0.00,	0.15],
                    [0.00,	0.00,	0.00,	0.00,	0.59,	0.26,	0.00,	0.15],
                    [0.00,	0.00,	0.00,	0.00,	0.00,	0.83,	0.10,	0.07],
                    [0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.96,	0.04],
                    [0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	0.00,	1.00]])


if sum(transitionMatrix[0,:])+sum(transitionMatrix[1,:])+sum(transitionMatrix[2,:])+sum(transitionMatrix[3,:])+sum(transitionMatrix[4,:])+sum(transitionMatrix[5,:])+sum(transitionMatrix[6,:])+sum(transitionMatrix[7,:]) != 8:
    print("Somewhere, something went wrong. Transition matrix, perhaps?")

else:
    print("All is gonna be okay, you should move on!! ;)")

   
# A function that implements the Markov model to forecast the state/mood.

#**************************************
# Forecast periods
months = 12
# Entry states
activityToday = "ND"    
#*******************************************

# Forecast
# Lines
Lines1= np.matrix(lines.iloc[:8,0])
print(Lines1)
print(monthlyNDforecast.iloc[1,0])
##########
############
#############
#Need to check forecast_lines0 return value
#print("Volume (#Lines) forecast after " + str(months) + " months for the scenario where starting state is " + activityToday  + " : ")
forecast_lines0 = transitionMatrix.T * Lines1.T
print(transitionMatrix)
print(Lines1)
print(forecast_lines0)

#monthlyNDforecastM1 = np.matrix([monthlyNDforecast.iloc[0,0],0,0,0,0,0,0,0]).T
#print(monthlyNDforecastM1)
#LinesM1 = (forecast_lines0 + monthlyNDforecastM1)
#print(LinesM1)
#forecast_linesM = transitionMatrix.T * LinesM1
#print(forecast_linesM)
        
#monthlyNDforecastM = np.matrix([monthlyNDforecast.iloc[i,0],0,0,0,0,0,0,0]).T
#LinesM = (LinesM + monthlyNDforecastM)
#forecast_linesM = transitionMatrix.T * LinesM.T
        
#monthlyNDforecastM = np.matrix([monthlyNDforecast.iloc[0,0],0,0,0,0,0,0,0]).T
#print(monthlyNDforecastM)
#print(forecast_lines0)
#LinesM = (forecast_lines0 + monthlyNDforecastM)
#print(LinesM)

#forecast_linesM = transitionMatrix * LinesM.T
#print(forecast_linesM)        
#**************************************
# Choose the starting state
forecast_linesM0 = transitionMatrix.T*(forecast_lines0 + np.matrix([monthlyNDforecast.iloc[0,0],0,0,0,0,0,0,0]).T)
forecast_linesM0
forecast_linesM1 = transitionMatrix.T*(forecast_linesM0 + np.matrix([monthlyNDforecast.iloc[1,0],0,0,0,0,0,0,0]).T)
forecast_linesM1
forecast_linesM2 = transitionMatrix.T*(forecast_linesM1 + np.matrix([monthlyNDforecast.iloc[2,0],0,0,0,0,0,0,0]).T)
forecast_linesM2
forecast_linesM3 = transitionMatrix.T*(forecast_linesM2 + np.matrix([monthlyNDforecast.iloc[3,0],0,0,0,0,0,0,0]).T)
forecast_linesM3

def activity_forecast(months, activityToday):
    #activityToday = activityToday
    activityList = [forecast_lines0]
    i = 0
    forecast_linesM = (forecast_lines0)
    
    
    while i != months:
        monthlyNDforecastM = np.matrix([monthlyNDforecast.iloc[i,0],0,0,0,0,0,0,0]).T
        #LinesM = (LinesM + monthlyNDforecastM)
        forecast_linesM = transitionMatrix.T * (forecast_linesM + monthlyNDforecastM)
        activityList.append(forecast_linesM)                                                      
                
        i += 1  
    return activityList

# To save every activityList
list_activity = []

activity_forecast(months, activityToday)
activity_forecast1 = np.asarray(activity_forecast(months, activityToday))
activity_forecast2 = activity_forecast1.T
shape =(8, months+1)

activity_forecast3= activity_forecast2.reshape(shape)
print(activity_forecast1)
print(activity_forecast2)
print(activity_forecast3)

#forecast_lines = pd.DataFrame(forecast_lines_ND,forecast_lines_31_60)
#forecast_lines_61_90, forecast_lines_91_180, forecast_lines_181_360, forecast_lines_360, forecast_lines_Closed)
#print(forecast_lines.T)
#print(forecast_lines_ND)
#forecast_lines_ND = forecast_lines.T

#To get the probability of starting at different state please change 
#titel = ("Volume (#Lines) forecast after " + str(days) + " days for the scenario where starting state is " + activityToday  + " : ")

#index = np.arange(len(states))
#plt.bar(states, forecast_lines_ND)
#activity_forecast3[0,0]

#plt.bar([1,2,3], [10,20,30])
import matplotlib
matplotlib.use('Agg')

'''
fig, ax = plt.subplots( nrows=1, ncols=1)

index = np.arange(months+20)
fig.xlabel('Months', fontsize=12)
ax.ylabel('No of lines', fontsize=12)
ax.xticks(index, fontsize=12, rotation=30)
ax.title(("Volume (#Lines) forecast for " + str(months) + " months for the scenario where starting state is " + activityToday  + " : "))

p0=ax.bar(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]), np.array(activity_forecast3[0,:]))
p1=ax.bar(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]), np.array(activity_forecast3[1,:]), bottom = np.array(activity_forecast3[0,:]),color='c')
p2=ax.bar(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]), np.array(activity_forecast3[2,:]), bottom = np.array(activity_forecast3[0,:])+np.array(activity_forecast3[1,:]),color='b')
p3=ax.bar(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]), np.array(activity_forecast3[3,:]), bottom = np.array(activity_forecast3[0,:])+np.array(activity_forecast3[1,:])+np.array(activity_forecast3[2,:]),color='k')
p4=ax.bar(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]), np.array(activity_forecast3[4,:]), bottom = np.array(activity_forecast3[0,:])+np.array(activity_forecast3[1,:])+np.array(activity_forecast3[2,:])+np.array(activity_forecast3[3,:]),color='y')
p5=ax.bar(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]), np.array(activity_forecast3[5,:]), bottom = np.array(activity_forecast3[0,:])+np.array(activity_forecast3[1,:])+np.array(activity_forecast3[2,:])+np.array(activity_forecast3[3,:])+np.array(activity_forecast3[4,:]),color='m')
p6=ax.bar(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]), np.array(activity_forecast3[6,:]), bottom = np.array(activity_forecast3[0,:])+np.array(activity_forecast3[1,:])+np.array(activity_forecast3[2,:])+np.array(activity_forecast3[3,:])+np.array(activity_forecast3[4,:])+np.array(activity_forecast3[5,:]),color='r')
p7=ax.bar(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]), np.array(activity_forecast3[7,:]), bottom = np.array(activity_forecast3[0,:])+np.array(activity_forecast3[1,:])+np.array(activity_forecast3[2,:])+np.array(activity_forecast3[3,:])+np.array(activity_forecast3[4,:])+np.array(activity_forecast3[5,:])+np.array(activity_forecast3[6,:]),color='g')

ax.legend((p0[0],p1[0],p2[0],p3[0],p4[0],p5[0],p6[0],p7[0]),("ND", "1-30", "31-60", "61-90", "91-180", "180-360", "360+", "Closed"), loc=2, fontsize=8)
#ax.plot([0,1,2], [10,20,3])

fig.savefig('figure.png')   # save the figure to file
plt.close(fig)
'''


index = np.arange(months+20)
plt.xlabel('Months', fontsize=12)
plt.ylabel('No of lines', fontsize=12)
plt.xticks(index, fontsize=12, rotation=30)
plt.title(("Volume (#Lines) forecast for " + str(months) + " months for the scenario where starting state is " + activityToday  + " : "))

p0=plt.bar(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]), np.array(activity_forecast3[0,:]))
p1=plt.bar(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]), np.array(activity_forecast3[1,:]), bottom = np.array(activity_forecast3[0,:]),color='c')
p2=plt.bar(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]), np.array(activity_forecast3[2,:]), bottom = np.array(activity_forecast3[0,:])+np.array(activity_forecast3[1,:]),color='b')
p3=plt.bar(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]), np.array(activity_forecast3[3,:]), bottom = np.array(activity_forecast3[0,:])+np.array(activity_forecast3[1,:])+np.array(activity_forecast3[2,:]),color='k')
p4=plt.bar(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]), np.array(activity_forecast3[4,:]), bottom = np.array(activity_forecast3[0,:])+np.array(activity_forecast3[1,:])+np.array(activity_forecast3[2,:])+np.array(activity_forecast3[3,:]),color='y')
p5=plt.bar(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]), np.array(activity_forecast3[5,:]), bottom = np.array(activity_forecast3[0,:])+np.array(activity_forecast3[1,:])+np.array(activity_forecast3[2,:])+np.array(activity_forecast3[3,:])+np.array(activity_forecast3[4,:]),color='m')
p6=plt.bar(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]), np.array(activity_forecast3[6,:]), bottom = np.array(activity_forecast3[0,:])+np.array(activity_forecast3[1,:])+np.array(activity_forecast3[2,:])+np.array(activity_forecast3[3,:])+np.array(activity_forecast3[4,:])+np.array(activity_forecast3[5,:]),color='r')
p7=plt.bar(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12]), np.array(activity_forecast3[7,:]), bottom = np.array(activity_forecast3[0,:])+np.array(activity_forecast3[1,:])+np.array(activity_forecast3[2,:])+np.array(activity_forecast3[3,:])+np.array(activity_forecast3[4,:])+np.array(activity_forecast3[5,:])+np.array(activity_forecast3[6,:]),color='g')

plt.legend((p0[0],p1[0],p2[0],p3[0],p4[0],p5[0],p6[0],p7[0]),("ND", "1-30", "31-60", "61-90", "91-180", "180-360", "360+", "Closed"), loc=2, fontsize=8)

plt.savefig('myfig')


#plt.legend(bbox_to_anchor=(1,1), bbox_transform =plt.gcf().transFigure)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
