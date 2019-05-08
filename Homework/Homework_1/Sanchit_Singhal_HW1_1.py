## Author : Sanchit Singhal
## Date : 29/01/2019

import pandas as pd
import matplotlib.pyplot as plt

#import data
sw = pd.read_csv('Southwest_Mar2013.csv')

#calculate depatures per airport
departing = sw.groupby('ORIGIN').sum()
departing.columns = ['Departing']

#calculate arrivals per airport
arriving = sw.groupby('DEST').sum()
arriving.columns = ['Arriving']

#data cleansing
NumPassengersByAirport = pd.concat([departing, arriving], axis=1, sort=True)
NumPassengersByAirport = NumPassengersByAirport.fillna(0)

#write out file for table
NumPassengersByAirport.to_csv('NumPassengersByAirport.csv',index=True)

#plot data
NumPassengersByAirport.plot.scatter(x='Departing', y='Arriving', title='Number of Departing vs Arriving Passengers on SW for all airports')
plt.show()

#top five airports for departures
print(departing.sort_values(by=['Departing'],ascending=False).head(5))
