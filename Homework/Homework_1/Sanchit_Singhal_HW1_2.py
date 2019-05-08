## Author : Sanchit Singhal
## Date : 29/01/2019

import pandas as pd

#import data
air = pd.read_csv('AllAirlines_Mar2013.csv')

#calculate total departures
tot_departures = air.groupby('ORIGIN')['PASSENGERS'].sum()
tot_departures = tot_departures.to_frame()

#filter data to only SW
air = air[air.UNIQUE_CARRIER_NAME == 'Southwest Airlines Co.']

#calculate southwest departures
departures_sw = air.groupby('ORIGIN')['PASSENGERS'].sum()
departures_sw = departures_sw.to_frame()

#calculate SW share
SW_share = departures_sw / tot_departures
SW_share.columns = ['Southwest']
SW_share = SW_share.fillna(0)

#write file
SW_share.to_csv('SouthwestShare.csv',index=True)
