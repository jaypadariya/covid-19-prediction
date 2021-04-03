# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 12:29:17 2020

@author: jaykumar.d.padariya
"""




# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
from fbprophet import Prophet
# Visualisation libraries
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import folium 
from folium import plugins
import plotly.express as px

#%%

#confirmed = pd.read_csv(r"D:\test\prophet\DistrictWiseReport20200810.csv")
confirmed = pd.read_csv(r"D:\test\rainfall\out2020\test\datatest.csv")

#df_confirmed = df_confirmed[:85]
confirmed.columns
#df_confirmed=df_confirmed.drop(['Unnamed: 8','Unnamed: 4'], axis = 1) 
#confirmed = confirmed.groupby(["Date"])
dfconfirmed = confirmed.groupby('date').sum()['Total_gj'].reset_index()
#%%
#fig = go.Figure()
##Plotting datewise confirmed cases
#fig.add_trace(go.Scatter(x=confirmed['Date'], y=confirmed['AVERAGE_TEMP'], mode='lines+markers', name='AVERAGE_TEMP_1',line=dict(color='blue', width=2)))
#
#fig.update_layout(title='chart for average temprature', xaxis_tickfont_size=14,yaxis=dict(title='Temp'))
#
#fig.show()
#
#fig = go.Figure()
#fig.add_trace(go.Scatter(x=confirmed['Date'], y=confirmed['AVERAGE_RH_1'], mode='lines+markers', name='AVERAGE_RH_1', line=dict(color='Red', width=2)))
#fig.update_layout(title='chart for average AVERAGE_Relative humidity', xaxis_tickfont_size=14,yaxis=dict(title='RH'))
#fig.show()
#
#fig = go.Figure()
#fig.add_trace(go.Scatter(x=confirmed['Date'], y=confirmed['AVERAGE_CO2_1'], mode='lines+markers', name='AVERAGE_CO2_1', line=dict(color='Green', width=2)))
#fig.update_layout(title='chart for average Co2', xaxis_tickfont_size=14,yaxis=dict(title='CO2'))
#fig.show()
#%%
dfconfirmed = confirmed.groupby('Date').sum()['Total_gj'].reset_index()
dfconfirmed.columns = ['ds','y']
#confirmed['ds'] = confirmed['ds'].dt.date
dfconfirmed['ds'] = pd.to_datetime(dfconfirmed['ds'])
#%%
#confirmed=pd.read_csv(r"/content/drive/My Drive/covid/Copy of time_series_covid19_confirmed_global (1) - time_series_covid19_confirmed_global (1) - Copy of time_series_covid19_confirmed_global (1) - time_series_covid19_confirmed_global (1).csv")
dfconfirmed.tail()
#confirmed=confirmed[:43]
#df_confirmed.tail(20)
dfconfirmed.rename(columns = {'Total_gj':'y'}, inplace = True) 
dfconfirmed.rename(columns = {'Date':'ds'}, inplace = True)
#%%


model_air=Prophet()
model_air.fit(dfconfirmed)
future_air=model_air.make_future_dataframe(periods=12, freq='D')
forecast_air=model_air.predict(future_air)
forecast_air[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12)



m = Prophet(interval_width=0.95)
m.fit(dfconfirmed)
future = m.make_future_dataframe(periods=10,freq='D')
future.tail()
#%%


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)
#forecast.to_csv("/content/drive/My Drive/BRO/forecast_co2.csv")
forecast=forecast[-5:]
forecast.columns
forecast=forecast.drop(['trend', 'trend_lower', 'trend_upper',
       'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
       'weekly', 'weekly_lower', 'weekly_upper', 'multiplicative_terms',
       'multiplicative_terms_lower', 'multiplicative_terms_upper'], axis = 1) 
#forecast=forecast.rename(index = {"ds": "Date","yhat_lower":"Lower_temp_probability","yhat_upper":"upper_temp_probability","yhat": "Predicted_Temp"},inplace = True) 
forecast.rename(columns = {"ds": "Date","yhat_lower":"Lower_co_probability","yhat_upper":"upper_CO2_probability","yhat": "Predicted_AVERAGE_TEMP_1"}, inplace = True) 
#forecast.to_csv("/content/drive/My Drive/BRO/forecast_co2_original.csv")
#%%
#confirmed_forecast_plot = m.plot(forecast)
from plotly.offline import init_notebook_mode, plot_mpl
fig = go.Figure()
fig.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Predicted_AVERAGE_TEMP_1'], mode='lines+markers', name='AVERAGE_Temp', line=dict(color='Green', width=2)))
fig.update_layout(title='chart for predicted average Temp', xaxis_tickfont_size=14,yaxis=dict(title='Temp'))
fig.show()

confirmed_forecast_plot =m.plot_components(forecast)