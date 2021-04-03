# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:50:02 2020

@author: jaykumar.d.padariya
"""


#Forecasting the gujarat  COVID-19 cases using Prophet
# importing the required libraries
import pandas as pd

# Visualisation libraries
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import folium 
from folium import plugins
import numpy as np
from datetime import date
import glob
# Manipulating the default plot size
plt.rcParams['figure.figsize'] = 10, 12

# Disable warnings 
import warnings
from fbprophet import Prophet
import os
warnings.filterwarnings('ignore')
#url = "https://gujcovid19.gujarat.gov.in/"
#table = pd.read_html(url)[0]
#table1=table[:-1]
#print(url)
today_date = date.today()

table =pd.read_csv(r"D:\test\prophet\DistrictWiseReport20200810.csv")
#%%

day_wise_confirm=table['Active'].astype(str).str.split(' ')
# print(np.array(table))
cnf=list()
cnf_cumulative=list()
for i in np.array(day_wise_confirm):
	cnf.append(int(i[0]))
	cnf_cumulative.append(int(i[-1]))

print(cnf) 
# print(table) 
table['count'] = cnf

day_wise_test=table['Tested'].astype(str).str.split(' ')
# print(np.array(table))
tst=list()
tst_cumulative=list()
for i in np.array(day_wise_test):
	tst.append(int(i[0]))
	tst_cumulative.append(int(i[-1]))

print(tst) 
# print(table) 
table['tested'] = tst

day_wise_Cured=table['Recovered'].astype(str).str.split(' ')
# print(np.array(table))
crd=list()
crd_cumulative=list()
for i in np.array(day_wise_Cured):
	crd.append(int(i[0]))
	crd_cumulative.append(int(i[-1]))


print(crd) 
# print(table) 
table['Cured'] = crd

day_wise_death=table['Deaths'].astype(str).str.split(' ')
# print(np.array(table))
dth=list()
dth_cumulative=list()
for i in np.array(day_wise_death):
	dth.append(int(i[0]))
	dth_cumulative.append(int(i[-1]))
print(dth) 
# print(table) 
table['death'] = dth
da= list()
for i in range(len(dth)):
	da.append(today_date)
table['DATE'] = da	
# table=table[:-1]
# print(table)
# table.to_excel(str(today_date)+"_report"+".xlsx") 
########every_report--------------#
df = pd.DataFrame()
df['DATE'] = da
df['count'] = cnf
df['District']=table['District']


#%%


#----------------------------final_report---------------------------#
current_file = pd.read_csv(r'D:\test\prophet\d.csv')# ------  currrent _ file-------------

finalDf = pd.concat([df,current_file]) 
# print(finalDf)
finalDf[['DATE','count','District']].to_csv(r'D:\test\training1_data_gujarat.csv') 


#%%


#----------------trainng_data----------------------------------#
df_gujarat= pd.read_csv(r'D:\test\training1_data_gujarat.csv')
#df_gujarat = df

#%%
p=df_gujarat['District']
dp=list()
#forecasting
dir_path=r'D:\test\prophet'+str(today_date)
city=[]
file=[]
  # print(filename)
#os.mkdir(dir_path)
flag=1
#------------prophet prediction part
for i in  p:
  name = i
  if i not in city:
    city.append(i)
    try:
      
      df_s=df_gujarat[df_gujarat['District']==name]
      df_s
      confirmed_A =df_s.groupby('DATE').sum()['count'].reset_index()
      confirmed_A.columns = ['ds','y']
      # confirmed_A
      m = Prophet(interval_width=0.99)
      m.fit(confirmed_A)
      future = m.make_future_dataframe(periods=4)
      forecast = m.predict(future)
      print("----------------------------------------------------------------------------------",i,i)
      forecast[['ds','yhat_upper']].tail().to_csv(dir_path+'/'+name+'.csv', index=False)
#      file.append(forecast[['ds','yhat_upper']].tail())
      # forecast['State']=name
      # forecast['yhat_upper'] = forecast['yhat_upper'].astype(np.int64)
      # if flag==1:
      #   result_1= pd.DataFrame()
      #   forecast[['ds','yhat_upper','State']].tail().transpose().to_csv('/content/drive/My Drive/Colab Notebooks/result_18_4/prediction.csv', index=False)
      #   flag=0
        
      # else:
      #   df_final= pd.read_csv('/content/drive/My Drive/Colab Notebooks/result_18_4/prediction.csv')
      #   result_1 = pd.concat([ df_final,forecast[['ds','yhat_upper','State']].tail().transpose()], join='outer', axis=0)
    except:
      print(i)  
      
      #%%
      
      # dir_path='/content/drive/My Drive/Colab Notebooks/result_1'+str(today_date)
filenames = glob.glob(dir_path + "/*.csv")

dfs = []
big_frame=pd.DataFrame()

for filename in filenames:
  cd=pd.read_csv(filename)
  cd['district']=filename.split(r'/')[-1].split('.')[0]
  dfs.append(cd)

# Concatenate all data into one DataFrame

big_frame = pd.concat(dfs, ignore_index=True)
# big_frame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
#%%
cumulative_file = pd.DataFrame()
cumulative_file['District Name']=table['District']
cumulative_file['Active']=cnf_cumulative
cumulative_file['Recovered']=crd_cumulative
cumulative_file['Quarantine']=table['Quarantine']
cumulative_file['Tested']=tst_cumulative
cumulative_file['Deaths']=dth_cumulative
# cumulative_file.to_excel(dir_path+"/main_table.xlsx")#main_table

#%%

big_frame["yhat_upper"] =big_frame["yhat_upper"].astype(int)
big_frame = big_frame.pivot(index='district', columns='ds', values='yhat_upper')
#--------------------only_predection_result-----------------------#
# big_frame.to_csv(dir_path+"/prophet.csv")

#%%
#------------------------------------cumulative_value---------------------------------#
cum_guj=pd.DataFrame()
# big_frame['2020-04-22']

# cum_guj=big_frame.copy
cum_guj['Active']=cnf_cumulative
# cum_guj['2020-04-22']=big_frame['2020-04-22']
for i in big_frame.columns:
  val=big_frame[i]
  temp_val=[]
  for j in val:
    if int(j)<0:
      temp_val.append(abs(int(j)))
    else:
      temp_val.append(int(j))
  cum_guj[i]=temp_val
cum_guj=cum_guj.cumsum(axis=1)
cum_guj['District Name']=table['District']
# cum_guj

#%%
##-------------latlong file ----------------------------#
df_lat_long= pd.read_csv('/content/drive/My Drive/Colab Notebooks/lat_long.csv')

cumulative_file['Latitude']=df_lat_long['Latitude']
cumulative_file['Longitude']=df_lat_long['Longitude']
df_bed= pd.read_csv('/content/drive/My Drive/Colab Notebooks/totalbeds.csv')
cumulative_file['Total Beds']=df_bed['Total Beds']
# cumulative_file['Bed Utilization']=1-((cumulative_file['Total Beds']-cumulative_file['Active'])/cumulative_file['Total Beds'])
cumulative_file['Beds Used']=cumulative_file['Active']-cumulative_file['Recovered']-cumulative_file['Deaths']
cumulative_file['Bed Remaining']=cumulative_file['Total Beds']-cumulative_file['Beds Used']

# cumulative_file
for i in cum_guj.columns:
  val=cum_guj[i]
  temp_val=[]
  for j in val:
    temp_val.append(j)
  cumulative_file[i]=temp_val
# cumulative_file  



#%%
  
  #--------------final_file--------------------#
cumulative_file.to_excel(dir_path+"/prophet_gujarat.xlsx")

#%%

results_path='/content/drive/My Drive/Colab Notebooks/results'+str(today_date)
try:
  os.mkdir(results_path)
except:
  pass

cumulative_file.to_excel(results_path+"/prophet_gujarat.xlsx")