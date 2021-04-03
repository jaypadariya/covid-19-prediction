

#Forecasting the gujarat  COVID-19 cases using Prophet
# importing the required libraries
import pandas as pd

# Visualisation libraries
import matplotlib.pyplot as plt
#%matplotlib inline
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
#finalDf[['DATE','count','District']].to_csv(r'D:\test\training1_data_gujarat.csv') 


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