# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 19:59:15 2022

@author: HP
"""

#importing required modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# def data_clean(dataframe):
#     """
#     This function takes the data frame as an argument and cleans the data 
#     frame by replacing NaN with median and returns the data frame
#     """
#     for country in dataframe.columns[3:].values:
#         dataframe[country] = dataframe[country].fillna(dataframe[country].median())
#         # dataframe[country] = (dataframe[country]-dataframe[country].mean())/dataframe[country].std()
#     return dataframe    

def data_trim(dataframe):
    dataframe = dataframe[(dataframe['Year'] > 1989) & (dataframe['Year'] <2021)]
    dataframe = dataframe[dataframe.columns[dataframe.columns.isin(['Year', 
                'Indicator Name','India', 'China', 'United Kingdom', 
                'United States', 'Brazil', 'France', 'Bangladesh'])]]
    return dataframe

# def outlier_check(dataframe):
#     plt.figure()
#     dataframe = dataframe.drop(['Year', 'Indicator Name'], axis=1)
#     dataframe.boxplot()
#     plt.show()

def data_clean2(dataframe):
    data1 = dataframe[dataframe.columns[dataframe.columns.isin(['Year', 
            'Indicator Name'])]]
    dataframe = dataframe.drop(['Year', 'Indicator Name'], axis=1)
    for value in dataframe.columns.values:
        dataframe[value] = pd.to_numeric(dataframe[value])
        dataframe[value] = dataframe[value].fillna(dataframe[value].median())
    #z = (dataframe - dataframe.mean()) / dataframe.std()
    
    # for value in z.columns.values:
    #     q1 = z[value].quantile(0.25)
    #     q3 = z[value].quantile(0.75)
    #     iqr = q3-q1
    #     whisker_width = 1.5
    #     lower_whisker = q1 - (whisker_width*iqr)
    #     upper_whisker = q3 + (whisker_width*iqr)
    #     for value2 in z[value]:
    #         if value2 > upper_whisker or value2 < lower_whisker:
    #             z[value] = z[value].replace(value2, z[value].median())
    result_data = pd.concat([data1, dataframe], axis=1)
    return result_data


def statistics(dataframe):
    for value in dataframe.columns[2:].values:
        print(stats.skew(dataframe[value]))

def outlier(df):
    for value in df.columns[2:]:
        q1 = df[value].quantile(0.25)
        q3 = df[value].quantile(0.75)
        iqr = q3-q1
        whisker_width = 1.5
        lower_whisker = q1 - (whisker_width*iqr)
        upper_whisker = q3 + (whisker_width*iqr)
        ls = df.index[(df[value] < lower_whisker) | (df[value] > upper_whisker)]
        print(ls)
def normalise(dataframe):
    dataframe = (dataframe - dataframe.mean())/dataframe.std()
    return dataframe     
    

df_wb = pd.read_csv("worldbank.csv")
df_wb = df_wb.drop("Unnamed: 0", axis=1)
     
df_no2 = df_wb.groupby('Indicator Code').get_group('EN.ATM.NOXE.KT.CE')
# df_no2 = data_clean(df_no2)  
df_no2 = data_trim(df_no2)
df_no2 = data_clean2(df_no2)
df_no2_new = df_no2.loc[(df_no2['Year']==1990)|(df_no2['Year']==2000)| 
                        (df_no2['Year']==2010)|(df_no2['Year']==2020)]
statistics(df_no2)
outlier(df_no2)
print(df_no2_new.columns.values)
# df_no2 = data_clean2(df_no2)
# outlier_check(df_no2)
# print(df_no2)
# df_no2['Year'] = pd.to_datetime(df_no2['Year'],format='%Y')
# print(df_no2)

df_yield = df_wb.groupby('Indicator Code').get_group('AG.YLD.CREL.KG')
# df_agri = data_clean(df_agri) 
df_yield = data_trim(df_yield)
df_yield = data_clean2(df_yield)
statistics(df_yield)
df_yield_new = df_yield.drop(['Indicator Name'], axis=1)
df_yield_rolling = df_yield_new.rolling(window=4).mean()

outlier(df_yield)

# df_yield_new = df_yield.loc[(df_yield['Year']==1990)|(df_yield['Year']==2000)| 
#                         (df_yield['Year']==2010)|(df_yield['Year']==2020)]
# df_yield_new = data_clean2(df_yield_new)



# df_france = []
# df_france = pd.DataFrame(data=df_france, columns =["NO2","Crop Yield"])
# df_france['Crop Yield'] = df_yield['France'].values
# df_france['NO2'] = df_no2['France'].values
# print(df_france.corr())


# df_UK = []
# df_UK = pd.DataFrame(data=df_UK, columns =["NO2","Crop Yield", "Rnwble elec prdction","rnwble enrgy cnsmption"])
# df_UK['Crop Yield'] = df_yield['United Kingdom'].values
# df_UK['NO2'] = df_no2['United Kingdom'].values
# print(df_UK.corr())


df_elrcty = df_wb.groupby('Indicator Code').get_group('EG.ELC.RNWX.KH')
# df_elrcty = data_clean(df_elrcty) 
df_elrcty = data_trim(df_elrcty)
df_elrcty = data_clean2(df_elrcty)
outlier(df_elrcty)



# outlier_check(df_elrcty)

df_rnwable = df_wb.groupby('Indicator Code').get_group('EG.FEC.RNEW.ZS') 
df_rnwable = data_trim(df_rnwable)
df_rnwable = data_clean2(df_rnwable)
# print(df_rnwable)
# outlier_check(df_rnwable)

for value in df_no2.columns[2:].values:
    df = []
    df = pd.DataFrame(data=df, columns =["NO2","Crop Yield"])
    df['Crop Yield'] = df_yield[value].values
    df['NO2'] = df_no2[value].values
    print(df.corr())
    

plt.figure(dpi=144,figsize=(12,10))
x_axis = np.arange(len(df_no2_new['Year']))
plt.bar(x_axis - 0.05,df_no2_new['Bangladesh'], width=0.05, label='Bangladesh')
plt.bar(x_axis - 0.1,df_no2_new['Brazil'], width=0.05, label='Brazil')
plt.bar(x_axis - 0.15,df_no2_new['China'], width=0.05, label='China')
plt.bar(x_axis - 0.2,df_no2_new['France'], width=0.05, label='France')
plt.bar(x_axis + 0.0,df_no2_new['United Kingdom'], width=0.05, label='United Kingdom')
plt.bar(x_axis + 0.05,df_no2_new['India'], width=0.05, label='India')
plt.bar(x_axis + 0.1,df_no2_new['United States'], width=0.05, label='United States')
plt.xticks(x_axis, df_no2_new['Year'])
plt.legend()
plt.show()


plt.figure(dpi=144, figsize=(10,7))
for value in df_yield_rolling.columns[1:].values:
    plt.plot(df_yield_rolling['Year'], df_yield_rolling[value], label = value)
    plt.legend(loc='best')
    plt.xlim(1990,2020)
    plt.xlabel("Year")
    plt.ylabel("kg/hectare")
    plt.title("Crop Yield")
plt.show()

plt.figure(dpi=144, figsize=(10,7))
plt.scatter(x=normalise(df_yield['China']), y=(normalise(df_no2['China'])), 
            label="China", marker="+")
plt.scatter(x=normalise(df_yield['India']), y=(normalise(df_no2['India'])), 
            label="India", marker="o")


plt.legend()
plt.xlabel("Crop Yield")
plt.ylabel("Nitrous Oxide emmission")
plt.title("Relation between crop yield and NO2 emission")

plt.show()

plt.figure(dpi=144, figsize=(10,7))
plt.scatter(x=normalise(df_yield['United Kingdom']), y=(normalise(df_no2['United Kingdom'])), 
            label="United KIngdom")


plt.legend()
plt.xlabel("Crop Yield")
plt.ylabel("Nitrous Oxide emmission")
plt.title("Relation between crop yield and NO2 emission")

plt.show()



