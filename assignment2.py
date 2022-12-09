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
import seaborn as sb
 

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
    result_data = pd.concat([data1, dataframe], axis=1)
    return result_data


def statistics(dataframe):
    for value in dataframe.columns[2:].values:
        print("skewness of ",value)
        print(stats.skew(dataframe[value]))


def outlier(df):
    index_list = []
    for value in df.columns[2:]:
        q1 = df[value].quantile(0.25)
        q3 = df[value].quantile(0.75)
        iqr = q3-q1
        whisker_width = 1.5
        lower_whisker = q1 - (whisker_width*iqr)
        upper_whisker = q3 + (whisker_width*iqr)
        ls = df.index[(df[value] < lower_whisker) | (df[value] > upper_whisker)]
        index_list.extend(ls)
    index_list = sorted(set(index_list))  
    df = df.drop(index_list)
    return df
 
   
def normalise(dataframe):
    dataframe = (dataframe - dataframe.mean())/dataframe.std()
    return dataframe     


df_wb = pd.read_csv("worldbank.csv")
df_wb = df_wb.drop("Unnamed: 0", axis=1)
     
df_no2 = df_wb.groupby('Indicator Code').get_group('EN.ATM.NOXE.KT.CE')  
df_no2 = data_trim(df_no2)
df_no2 = data_clean2(df_no2)
df_no2_new = df_no2.loc[(df_no2['Year']==1990)|(df_no2['Year']==2000)| 
                        (df_no2['Year']==2010)|(df_no2['Year']==2020)]
statistics(df_no2)
print(df_no2_new.columns.values)


df_yield = df_wb.groupby('Indicator Code').get_group('AG.YLD.CREL.KG')
df_yield = data_trim(df_yield)
df_yield = data_clean2(df_yield)
statistics(df_yield)
df_yield_new = df_yield.drop(['Indicator Name'], axis=1)
df_yield_rolling = df_yield_new.rolling(window=4).mean()


df_elrcty = df_wb.groupby('Indicator Code').get_group('EG.ELC.RNWX.KH')
df_elrcty = data_trim(df_elrcty)
df_elrcty = data_clean2(df_elrcty)
clean_df_elrcty = outlier(df_elrcty)
# clean_df_elrcty_trim = clean_df_elrcty.loc[(clean_df_elrcty['Year']==1990)|(clean_df_elrcty['Year']==2000)| 
#                         (clean_df_elrcty['Year']==2009)|(clean_df_elrcty['Year']==2020)]
# print(clean_df_elrcty_trim)
# df_elrcty_new = []
# df_elrcty_new = pd.DataFrame(data=df_elrcty_new, columns=['Country', '1990', 
#                 '2000', '2009', '2020'])
# df_elrcty_new['Country'] = clean_df_elrcty.columns[2:].values
# df_elrcty_new['1990'] = clean_df_elrcty_trim.iloc[0 ,2:].values
# df_elrcty_new['2000'] = clean_df_elrcty_trim.iloc[1 ,2:].values
# df_elrcty_new['2009'] = clean_df_elrcty_trim.iloc[2 ,2:].values
# df_elrcty_new['2020'] = clean_df_elrcty_trim.iloc[3 ,2:].values
# print(df_elrcty_new)


df_rnwable = df_wb.groupby('Indicator Code').get_group('EG.FEC.RNEW.ZS') 
df_rnwable = data_trim(df_rnwable)
df_rnwable = data_clean2(df_rnwable)
clean_df_rnwable = outlier(df_rnwable)





for value in df_no2.columns[2:].values:
    df = []
    df = pd.DataFrame(data=df, columns=["NO2", "Crop Yield", 
                                        " electricity production", 
                                        "energy consumption "])
    df['Crop Yield'] = df_yield[value].values
    df['NO2'] = df_no2[value].values
    df['electricity production'] = df_elrcty[value].values
    df['energy consumption'] = df_rnwable[value].values
    plt.figure(dpi=144, figsize=(10,7))
    dataplot = sb.heatmap(df.corr(), annot=True)
    plt.title(value)
    plt.show()

plt.figure(dpi=144,figsize=(12,10))
x_axis = np.arange(len(df_no2_new['Year']))
plt.bar(x_axis - 0.05,df_no2_new['Bangladesh'], width=0.05, label='Bangladesh')
plt.bar(x_axis - 0.1,df_no2_new['Brazil'], width=0.05, label='Brazil')
plt.bar(x_axis - 0.15,df_no2_new['China'], width=0.05, label='China')
plt.bar(x_axis - 0.2,df_no2_new['France'], width=0.05, label='France')
plt.bar(x_axis + 0.0,df_no2_new['United Kingdom'], width=0.05, 
        label='United Kingdom')
plt.bar(x_axis + 0.05,df_no2_new['India'], width=0.05, label='India')
plt.bar(x_axis + 0.1,df_no2_new['United States'], width=0.05, 
        label='United States')
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
plt.scatter(x=normalise(df_rnwable['Bangladesh']), 
            y=(normalise(df_no2['Bangladesh'])), label="Bangladesh")
plt.legend()
plt.xlabel("Renewable energy consumption")
plt.ylabel("Nitrous Oxide emmission")
plt.title("Relation between Renewable energy consumption and NO2 emission")
plt.show()

# plt.figure(dpi=144,figsize=(12,10))
# x_axis = np.arange(len(df_elrcty_new['Country']))
# plt.bar(x_axis - 0.25, df_elrcty_new['1990'], width=0.1, label='1990')
# plt.bar(x_axis - 0.15, df_elrcty_new['2000'], width=0.1, label='2000')
# plt.bar(x_axis + 0.15, df_elrcty_new['2009'], width=0.1, label='2009')
# plt.bar(x_axis + 0.25, df_elrcty_new['2020'], width=0.1, label='2020')
# plt.xticks(x_axis, df_elrcty_new['Country'])
# plt.legend()
# plt.show()
plt.figure(dpi=144, figsize=(12,9))
plt.pie([clean_df_elrcty['Brazil'].mean(), clean_df_elrcty['China'].mean(), 
         clean_df_elrcty['France'].mean(), clean_df_elrcty['India'].mean(), 
         clean_df_elrcty['United Kingdom'].mean(), 
         clean_df_elrcty['United States'].mean()], 
         autopct=lambda p:'{:.0f}%'.format(p))
plt.legend(['Brazil', 'China', 'France', 'India', 'United Kingdom', 
            'United States'])
plt.title("Average Renewable Electricity Production")
plt.show()

plt.figure(dpi=144, figsize=(10,7))
for value in clean_df_rnwable.columns[2:].values:
    plt.plot(clean_df_rnwable['Year'], clean_df_rnwable[value], label = value)
    plt.legend(loc='best')
    plt.xlim(1990,2020)
    plt.xlabel("Year")
    plt.ylabel("Percentage of consumption")
    plt.title("Renewable energy consumption")
plt.show()
