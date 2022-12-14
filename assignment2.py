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


#defining functions
def getfile(filename):
    """ This function takes file name as an argument. Read the file into a 
    dataframe and perform transposing of dataframe to transform countries into 
    columns. Finally returns the original dataframe and transposed 
    dataframe."""
    # Reading the csv file into a dataframe
    df_worldbank = pd.read_csv(filename) 
    df_worldbank1 = []
    #Creating another dataframe with coulmns Year, Indicator Name, and 
    #indicator code. 
    df_worldbank1 = pd.DataFrame(data=df_worldbank1, 
                                 columns=['Year', 'Indicator Name', 
                                          'Indicator Code'])
    #Storing the year values as cumcount of the indicator code.
    df_worldbank1['Year'] = df_worldbank.groupby( 
        ['Indicator Code']).cumcount().add(1)
    #Creating a for loop to replace cumcount values in the year column with
    #actual year values
    n=1960
    for i in range(1,267):
        for value in df_worldbank1['Year']:
            if value == i:
                df_worldbank1['Year'] = df_worldbank1['Year'].replace(value, n)
        n=n+1
    # creating another dataframe df_worldbank2 and copying df_worldbank1 to it.  
    df_worldbank2 = df_worldbank1.iloc[:4712, :].copy()
    # populating columns indicator name and indicator code of df_worldbank2
    #with df_worldbank column values.
    df_worldbank2['Indicator Name'] = df_worldbank.iloc[:4712, [2]]    
    df_worldbank2['Indicator Code'] = df_worldbank.iloc[:4712, [3]]
    #Getting the unique values from the column Country Name of df_worldbank
    #and creating a list country of it.
    country = []
    for value in df_worldbank['Country Name']:
        if value not in country:
            country.append(value)
    #Getting the column from the 4 the column of df_worldbank and creating
    #a list (list of years) 
    Year_list = []
    for value in df_worldbank.columns[4:].values:
        Year_list.append(value)
    #Creating a list indicator_value and using for loop to fetch the indicator
    #values of each country in each year and storing it in the indicator_value
    #list
    indicator_value = []
    for country_name in country:
        for year in Year_list:
            indicator_value.append(df_worldbank[year]. 
                                   loc[df_worldbank["Country Name"]== 
                                       country_name])
    #Getting the uniques values from the column indicator code and 
    #creating a list out of it.    
    indicator_code = []
    for value in df_worldbank2['Indicator Code']:
        if value not in indicator_code:
            indicator_code.append(value)
    #Creating an empty dataframe with columns as names of the countries  from
    #the country list     
    df_country = []
    df_country = pd.DataFrame(data=df_country, columns=country)   
    #Concatenating df_country with df_worldbank2 with axis=1 so that the 
    #dataframe df_worldbank2 contains the columns Years, Indicator Name, 
    #Indicator code and the list of countries.
    df_worldbank2 = pd.concat([df_worldbank2, df_country], axis=1)
    #Using for loop to populate each countries columns with its indicators 
    #value in each year.
    j=0
    for value in country:
        n=0
        for i in range(len(indicator_code), df_worldbank2.shape[0]+1, 
                       len(indicator_code)):
            df_worldbank2[value].iloc[n:i] = indicator_value[j]
            n = i
            j=j+1
    #returning transposed dataframe and original dataframe.
    return df_worldbank2, df_worldbank
             
                     
def data_trim(dataframe):
    """This function takes dataframe as an argument and trims the dataframe
    that satisfies a given condition and filters out only the required 
    columns and returns the trimmed data."""
    dataframe = dataframe[(dataframe['Year'] > 1989) & (dataframe['Year'] < 
                                                        2021)]
    dataframe = dataframe[dataframe.columns[dataframe.columns.isin(['Year', 
                'Indicator Name','India', 'China', 'United Kingdom', 
                'United States', 'Brazil', 'France', 'Bangladesh'])]]
    return dataframe


def data_clean2(dataframe):
    """This function cleans the dataframe by converting it into a numerical 
    type and fill NaN values with median and returns the cleaned data"""
    #Storing columns, Year and Indicator name into dataframe data1
    data1 = dataframe[dataframe.columns[dataframe.columns.isin(['Year', 
            'Indicator Name'])]]
    #Dropping the columns that are not required for cleaning
    dataframe = dataframe.drop(['Year', 'Indicator Name'], axis=1)
    #Using a for loop to get each attribute of the dataframe for cleaning
    for value in dataframe.columns.values:
        dataframe[value] = pd.to_numeric(dataframe[value])
        dataframe[value] = dataframe[value].fillna(dataframe[value].median())
    #Concatenating the cleaned dataframe with data1
    result_data = pd.concat([data1, dataframe], axis=1)
    return result_data


def statistics(dataframe):
    """This function takes the dataframe as an argument and find the skewness
    and kurtosis of the distribution."""
    print("skewness = ")
    print(stats.skew(dataframe))
    print("kurtosis = ")
    print(stats.kurtosis(dataframe))


def outlier(df):
    """This function takes dataframe as an argument and then removes the 
    outliers from the data and returns the dataframe"""
    index_list = []
    #Using a forloop to get each attribute of the dataframe 
    for value in df.columns[2:]:
        q1 = df[value].quantile(0.25) #Getting first quantile
        q3 = df[value].quantile(0.75) #Getting the third quantile
        iqr = q3-q1 #Finding the inner quantile range
        whisker_width = 1.5
        lower_whisker = q1 - (whisker_width*iqr) #finding the lower whisker
        upper_whisker = q3 + (whisker_width*iqr) #Finding the upper whisker
        # Finding the index values of the data that are greater than upper
        #whisker and lower than lower whisker 
        ls = df.index[(df[value] < lower_whisker) | (df[value] > 
                                                     upper_whisker)]
        # Storing the index values into an index list
        index_list.extend(ls)
    #Using sorted and set function to remove the repeated index values and 
    #sort the index values
    index_list = sorted(set(index_list))
    #Dropping the data points from the dataframe whose index values are in the 
    #index list
    df = df.drop(index_list)
    return df
 
   
def normalise(dataframe):
    """This function takes a dataframe as an argument and normalises the 
    dataframe"""
    dataframe = (dataframe - dataframe.mean())/dataframe.std()
    return dataframe     


#Calling the getfile function with the filename as argument.
df_wb, df_wb2 = getfile("API_19_DS2_en_csv_v2_4700503.csv")

#Using group by function to group the data on the basis of the Indicator code
#and getting the data of a specific indicator, NO2 emmission.    
df_no2 = df_wb.groupby('Indicator Code').get_group('EN.ATM.NOXE.KT.CE') 
#calling the data_trim function.  
df_no2 = data_trim(df_no2)
#Calling the data_clean2 function.
df_no2 = data_clean2(df_no2)
#Calling the statistics function
UK_no2 = normalise(df_no2['United Kingdom'])
print("Skewness and kurtosis of United Kingdom in NO2 emission")
statistics(UK_no2)

#Using group by function to group the data on the basis of the Indicator code
#and getting the data of a specific indicator, Cereal Yield.
df_yield = df_wb.groupby('Indicator Code').get_group('AG.YLD.CREL.KG')
#calling the data_trim function.
df_yield = data_trim(df_yield)
#Calling the data_clean2 function.
df_yield = data_clean2(df_yield)
#Dropping the column Indicator Name from the dataframe
df_yield_new = df_yield.drop(['Indicator Name'], axis=1)
df_yield_rolling = df_yield_new.rolling(window=4).mean()

#Using group by function to group the data on the basis of the Indicator code
#and getting the data of a specific indicator, Renewable electricity 
#production.
df_elrcty = df_wb.groupby('Indicator Code').get_group('EG.ELC.RNWX.KH')
#calling the data_trim function.
df_elrcty = data_trim(df_elrcty)
#Calling the data_clean2 function.
df_elrcty = data_clean2(df_elrcty)
#Calling the outlier function to remove outliers from the data.
clean_df_elrcty = outlier(df_elrcty)

#Using group by function to group the data on the basis of the Indicator code
#and getting the data of a specific indicator, Renewable energy consumption.
df_rnwable = df_wb.groupby('Indicator Code').get_group('EG.FEC.RNEW.ZS') 
#calling the data_trim function.
df_rnwable = data_trim(df_rnwable)
#Calling the data_clean2 function.
df_rnwable = data_clean2(df_rnwable)
#Calling the outlier function to remove outliers from the data.
clean_df_rnwable = outlier(df_rnwable)

#Using for loop to get the list of countries 
for value in df_no2.columns[2:].values:
    df = []
    #Creating a Data Frame to store the Indicator Values of each country
    df = pd.DataFrame(data=df, columns=["NO2", "Cereal Yield", 
                                        " electricity production", 
                                        "energy consumption "])
    df['Cereal Yield'] = df_yield[value].values
    df['NO2'] = df_no2[value].values
    df['electricity production'] = df_elrcty[value].values
    df['energy consumption'] = df_rnwable[value].values
    # Plotting a heat map to show the correlation between each indicators of 
    #a country.
    plt.figure(dpi=144, figsize=(10,7))
    dataplot = sb.heatmap(df.corr(), annot=True)
    plt.title(value)
    plt.show()
    
#Plotting a Bar Plot to show the NO2 emmission of each country over the years
plt.figure(dpi=144,figsize=(10,7))
#filtering the dataframe df_no2 to get the data of specific years
df_no2_new = df_no2.loc[(df_no2['Year']==1990)|(df_no2['Year']==2000)| 
                        (df_no2['Year']==2010)|(df_no2['Year']==2020)]
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
plt.rcParams['font.size'] = '12'
plt.xlabel("Years")
plt.title("NO2 Emmission(thousand metric tons of CO2 equivalent)")
plt.show()

#Plotting a line plot to the show the trend in the cereal yields over the 
#years.
plt.figure(dpi=144, figsize=(10,7))
for value in df_yield_rolling.columns[1:].values:
    plt.plot(df_yield_rolling['Year'], df_yield_rolling[value], label = value)
    plt.legend(loc='best')
    plt.xlim(1990,2020)
    plt.xlabel("Year")
    plt.ylabel("kg/hectare")
    plt.title("Cereal Yield")
plt.show()

#Plotting a Scatter plot to show the realtion between NO2 emmission and
#cereal yield of China and India.
plt.figure(dpi=144, figsize=(10,7))
#Calling normalise function as an argument inside scatter plot
plt.scatter(x=normalise(df_yield['China']), y=(normalise(df_no2['China'])), 
            label="China", marker="+")
plt.scatter(x=normalise(df_yield['India']), y=(normalise(df_no2['India'])), 
            label="India", marker="o")
plt.legend()
plt.xlabel("Crop Yield")
plt.ylabel("Nitrous Oxide emmission")
plt.title("Relation between crop yield and NO2 emission")
plt.show()

#Plotting a scatter plot to show the relation between NO2 emmission and 
#renewable energy consumption.
plt.figure(dpi=144, figsize=(10,7))
#Calling the normalise function as an argument inside scatter plot
plt.scatter(x=normalise(df_rnwable['Bangladesh']), 
            y=(normalise(df_no2['Bangladesh'])), label="Bangladesh")
plt.legend()
plt.xlabel("Renewable energy consumption")
plt.ylabel("Nitrous Oxide emmission")
plt.title("Relation between Renewable energy consumption and NO2 emission")
plt.show()

# Plotting a pie chart to show the average production of renewable electricity
# of each country over the years.
plt.figure(dpi=144, figsize=(25,10))
#Using numpy mean function to find the average of each country
plt.pie([np.mean(clean_df_elrcty['Brazil']), np.mean(clean_df_elrcty['China']), 
         np.mean(clean_df_elrcty['France']), np.mean(clean_df_elrcty['India']), 
         np.mean(clean_df_elrcty['United Kingdom']), 
         np.mean(clean_df_elrcty['United States'])], 
         autopct=lambda p:'{:.0f}%'.format(p))
plt.legend(['Brazil', 'China', 'France', 'India', 'United Kingdom', 
            'United States'])
plt.title("Average Renewable Electricity Production")
plt.show()

#Plotting a line plot using pandas to show the trend in the renewable energy 
#consumption over the years.
clean_df_rnwable.plot("Year", ['Bangladesh', 'Brazil', 'China', 'France', 
                      'United Kingdom', 'India', 'United States'], 
                      figsize=(10,7))
plt.ylabel("Percentage of consumption")
plt.title("Renewable energy consumption")
plt.show()

#Using a histogram to plot the distribution of No2 emission in United Kingdom
plt.figure(dpi=144)
plt.hist(UK_no2, bins=4)
plt.title("NO2 emission of United Kingdom")
plt.show()

