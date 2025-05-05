import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
import csv
import re
import os
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from numpy import dot
from numpy.linalg import norm

datafilepath1 = 'data/Offences_Dec_General.csv'                # table 01
datafilepath2 = 'data/Offences_Dec_Offence_types.csv'          # table 02
datafilepath3 = 'data/Offences_June_General.csv'               # table 01
datafilepath4 = 'data/Offences_June_Offence_types.csv'         # table 02
datafilepath5 = 'data/Offences_Jan_2020_June_2020.csv'
datafilepath6 = 'data/NCOV_COVID_Cases_by_LGA_Source_2021.csv'
datafilepath7 = 'data/Covid_cases_dec.csv'
datafilepath8 = 'data/Covid_cases_june.csv'
datafilepath9 = 'data/Half_year_Covid_cases.csv'

# Useful functions - from Preprocessing of Crime Data
def remove_commas_int(df, column_name):
    """
    removes all commas in column in given dataframe and convert column values to integers
    """
    # remove commas
    pattern = r','
    for i in range(len(df)):
        df.loc[i, column_name] = re.sub(pattern, '', df.loc[i, column_name])   
    
    df = df.astype({column_name: int})
    return df

def remove_total_rows(df):
    """
    accepts dataframe, removes unnecessary rows from dataframe: unnecessary years and total data, 
    specifically for dataframes aggregate data - without offence divisions)
    """
    rows_remove = []
    
    # removes rows with totals for each Police Region
    for i in range(len(df)):
        if df.loc[i, 'Local Government Area'] == 'Total':
            rows_remove.append(i)
        elif df.loc[i, 'Year'] < 2017:
            rows_remove.append(i)
    df = df.drop(rows_remove, axis=0)
    
    # reset row numbers
    df = df.reset_index()
    return df

def keep_total_rows(df):
    """
    accepts dataframe, removes unnecessary rows from dataframe: unnecessary years and NON-total data,
    specifically for dataframes aggregate data - without offence divisions
    """
    rows_remove = []

    # removes rows without totals for each Police Region
    for i in range(len(df)):
        if df.loc[i, 'Local Government Area'] != 'Total':
            rows_remove.append(i)
        elif df.loc[i, 'Year'] < 2017:
            rows_remove.append(i)
    df.drop(rows_remove, axis=0, inplace=True)
    
    # reset row numbers
    df = df.reset_index(drop=True)
    return df

def group_types_agg(df):
    """
    accepts a dataframe, calculates the total offence count in a given year
    """
    df = (df.groupby(['Year']).agg({'Offence Count': ['sum']}))
    df.columns = ['Offence Count']
    df = df.reset_index()
    return df

def remove_commas(df, column_name):
    """
    removes all commas in column in given dataframe
    """
    pattern = r','
    for i in range(len(df)):
        df.loc[i, column_name] = re.sub(pattern, '', df.loc[i, column_name])
    return df

def remove_row_col_offence_div(df):
    """
    accepts dataframe, removes unnecessary columns/rows from dataframe and converts 
    numerical values to float, specifically for dataframes with offence division
    """
    rows_remove = []
    
    # removes unnecessary columns
    df = df.drop(['Offence Subdivision', 'Offence Subgroup'], axis=1)
    
    # removes rows with unwanted years
    for i in range(len(df)):
        if df.loc[i, 'Year'] < 2017:
            rows_remove.append(i)
    df = df.drop(rows_remove, axis=0)
    
    df = remove_commas(df, 'Offence Count')
    df = remove_commas(df, 'PSA Rate per 100,000 population')
    df = remove_commas(df, 'LGA Rate per 100,000 population')
    df = df.astype({'Offence Count': int, 'PSA Rate per 100,000 population': float, 
                    'LGA Rate per 100,000 population': float})
    return df

def group_types_div(df):
    """
    accepts a dataframe, calculates the total offence count, PSA Rate and LGA Rate per 
    offence division in a given year, PSA, LGA
    """
    
    df = (df.groupby(['Year', 'Police Service Area', 'Local Government Area', 'Offence Division']).
           agg({'Offence Count': ['sum'], 'PSA Rate per 100,000 population': ['mean'], 
               'LGA Rate per 100,000 population': ['mean']}))
    df.columns = ['Offence Count', 'PSA Rate per 100,000 population', 'LGA Rate per 100,000 population']
    df = df.reset_index()
    return df
###########################################################################################################

# PREPROCESSING DATA FROM datafilepath1 and datafilepath3

# transfers data from csv file into dataframe
offences_dec_lga = pd.read_csv(datafilepath1)
offences_june_lga = pd.read_csv(datafilepath3)

# removes unnecessary years and rows with totals
offences_dec_lga = remove_total_rows(offences_dec_lga)
offences_june_lga = remove_total_rows(offences_june_lga)



# tranfers data from csv file into dataframe
offences_dec_aggregate = pd.read_csv(datafilepath1)
offences_june_aggregate = pd.read_csv(datafilepath3)

# removes unnecessary years and NON-total data
offences_dec_aggregate = keep_total_rows(offences_dec_aggregate)
offences_june_aggregate = keep_total_rows(offences_june_aggregate)
    
# remove all commas from values in 'Offence Count' column
offences_dec_aggregate = remove_commas_int(offences_dec_aggregate, 'Offence Count')
offences_june_aggregate = remove_commas_int(offences_june_aggregate, 'Offence Count')

# calculates total offence count per year in Victoria
dec_agg_data = group_types_agg(offences_dec_aggregate)
june_agg_data = group_types_agg(offences_june_aggregate)

# uses 6-month data from Jan 2020 to June 2020 to determine offence count per 6-month interval

offences_jan_june = pd.read_csv(datafilepath5)
# calculate offence count for Jan-June 2020 period
offences_6month = 0
columns = list(offences_jan_june)
for i in range(1, len(columns)):
    offences_6month += offences_jan_june[columns[i]][4]

data_dict_half = {'Year': [2021, 2020, 2020, 2019, 2019, 2018, 2018, 2017, 2017, 2016], 
                  'Time Period': ['Jan-June', 'July-Dec','Jan-June', 'July-Dec', 'Jan-June', 
                                  'July-Dec', 'Jan-June', 'July-Dec', 'Jan-June', 'July-Dec'], 
                  'Offence Count':[0, 0, offences_6month, 0, 0, 0, 0, 0, 0, 0]}

# calculate offence count for 6-month intervals
(data_dict_half['Offence Count'])[1] = dec_agg_data.loc[3, 'Offence Count'] - offences_6month
(data_dict_half['Offence Count'])[0] = (june_agg_data.loc[4, 'Offence Count'] - 
                                        (data_dict_half['Offence Count'])[1]) 
(data_dict_half['Offence Count'])[3] = june_agg_data.loc[3, 'Offence Count'] - offences_6month

# dictionary mapping row number to Year for dec_agg_data and june_agg_data
year_row = {2017: 0, 2018: 1, 2019: 2, 2020: 3, 2021: 4}

for i in range(4, len(data_dict_half['Offence Count'])):
    if (data_dict_half['Time Period'])[i] == 'Jan-June':
        (data_dict_half['Offence Count'])[i] = (dec_agg_data.loc[year_row[(data_dict_half['Year'])[i]], 
                                                                 'Offence Count'] -
                                                (data_dict_half['Offence Count'])[i-1])
    else:
        (data_dict_half['Offence Count'])[i] = (june_agg_data.loc[year_row[(data_dict_half['Year'])[i]+1], 
                                                                  'Offence Count'] -
                                                (data_dict_half['Offence Count'])[i-1])

# dataframe containg offence count in Victoria per 6-month period
half_yr_agg_data = pd.DataFrame(data=data_dict_half)



# PREPROCESSING DATA FROM datafilepath2 and datafilepath4

# transfers data from csv file into dataframe
offence_type_dec_lga = pd.read_csv(datafilepath2)

# transfers data from csv file into dataframe
offence_type_june_lga = pd.read_csv(datafilepath4)

# removes unnecessary columns/rows from dataframe and converts numerical values to float
offence_type_dec_lga = remove_row_col_offence_div(offence_type_dec_lga)
offence_type_june_lga = remove_row_col_offence_div(offence_type_june_lga)

# calculates the total offence count, PSA Rate and LGA Rate per offence division in 
# a given year, PSA, LGA
grouped_dec_type = group_types_div(offence_type_dec_lga)
grouped_june_type = group_types_div(offence_type_june_lga)

#########################################################################################################

### Data wrangling for COVID-19 cases data

covid_data = pd.read_csv(datafilepath6,dtype=None)

cases_per_month = dict()
cases_per_lga = dict()
lga_list = list()

month = 1
year = 2020

while(True):
    for index in range(len(covid_data)):
        
        if covid_data["Localgovernmentarea"][index].lower() != 'overseas':
            if covid_data["Localgovernmentarea"][index] not in lga_list:
                lga_list.append(covid_data["Localgovernmentarea"][index])
        
            if month<10:
                date = "0"+str(month)+"/"+str(year)
            else:
                date = str(month)+"/"+str(year)
        
            if date in covid_data["diagnosis_date"][index]:
                if date not in cases_per_month:
                    cases_per_month[date] = 1
                else:
                    cases_per_month[date] += 1
                
                lga = covid_data["Localgovernmentarea"][index]+" "+date
                if lga not in cases_per_lga:
                    cases_per_lga[lga] = 1
                else:
                    cases_per_lga[lga] += 1
                
      
    if month == 10 and year == 2021:
        break
    else:
        if month == 12:
            month = 1
            year = 2021
        else:
            month += 1
        
### Printing graphs to visualise each LGA Covid-19 cases in a month
for lga_name in lga_list:
    covid_cases = list()
    date = list()
    for data in cases_per_lga:
        if lga_name in data:
            covid_cases.append(cases_per_lga.get(data))
            date.append(data.replace(lga_name, '', 1))
            
    plt.title('Covid data per LGA per month: ' + lga_name)
    plt.bar(date, covid_cases)
    plt.xticks(date, rotation='vertical')
    plt.ylabel('No. of COVID-19 cases')
    plt.xlabel('Date')
    plt.clf()
            
        
### Wrangling for 6-month periods

half_year_header = ['Year', 'Time period', 'Local Government Area', 'Covid-19 cases']
half_year_periods = list()

for lga in lga_list:
    cases_count = 0
    month = 1
    year = 2020
    
    while(True):
    
        if month<10:
            dict_id = lga+" 0"+str(month)+"/"+str(year)
        else:
            dict_id = lga+" "+str(month)+"/"+str(year)
        
        if dict_id in cases_per_lga:
            cases_count+=cases_per_lga.get(dict_id)
        
        if month == 10 and year == 2021:
            break
        else:
            if month == 12:
                month = 1
                year += 1 
            else:
                month += 1

            
        if month == 7:
            half_year_periods.append([year, 'Jan - June', lga, cases_count])
            cases_count = 0
        elif month == 1:
            half_year_periods.append([year-1, 'July - Dec', lga, cases_count])
            cases_count = 0
        
with open('data/Half_year_Covid_cases.csv', 'w') as csvfile: 
    csvwriter = csv.writer(csvfile)
    
    csvwriter.writerow(half_year_header)
    
    for data in half_year_periods:
        csvwriter.writerow(data)
        
        
### Wrangling for Covid-19 data year ending in June

covid_header = ['Year', 'Year Ending', 'Local Government Area', 'Covid-19 cases']
ending_june = list()
ending_dec = list()

for lga in lga_list:
    june_cases_count = 0
    dec_cases_count = 0
    month = 1
    year = 2020
    
    while(True):
    
        if month<10:
            dict_id = lga+" 0"+str(month)+"/"+str(year)
        else:
            dict_id = lga+" "+str(month)+"/"+str(year)
        
        if dict_id in cases_per_lga:
            june_cases_count+=cases_per_lga.get(dict_id)
            dec_cases_count+=cases_per_lga.get(dict_id)

            
        if month == 10 and year == 2021:
            break
        else:
            if month == 12:
                month = 1
                year += 1 
            else:
                month += 1
                
        
        if month == 7:
            ending_june.append([year, 'June', lga, june_cases_count])
            june_cases_count = 0
            
        if month == 1:
            ending_dec.append([year-1, 'December', lga, dec_cases_count])
            dec_cases_count = 0

with open('data/Covid_cases_dec.csv', 'w') as csvfile: 
        csvwriter = csv.writer(csvfile)
    
        csvwriter.writerow(covid_header)
    
        for data in ending_dec:
            csvwriter.writerow(data)
            
with open('data/Covid_cases_june.csv', 'w') as csvfile: 
        csvwriter = csv.writer(csvfile)
    
        csvwriter.writerow(covid_header)
    
        for data in ending_june:
            csvwriter.writerow(data)
            
            
### Visualisation for overview trend in COVID-19 and crime rate in Victoria

covid_data_dec = pd.read_csv(datafilepath7,dtype=None)
covid_data_june = pd.read_csv(datafilepath8,dtype=None)
covid_data_halfyear = pd.read_csv(datafilepath9,dtype=None)
crime_data_dec = pd.read_csv(datafilepath1,dtype=None)
crime_data_june = pd.read_csv(datafilepath3,dtype=None)

year = 2017

crime_vic = dict()
covid_vic = dict()
key_list = list()
crime_list = list()
covid_list = list()
while(year<2022):
    vic_crime_june = 0
    vic_crime_dec = 0
    vic_covid_june = 0
    vic_covid_dec = 0
    
    for index1 in range(len(crime_data_june)):
        if crime_data_june['Year'][index1] == year:
            vic_crime_june+=int(crime_data_june['Offence Count'][index1].replace(',',''))
            
    for index2 in range(len(crime_data_dec)):
        if crime_data_dec['Year'][index2] == year:
            vic_crime_dec+=int(crime_data_dec['Offence Count'][index2].replace(',',''))
            
    for index3 in range(len(covid_data_june)):
        if covid_data_june['Year'][index3] == year:
            vic_covid_june += covid_data_june['Covid-19 cases'][index3]
    
    for index4 in range(len(covid_data_dec)):
        if covid_data_dec['Year'][index4] == year:
            vic_covid_dec += covid_data_dec['Covid-19 cases'][index4]
    
    
    key_list.append('June ' + str(year))
    crime_list.append(vic_crime_june)
    covid_list.append(vic_covid_june)
    
    if year != 2021:
        key_list.append('December ' + str(year))
        crime_list.append(vic_crime_june)
        covid_list.append(vic_covid_june)
        
    year += 1
    
fig, ax1 = plt.subplots()

color = 'tab:red'
plt.xticks(rotation = 'vertical')
plt.title("Overview trend of COVID-19 and Crimes in Victoria (2017-2021)", pad=20)
ax1.set_xlabel('Time period')
ax1.set_ylabel('No. of recorded offenses in Victoria', color=color)
ax1.plot(key_list, crime_list, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('No. of COVID-19 cases in Victoria', color=color)  # we already handled the x-label with ax1
ax2.plot(key_list, covid_list, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Graph 1: Overview trend of COVID-19 and Crimes in Victoria (2017-2021)', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()


### Creating the graphs to check LGA offence records per offence division

offences_june = pd.read_csv(datafilepath4, dtype=None)

# To check which types of offence divisions are the most frequently happening in Victoria (Division A: Crimes against the person)
offence_div = dict()
for i in range(len(offences_june)):
    if offences_june["Offence Division"][i] not in offence_div:
        offence_div[offences_june["Offence Division"][i]] = 1
    else:
        offence_div[offences_june["Offence Division"][i]] += 1

print(offence_div)

# Removing all bracket elements and texts from the LGAs
pattern = r'\([^()]*\)'
for index in range(len(lga_list)):
    replaced = re.sub(pattern, '', lga_list[index])
    lga_list[index] = replaced.rstrip(" ")

offences_vic = list()
for i in offence_div:
    crime_count = list()
    
    for lga in lga_list:
        lga_crime = 0
        for index in range(len(offences_june)):
            if offences_june["Local Government Area"][index] == lga and offences_june["Offence Division"][index] == i:
                lga_crime+=1
                
        crime_count.append(lga_crime)
        
    print(crime_count)
    fig, ax = plt.subplots(figsize=(30, 6))

    # Create bar graph
    ax.bar(lga_list, crime_count)
    plt.xticks(lga_list, rotation='vertical')
    plt.title("Recorded offences per LGA for Offence Division, Year ending in June: "+ i)
    plt.xlabel("Local Government Areas in Victoria")
    plt.ylabel("No. of recorded offences")
    plt.savefig("Recorded offences per LGA for Offence Division, Year ending in June: "+ i, dpi=300, bbox_inches='tight')

    plt.show()
    plt.clf()


#############################################################################################################################

### Visualisations for crime rate per 100000 population of LGA vs number of COVID-19 cases in LGA
## Data wrangling

# Function that displays entire dataframe on Jupyter Notebook instead of truncated version
def display_full_dataframe(df):
    pd.set_option('display.max_rows', None, 'display.max_columns', None)
    display(df)
    pd.reset_option('display.max_rows', 'display.max_cols')

offences_dec_lga = offences_dec_lga.drop(columns=["index", "Police Region", "Offence Count"])
offences_dec_lga = offences_dec_lga.drop(offences_dec_lga[offences_dec_lga["Year"]!=2020].index)
offences_dec_lga = offences_dec_lga.drop(index=[79, 80])
offences_dec_lga = offences_dec_lga.reset_index()
offences_dec_lga = offences_dec_lga.drop(columns=["index"])
offences_dec_lga["Local Government Area"] = offences_dec_lga["Local Government Area"].str.strip()
offences_dec_lga = remove_commas(offences_dec_lga, "Rate per 100,000 population")
offences_dec_lga["Rate per 100,000 population"] = offences_dec_lga["Rate per 100,000 population"].astype(float)

offences_june_lga = offences_june_lga.drop(columns=["index", "Police Region", "Offence Count"])
offences_june_lga = offences_june_lga.drop(offences_june_lga[offences_june_lga["Year"]<2020].index)
offences_june_lga = offences_june_lga.drop(index=[79, 80, 160, 161])
offences_june_lga = offences_june_lga.reset_index()
offences_june_lga = offences_june_lga.drop(columns=["index"])
offences_june_lga["Local Government Area"] = offences_june_lga["Local Government Area"].str.strip()
offences_june_lga = remove_commas(offences_june_lga, "Rate per 100,000 population")
offences_june_lga["Rate per 100,000 population"] = offences_june_lga["Rate per 100,000 population"].astype(float)

dec2020_LGA_crime_rate = offences_dec_lga
jun2020_LGA_crime_rate = offences_june_lga
jun2020_LGA_crime_rate = jun2020_LGA_crime_rate.drop(jun2020_LGA_crime_rate[jun2020_LGA_crime_rate["Year"]!=2020].index)
jun2020_LGA_crime_rate = jun2020_LGA_crime_rate.reset_index()
jun2020_LGA_crime_rate = jun2020_LGA_crime_rate.drop(columns=["index"])
jun2021_LGA_crime_rate = offences_june_lga
jun2021_LGA_crime_rate = jun2021_LGA_crime_rate.drop(jun2021_LGA_crime_rate[jun2021_LGA_crime_rate["Year"]!=2021].index)
jun2021_LGA_crime_rate = jun2021_LGA_crime_rate.reset_index()
jun2021_LGA_crime_rate = jun2021_LGA_crime_rate.drop(columns=["index"])

covid_data_halfyear = covid_data_halfyear.rename(columns={"Time period": "Year ending"})
covid_data_halfyear = covid_data_halfyear.replace("Jan - June", "June")
covid_data_halfyear = covid_data_halfyear.replace("July - Dec", "December")
covid_data_halfyear = covid_data_halfyear.drop(index=[69, 70, 71, 195, 196, 197])
covid_data_halfyear = covid_data_halfyear.reset_index()
covid_data_halfyear = covid_data_halfyear.drop(columns=["index"])
for i in range(len(covid_data_halfyear)):
    if covid_data_halfyear["Local Government Area"][i][-1]==")":
        covid_data_halfyear = covid_data_halfyear.replace(covid_data_halfyear["Local Government Area"][i], covid_data_halfyear["Local Government Area"][i].split("(")[0])
covid_data_halfyear["Local Government Area"] = covid_data_halfyear["Local Government Area"].str.strip()

dec2020_LGA_COVID_cases = covid_data_halfyear
dec2020_LGA_COVID_cases = dec2020_LGA_COVID_cases.drop(dec2020_LGA_COVID_cases[dec2020_LGA_COVID_cases["Year"]!=2020].index)
dec2020_LGA_COVID_cases = dec2020_LGA_COVID_cases.drop(dec2020_LGA_COVID_cases[dec2020_LGA_COVID_cases["Year ending"]!="December"].index)
dec2020_LGA_COVID_cases = dec2020_LGA_COVID_cases.reset_index()
dec2020_LGA_COVID_cases = dec2020_LGA_COVID_cases.drop(columns=["index"])

jun2020_LGA_COVID_cases = covid_data_halfyear
jun2020_LGA_COVID_cases = jun2020_LGA_COVID_cases.drop(jun2020_LGA_COVID_cases[jun2020_LGA_COVID_cases["Year"]!=2020].index)
jun2020_LGA_COVID_cases = jun2020_LGA_COVID_cases.drop(jun2020_LGA_COVID_cases[jun2020_LGA_COVID_cases["Year ending"]!="June"].index)
jun2020_LGA_COVID_cases = jun2020_LGA_COVID_cases.reset_index()
jun2020_LGA_COVID_cases = jun2020_LGA_COVID_cases.drop(columns=["index"])

jun2021_LGA_COVID_cases = covid_data_halfyear
jun2021_LGA_COVID_cases = jun2021_LGA_COVID_cases.drop(jun2021_LGA_COVID_cases[jun2021_LGA_COVID_cases["Year"]!=2021].index)
jun2021_LGA_COVID_cases = jun2021_LGA_COVID_cases.drop(jun2021_LGA_COVID_cases[jun2021_LGA_COVID_cases["Year ending"]!="June"].index)
jun2021_LGA_COVID_cases = jun2021_LGA_COVID_cases.reset_index()
jun2021_LGA_COVID_cases = jun2021_LGA_COVID_cases.drop(columns=["index"])

## Data visualisations
jun2020_LGA_crime_rate_and_COVID_cases = pd.merge(jun2020_LGA_crime_rate, jun2020_LGA_COVID_cases, how='inner', on="Local Government Area")
jun2020_plot = sns.lmplot(x="Covid-19 cases", y="Rate per 100,000 population", data=jun2020_LGA_crime_rate_and_COVID_cases).fig.suptitle("Number of COVID-19 cases in Victorian LGAs vs Crime rate of Victorian LGAs per 100,000 population in June 2020", y=1)
plt.savefig("Number of COVID-19 cases in Victorian LGAs vs Crime rate of Victorian LGAs per 100,000 population in June 2020", dpi=300, bbox_inches="tight")

dec2020_LGA_crime_rate_and_COVID_cases = pd.merge(dec2020_LGA_crime_rate, dec2020_LGA_COVID_cases, how='inner', on="Local Government Area")
dec2020_plot = sns.lmplot(x="Covid-19 cases", y="Rate per 100,000 population", data=dec2020_LGA_crime_rate_and_COVID_cases).fig.suptitle("Number of COVID-19 cases in Victorian LGAs vs Crime rate of Victorian LGAs per 100,000 population in December 2020", y=1)
plt.savefig("Number of COVID-19 cases in Victorian LGAs vs Crime rate of Victorian LGAs per 100,000 population in December 2020", dpi=300, bbox_inches="tight")

jun2021_LGA_crime_rate_and_COVID_cases = pd.merge(jun2021_LGA_crime_rate, jun2021_LGA_COVID_cases, how='inner', on="Local Government Area")
jun2021_plot = sns.lmplot(x="Covid-19 cases", y="Rate per 100,000 population", data=jun2021_LGA_crime_rate_and_COVID_cases).fig.suptitle("Number of COVID-19 cases in Victorian LGAs vs Crime rate of Victorian LGAs per 100,000 population in June 2021", y=1)
plt.savefig("Number of COVID-19 cases in Victorian LGAs vs Crime rate of Victorian LGAs per 100,000 population in June 2021", dpi=300, bbox_inches="tight")
