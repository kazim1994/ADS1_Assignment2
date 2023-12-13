#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 06:20:40 2023

@author: Muhammad Kazim
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r"C:\Users\Hp\Downloads\API_19_DS2_en_csv_v2_6183479\API_19_DS2_en_csv_v2_6183479.csv"

file = pd.read_csv(file_path, skiprows=4)
print(file.head())

# A Method to create two separate datasets  
def Data_Process(file):
    '''This methods takes the original Dataset as input and provides
    two separate datasets as output where one has years as columns
    and other has coutries as columns'''
    latin_countries = ['Colombia','Argentina','Uruguay','Paraguay','Brazil','Bolivia','Peru','Chile','Ecuador']
    indicators = ['Cereal yield (kg per hectare)','Methane emissions (kt of CO2 equivalent)','Electricity production from oil sources (% of total)','Access to electricity (% of population)','Mortality rate, under-5 (per 1,000 live births)']

# filtering the data for the indicator i selected and the latin countries   
    data1 = file[(file['Indicator Name'].isin(indicators)) & (file['Country Name'].isin(latin_countries))]

# Transposing the data for making my visualization easy   
    data_trans = data1.T
    data_trans.columns = data_trans.iloc[0]
    data_trans = data_trans.iloc[1:]
    data_trans = data_trans[data_trans.index.str.isnumeric()]
    data_trans.index = pd.to_numeric(data_trans.index)
    data_trans['Years'] = data_trans.index
    data1 = data1.reset_index(drop=True)
    
    return data_trans, data1

# A method to get insights
def data_Stat1(data1):  
    '''This method is takes the normal dataset and provides basic 
    stats using .describe function'''
# Calculate summary statistics for the input DataFrame
    summary_statistics = data1.describe()
    
    # Append the standard deviation and average to the summary statistics DataFrame
    std_dev = summary_statistics.loc['std'] 
    avg = summary_statistics.loc['mean'] 
    print('\nstd\n', std_dev)
    print('\navg\n', avg)
    return summary_statistics


# Method to create separate Dataframes for my selected Latin Countries
def create_country_dataframes(file, latin_countries):
    ''' This method takes the original dataset as input and 
    gives multiple dataframes as output (dataframe of each country) 
    which are defined when calling this method and it also drops 
    multiple columns'''
    for country in latin_countries:
        country_df = file[file['Country Name'] == country].copy()
        country_df.name = country
        
        # Dropping unnecessary year columns
        columns_to_drop = ['1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', '1970',
                           '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981',
                           '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992',
                           '1993', '1993', '1994', '1995', '1996', '1997', '1998', '1999', 'Unnamed: 67']

        country_df.drop(columns=columns_to_drop, axis=1, inplace=True)
        print(f"Dropped columns for {country}")
        
        globals()[country] = country_df
 
# Method for Plotting line charts for Indicators 
def plot_indicator_for_countries(dataframes, countries, indicator):
    ''' This method creates a line chart by taking dataframes, countries 
    and selected indicators as input'''
    plt.figure(figsize=(12, 8))  # Adjust the figure size as needed

    for country in countries:
        # Filter data for the current country
        country_df = globals()[country]

        # Filter data for the selected indicator
        indicator_data = country_df[country_df['Indicator Name'] == indicator].iloc[0, 4:]

        # Plot the data for the current indicator and country
        plt.plot(country_df.columns[5:], indicator_data[:-1], label=f"{country} - {indicator}")

    plt.title(f"Line Chart for {indicator} - Latin American Countries")
    plt.xlabel('Year')
    plt.xticks(rotation=90)
    plt.ylabel('Value')
    plt.legend(latin_countries, fontsize='7')
    plt.show()        
        
# Method for plotting correlation heatmap 
def calculate_corr_heat_map(country_df, selected_years):
    ''' This method creates a correlatiion heatmap by taking the
    specific country and year as input'''
    selected_indicators = [
        'Mortality rate, under-5 (per 1,000 live births)',
        'Access to electricity (% of population)',
        'Methane emissions (kt of CO2 equivalent)',
        'Cereal yield (kg per hectare)',
        'Electricity production from oil sources (% of total)'
    ]
    
    # Filter the country's data for the selected indicators
    #selected_data = country_df[country_df['Indicator Name'].isin(selected_indicators)]
    selected_data = file[file['Indicator Name'].isin(selected_indicators)]
    
    # Pivot and calculate correlation between indicators for the selected years
    indicator_corr = selected_data.pivot_table(index='Country Name', columns='Indicator Name', values=selected_years)
    correlation_matrix = indicator_corr.corr()
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt='.2f')
    plt.title(f'Correlation Heatmap between Indicators of {country_df.name}')
    plt.show()

# Method for plotting bar charts of selected indicators of specific years
def plot_selected_years_bar_chart(selected_indicators, latin_countries, selected_years):
    ''' This mehtod creates a bar chart of all selected latin countries
    with specific indicators and specific years'''
    plt.figure(figsize=(12, 8))
    bar_width = 0.10

    for i, country in enumerate(latin_countries):
        country_df = globals()[country]

        for indicator in selected_indicators:
            indicator_data = country_df[country_df['Indicator Name'] == indicator].loc[:, selected_years]
            indicator_data = indicator_data.ffill(axis=1)

            plt.bar(
                [pos + i * bar_width for pos in range(len(selected_years))],
                indicator_data.values.flatten(),
                width=bar_width,
                label=f"{country} - {indicator}",
                alpha=0.7
            )

    plt.title(f"Bar Chart for {selected_indicators[0]} - Latin American Countries")
    plt.xlabel('Year')
    plt.xticks([pos + (len(latin_countries) - 1) * bar_width / 2 for pos in range(len(selected_years))], selected_years)
    plt.ylabel(selected_indicators[0])
    plt.legend(latin_countries, fontsize='7', loc='upper left')
    plt.show()

# Method for creating scatter plot using numpy
def create_scatter_plot(selected_indicator, countries):
    ''' This method creates a scatter plot of specific countries
    with specific indicators'''
    plt.figure(figsize=(12, 8))
    x_values = np.array([])
    y_values = np.array([])

    for country in countries:
        country_df = globals()[country]
        
        if 'Year' in country_df.columns:
            indicator_data = country_df[(country_df['Indicator Name'] == selected_indicator) & (country_df['Year'])].iloc[0, -1]
        else:
            indicator_data = country_df.iloc[0, -1]

        indicator_values = np.array(indicator_data)
        indicator_series = pd.Series(indicator_values).ffill()

        x_values = np.append(x_values, np.full_like(indicator_series, len(x_values)))
        y_values = np.append(y_values, indicator_series)

    plt.scatter(x_values, y_values, s=100, alpha=0.7, color='b')
    plt.xlabel('Country')
    plt.ylabel(selected_indicator)
    plt.title(f"Scatter Plot for {selected_indicator} - Latin Countries")
    plt.xticks(range(len(countries)), countries)
    plt.show()

# Calling all methods specifically 
data_trans, data1 = Data_Process(file)

# Call data_Stat1 function and store the returned values
summary_statistics = data_Stat1(data1)

print('\n', summary_statistics)
print('\n', data1)
print('\n', data_trans)

latin_countries = ['Colombia', 'Argentina', 'Uruguay', 'Paraguay', 
                   'Brazil', 'Bolivia', 'Peru', 'Chile', 'Ecuador'
                   ]
create_country_dataframes(file, latin_countries)

dataframes = [Colombia, Argentina, Uruguay, Paraguay, Brazil, Bolivia, Peru, Chile, Ecuador]
countries = ['Colombia', 'Argentina', 'Uruguay', 'Paraguay', 'Brazil',
             'Bolivia', 'Peru', 'Chile', 'Ecuador']

selected_indicator = 'Mortality rate, under-5 (per 1,000 live births)'
plot_indicator_for_countries(dataframes, countries, selected_indicator)

selected_indicators = 'Methane emissions (kt of CO2 equivalent)'
plot_indicator_for_countries(dataframes, countries, selected_indicators)

selected_years = ['2000']  # Replace with the desired years
calculate_corr_heat_map(Bolivia, selected_years)

#selected_years = ['2000','2010']  # Replace with the desired years
selected_years = ['2020']
calculate_corr_heat_map(Brazil, selected_years)

selected_years = ['2015']  # Replace with the desired years
calculate_corr_heat_map(Chile, selected_years)

selected_indicators = ['Cereal yield (kg per hectare)']
selected_years = ['2000', '2005', '2010', '2015']
plot_selected_years_bar_chart(selected_indicators, latin_countries, selected_years)

selected_indicators = ['Electricity production from oil sources (% of total)']
selected_years = ['2001', '2003', '2005', '2007', '2009', '2011', '2013', '2015']
plot_selected_years_bar_chart(selected_indicators, latin_countries, selected_years)

selected_indicator = 'Access to electricity (% of population)'
selected_countries = ['Colombia', 'Argentina', 'Uruguay', 'Brazil', 'Peru', 'Chile',]
create_scatter_plot(selected_indicator, selected_countries)

