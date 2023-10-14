'''
Goal is to normalize total energy consumption of set of OEDI ComStock buildings and then calc their carbon
footprints using eia 930 emissions factors.
'''

import os
import pandas as pd

# set parent directory and filepaths
parent_dir = os.path.dirname(os.getcwd())
dir_oedi = os.path.join(parent_dir, 'data', 'oedi')
path_emissions_data = os.path.join(parent_dir, 'data', 'Region_NE_cleaned.csv')

# read oedi data and concat all data into single dataframe with building types as columns
df_agg = None
for filename in os.listdir(dir_oedi):
    if filename.endswith('.csv'):
        filepath = os.path.join(dir_oedi, filename)
        df = pd.read_csv(
            filepath,
            usecols=['timestamp', 'out.electricity.total.energy_consumption'],
            index_col='timestamp',
            parse_dates=True,
        )
        df.columns = [filename.rstrip('.csv')]
        if df_agg is None:
            df_agg = df
        else:
            df_agg = pd.concat([df_agg, df], axis=1)

# make it hourly
df_agg = df_agg.resample('h').mean()

# # transform the dataframe so that each column sums to the same amount (i.e. each building has the same annual usage)
# df_sum = df_agg.sum()
# df_transformed = (df_agg/df_sum)*100000 #for one hundred megawatt-hours

# now read emissions data and then join it with usage data
df_co2 = pd.read_csv(
    path_emissions_data,
    usecols=['Local time', 'CO2 Emissions Intensity for Consumed Electricity'],
    index_col='Local time',
    parse_dates=True
)
df_co2.rename(columns={'CO2 Emissions Intensity for Consumed Electricity': 'emissions intensity'}, inplace=True)

# deal with DST - probably better to localize both dfs but I don't have time for that rn
df_co2 = df_co2[~df_co2.index.duplicated(keep='first')]

# join dfs and then drop nans to leave us with only timestamps for which both sources have data (late 2018)
# df_joined = pd.concat([df_co2, df_transformed], axis=1)
df_joined = pd.concat([df_co2, df_agg], axis=1)
df_joined.dropna(inplace=True)

# transform the dataframe so that each column sums to the same amount (i.e. each building has the same annual usage)
emissions_col = df_joined.pop('emissions intensity')
df_sum = df_joined.sum()
df_transformed = (df_joined/df_sum)*100000 #for one hundred megawatt-hours
df_transformed['emissions intensity'] = emissions_col


# now multiply emissions intensity column by the rest into a new dataframe
footprint_df = df_transformed.multiply(df_transformed['emissions intensity'], axis='index')
footprint_df.drop(columns='emissions intensity', inplace=True)


print(footprint_df)

pass