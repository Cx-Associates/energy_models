'''

'''

import os
import pandas as pd
from src import open_meteo
from src.utils import Dataset, TOWT

# set filepath for eia emissions data (EIA-930)
parent_dir = os.path.dirname(os.getcwd())
filepath_emissions_data = os.path.join(parent_dir, 'data', 'Region_NE_cleaned.csv')

# set time frame
time_frame = ('2022-01-01', '2022-12-31')

# get weather series
weather_series = open_meteo.open_meteo_get(time_frame)

# read emissions data
df = pd.read_csv(
    filepath_emissions_data,
    usecols=['Local time', 'CO2 Emissions Intensity for Consumed Electricity'],
    index_col='Local time',
    parse_dates=True
)
df.rename(columns={
    'CO2 Emissions Intensity for Consumed Electricity': 'emissions intensity'
}, inplace=True)

# truncate emissions to same time frame as weather data
emissions_series = df['emissions intensity'].truncate(
    before=time_frame[0], after=time_frame[1]
).rename('emissions_intensity')

# join the series
df = pd.concat([weather_series, emissions_series], axis=1)

pass
