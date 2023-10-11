'''

'''

import pandas as pd
from src import open_meteo
from src.utils import TOWT

# set filepath for eia emissions data (EIA-930)
filepath_emissions_data = r'F:\NON PROJECT\Business Development\Presentations\2023 REV\Evolving Technology for a ' \
                          r'Renewable Energy Future\data\Region_NE.xlsx'

# set time frame
time_frame = ('2022-01-01', '2022-12-31')

# get weather series
# weather = open_meteo.open_meteo_get(time_frame)

# read emissions data
df = pd.read_excel(
    filepath_emissions_data,
    sheet_name='Published Hourly Data',
    usecols=['UTC time', 'CO2 Emissions Intensity for Consumed Electricity'],
    parse_dates=True
)

pass
emissions_series = df['colname']


