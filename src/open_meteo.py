'''

'''

import requests
import pandas as pd

lat = 44.48
long = -73.21

forecast_url = f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={long}' \
               f'&current_weather=true' \
               f'&hourly=temperature_2m' \
               f'&temperature_unit=fahrenheit'


def get_historic_url(time_frame):
    '''Given a time frame, get the historical weather

    :type time_frame: tuple
    '''
    time_frame = list(time_frame)
    if not isinstance(time_frame[0], str):
        start = time_frame[0].date().strftime('%Y-%m-%d')
        time_frame[0] = start
    if not isinstance(time_frame[1], str):
        end = time_frame[1].date().strftime('%Y-%m-%d')
        time_frame[1] = end
    time_frame = (time_frame[0], time_frame[1])

    str_req = f'https://archive-api.open-meteo.com/v1/era5?latitude={lat}&longitude={long}' \
              f'&start_date={time_frame[0]}&end_date={time_frame[1]}' \
              f'&hourly=temperature_2m' \
              f'&temperature_unit=fahrenheit'

    return str_req


def get_hist_and_forecast(past_days=14, forecast_days=14):
    '''

    :return:
    '''
    url = f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={long}&hourly=temperature_2m' \
          f'&temperature_unit=fahrenheit&past_days={past_days}&forecast_days={forecast_days}'
    res = requests.get(url)
    dict_ = res.json()
    series = pd.Series(data=dict_['hourly']['temperature_2m'], index=dict_['hourly']['time'])
    series.index = pd.to_datetime(series.index)
    series.name = 'temp'

    return series


def open_meteo_get(time_frame='forecast'):
    """

    :param time_frame:
    :param lat:
    :param long:
    :return:
    """
    url = None
    if time_frame == 'forecast':
        url = forecast_url
    elif type(time_frame) == tuple:
        url = get_historic_url(time_frame)

    res = requests.get(url)
    dict_ = res.json()
    series = pd.Series(data=dict_['hourly']['temperature_2m'], index=dict_['hourly']['time'])
    series.index = pd.to_datetime(series.index)
    series.name = 'temp'

    return series
