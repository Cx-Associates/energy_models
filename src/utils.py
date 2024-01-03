import sys, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate
import matplotlib as mpl
import matplotlib.pyplot as plt
from subrepos.energy_models.src.apis.open_meteo import open_meteo_get

# for interactive plotting while debugging in PyCharm
plt.interactive(True)
mpl.use('TkAgg')


sys.path.append('')

# directories
dir_here = r'src'
dir_parent = os.path.dirname(dir_here)
dir_data = os.path.join(dir_parent, 'data')

#ToDo: normalize function for all regression factor-making


def read_weather_data(filepath, colname, freq='h', irreg=False):
    """Convenience function used to read weather data from a local file

    """
    try:
        df = pd.read_csv(
            filepath,
            usecols=['datehour'] + [colname],
            index_col='datehour',
            parse_dates=True
        )
    except (KeyError, ValueError) as error:
        df = pd.read_csv(
            filepath,
            index_col=[0],
            parse_dates=True
        )
        df.columns = [colname]
    df = df[~df.index.duplicated(keep='last')]
    df[colname] = pd.to_numeric(df[colname], errors='coerce')
    if irreg:
        df = df.resample('1min').ffill()
    df = df.resample(freq).mean()
    return df


def balance_point_transform(series, temp):
    """

    :param series:
    :param temp:
    :return:
    """
    return series.apply(lambda x: np.amax([x - temp, 0]) + temp)


def TOxT_column_labels(n_bins):
    """helper function for TOWT and TODT classes

    :param n_bins:
    :return:
    """
    labels = np.arange(1, n_bins + 1)
    labels_index = [np.int(x - 1) for x in labels]
    labels = ['t' + str(x) for x in labels]
    return labels, labels_index


def get_projected_year(tmy, history, forecast=None):
    '''

    :param history:
    :param forecast:
    :param tmy:
    :return:
    '''
    if isinstance(history, str):
        df_history = pd.read_csv(history)
    else:
        df_history = history
    if forecast is not None:
        if isinstance(forecast, str):
            df_forecast = pd.read_csv(forecast)
    else:
        df_forecast = forecast
    if isinstance(tmy, str):
        df_tmy = pd.read_csv(tmy)
    else:
        df_tmy = tmy
    df = df_tmy
    df[df.index == df_forecast.index] = df_forecast
    df[df.index == df_history.index] = df_history

    return df


def safe_sum(df):
    '''Calculates the sum of a timeseries array (dataframe or series) assuming any gaps in data are averages.
    Otherwise, you might have missing data for a few intervals without realizing it, and then a sum over the whole
    period would result in an artificially low value.

    Result should be the same as df.sum() if there are no missing timestamps.

    :param df: (pandas DataFrame or Series)
    :return: (float) the sum of the dataframe's columns (or series values)
    '''
    freq = pd.infer_freq(df.index)
    if not isinstance(freq, str):
        msg = '**! Could not infer frequency of dataframe; assuming it is hourly. Any summation metrics should be ' \
              'checked for accuracy. Dataframe header: \n \n'
        print(msg + df.head())
    range = df.index[-1] - df.index[0]
    divisor = pd.Timedelta(1, freq)
    len = range / divisor
    avg = df.mean()
    sum_ = len * avg

    return sum_

class TimeFrame():
    def __init__(self, arg):
        if isinstance(arg, tuple):
            self.tuple = arg
            # ToDo: check for correct api formatting
        elif isinstance(arg, str):
            pass
            # ToDO: auto-parse to ensure this is API-friendly
            # ToDo: where should TZ localization take place? Project has coordinates ...

class Dataset:
    """Generic dataset class comprising an energy time-series and a weather time-series.

    """
    #ToDO: update energy to use "energy" and not "kWh" in case MMBtu data from gas etc. is enabled

    def __init__(
            self,
            # energy_filepath=None,
            # weather_filepath=None,
            df=None,
            *args, **kwargs
    ):
        self.name = None
        self.energy_series = None
        self.temperature_series = None
        self.normalized_temperature_series = None
        self.df_joined = None
        self.sparse_df = None
        self.display_start = None
        self.display_end = None
        self.__dict__.update(kwargs.copy())

        if df is not None:
            pass
        else:
            df = pd.read_csv(self.energy_filepath, index_col=[0], parse_dates=True)
            # ToDo: write function to auto-recognize frequency. instead, resampling below df to hourly regardless
            s_energy = df['kWh'] #ToDo: handle differently if kW or kWh as far as resampling
            self.energy_series = s_energy.dropna()

            try:
                df = read_weather_data(self.weather_filepath, 'temp')
                s_temp = df['temp'] #ToDo: change naming convention to 'temp' not weather, and then have this
                # selected column using .iloc
            except pd.errors.EmptyDataError: #ToDo: no idea why I'm getting this
                df = pd.read_csv(self.weather_filepath, index_col=[0], parse_dates=True)
                df.columns = ['temp']
                s_temp = df['temp']

                #ToDo: wrap block below into iterative function
                s_energy = s_energy[~s_energy.index.duplicated(keep='last')]
                s_energy = pd.to_numeric(s_energy, errors='coerce')
                s_temp = s_temp[~s_temp.index.duplicated(keep='last')]
                s_temp = pd.to_numeric(s_temp, errors='coerce')

                df = pd.concat([s_energy, s_temp], axis=1)
                df.sort_index(inplace=True)
                df = df.resample('h').mean()

        self.df_full = df
        df.dropna(inplace=True)
        self.df_trimmed = df

        try:
            self.temperature_series = df['temp']
            self.X = self.temperature_series
        except KeyError:
            pass
        try:
            self.energy_series = df['kWh']
            self.Y = self.energy_series
        except KeyError:
            try:
                energy_colname = kwargs['energy_colname']
                self.energy_series = df[energy_colname]
                self.Y = self.energy_series
            except KeyError:
                pass
    def check_zeroes(self):
        return

    def check_sticky(self):
        return

class Var():
    """

    """
    def __init__(self, data=None):
        self.data = data
        self.train = None
        self.test = None
        self.pred = None
        self.norm = None

class Scores():
    """

    """
    def __init__(self):
        self.rsq = None
        self.cvrmse = None
        self.savings_uncertainty = None
        self.fsu = None

class Model():
    """Class intended for parameter modeling, feature_engineering.g. energy modeling, based on time-series data, using conventions
    like baseline period (sometimes divided into training and testing sets) and performance period. Reference ASHRAE
    14, USDOE's Superior Energy Performance (SEP), and other guidelines for determining building or process
    performance against some baseline period. Y is typically but not necessarily energy consumption.

    Typically instantiated as one of the child classes, feature_engineering.g. TOWT or TODT.

    """
    def __init__(self, data=None, **kwargs):
        """

        :param data: preferably a pandas.DataFrame, but can also be a list (feature_engineering.g. a list of pd.series or
        pd.dataframes), or any other type.
        :param kwargs: will be made into attributes.
        """
        self.reg = None  # or self.clf?
        self.X, self.Y, self.y = Var(), Var(), Var()
        self.scores = {}
        self.Y_col = None
        self.performance_actual = None
        self.performance_pred = None
        self.location = ()
        self.dataframe = None
        self.weather_data = None
        self.data = None
        self.frequency = 'hourly'
        self.report = {}
        if data is not None:
            if isinstance(data, pd.DataFrame):
                self.dataframe = data
            if not isinstance(data, list):
                data = [data]
            self.data = data
        for key in kwargs:
            self.__setattr__(key, kwargs[key])

        self.set_time_frames()

    def set_from_df(self, df, Y_col, X_col):
        self.Y = Var(df[Y_col])
        self.X = Var(df[X_col])

    def set_time_frames(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        time_frames = {}
        min_date, max_date = None, None
        if self.data is not None:
            # steps through all DataFrames in data argument (should be list) and sets the whole class's min,
            # max dates based on the widest range of their timeseries indices.
            for df in self.data:
                if not isinstance(df, pd.DataFrame):
                    raise ('In order to set time frames, all data passed into model.data must be type pd.DataFrame '
                           'with Timeseries index.')
                df.sort_index(inplace=True)
                if min_date is None and max_date is None:
                    min_date, max_date = df.index[0], df.index[-1]
                else:
                    this_min, this_max = df.index[0], df.index[-1]
                    if this_min < min_date:
                        min_date = this_min
                    if this_max > max_date:
                        max_date = this_max
                total = TimeFrame((min_date, max_date))
                time_frames.update({'total': total})
        if "baseline" in kwargs.keys():
            str_ = kwargs['baseline']
            baseline = TimeFrame(str_)
            time_frames.update({'baseline': baseline})
            min_date, max_date = baseline.tuple[0], baseline.tuple[1]
        if "performance" in kwargs.keys():
            pass #ToDo: refactor above lines and repeat for performance and report
        if "report" in kwargs.keys():
            pass
        if None in [min_date, max_date]:
            print('!! No time frames passed. Need any of: baseline, performance, or report kwargs. This model '
                  'instance still needs time frames to be set.')
        # for key, value in time_frames.items():
        #     start, end = value.tuple[0], value.tuple[1]
        #     if start < min_date:
        #         min_date = start
        #     if end > max_date:
        #         max_date = end
        # total = TimeFrame((min_date, max_date))
        # time_frames.update({'total': total})
        self.time_frames = time_frames

    def join_weather_data(self, location=None, time_frame=None):
        """Retrieves weather data from web service, joins it with self.dataframe, and returns the base series as
        self.weather_data.

        :param location: tuple in format (lat, long)
        :param time_frame: instance of TimeFrame object (see class)
        :return: n/a
        """
        if location is None:
            location = self.location
        if location is None:
            raise Exception('Must pass lat, long location tuple into get_weather_data, either by setting it prior as '
                            'a class attribute, or passing it explicitly as an argument.')
        if time_frame is None:
            time_frame = self.time_frames['total'].tuple
            #ToDo: raise error if no index
        s_weather = open_meteo_get(location, time_frame)
        self.weather_data = s_weather
        df = self.dataframe.resample('h').mean()
        df = pd.concat([df, s_weather], axis=1)
        df.dropna(inplace=True)
        self.dataframe = df

    def set_balance_point(self, cooling=None, heating=None):
        s = balance_point_transform(self.X['temp'], cooling)
        self.X.train = self.X.copy()
        self.X.train['temp_bp'] = s
        self.X.test['temp_bp'] = s

    def clear_balance_points(self):
        if 'temp_bp' in self.X.columns:
            self.X.drop(columns='temp_bp', inplace=True)
        try:
            if 'temp_bp' in self.X.train.columns:
                self.X.train.drop(columns='temp_bp', inplace=True)
        except AttributeError:
            pass
        self.bp_cooling = None

    def train(self):
        """Setting train and test arrays the same is acceptable for models that don't need to split train and test
        sets, feature_engineering.g. acceptable for ordinary least squares linear regression. However, distinction between train and
        test sets can apply if, for example, you want to exclude particular date ranges from the training set but
        not the test set.

        :return:
        """
        if self.X.train is None:
            try:
                before, after = self.time_frames['baseline'].tuple[0], self.time_frames['baseline'].tuple[1]
                X = self.X.data.truncate(before=before, after=after)
                Y = self.Y.data.truncate(before=before, after=after)
            except KeyError:
                msg = f"!! No baseline period in time_frames attribute. Model will be trained on entire length of " \
                      f"time in self.data attributes. \n Model object: {self} \n"
                print(msg)
                X, Y = self.X.data, self.Y.data
            # ToDo: prune X or Y to ensure they're same length
            self.X.train, self.Y.train = X, Y
        else:
            X, Y = self.X.train, self.Y.train
        reg = LinearRegression().fit(X, Y)
        y_pred = reg.predict(X)
        self.y.train = pd.Series(y_pred, index=Y.index, name='predicted')

        # save regressor object as model attribute for making predictions later
        self.reg = reg

        # self.y.test = pd.Series(data=y_pred, index=Y.index, name='predicted')
        # self.Y.test, self.Y.train = Y, Y
        # self.X.test, self.X.train = X, X

        # score training set metrics
        self.score(on='train')

    def test(self):
        """

        :return:
        """
        if self.reg is None: # or self.clf is None:
            msg = 'Model is being asked to test and has no reg or clf attribute. Need to train model before testing.'
            raise Exception(msg)
        if self.X.test is None:
            self.X.test = self.X.data
        y_pred = self.reg.predict(self.X.test)
        self.y.test = pd.Series(y_pred, index=self.X.test.index, name='predicted')
        self.Y.test = self.Y.data
        #ToDo: improve this at all with indexing? Train-test-split to be addressed in child classes that use it

    def predict(self, time_frame):
        """

        :param time_frame:
        :return:
        """
        before, after = time_frame.tuple[0], time_frame.tuple[1]
        if self.reg is None: # or self.clf is None:
            msg = 'Model is being asked to test and has no reg or clf attribute. Need to train model before testing.'
            raise Exception(msg)
        if self.X.pred is None:
            self.X.pred = self.X.data.truncate(before, after)
            if len(self.X.pred) == 0:
                msg = f'Timeseries data between {before} and {after} not found in self.X.data. Please run a method ' \
                      f'to fill self.X.data with data for this time period.'
                raise AttributeError(msg)
            self.Y.pred = self.Y.data.truncate(before, after)
            #ToDO: error-checking to make sure the lengths of X and Y are the same here
        y_pred = self.reg.predict(self.X.pred)
        self.y.pred = pd.Series(y_pred, index=self.X.pred.index, name='predicted')

    def score(self, on='test'):
        """

        :return:
        """
        Y, y = None, None
        if on == 'train':
            Y, y = self.Y.train, self.y.train
        elif on == 'test':
            Y, y = self.Y.test, self.y.test
        if self.frequency == 'hourly':
            self.scores[f'{on}_set_hourly'] = Scores()
            rsq = r2_score(Y, y)
            self.scores[f'{on}_set_hourly'].rsq = rsq
            mse = mean_squared_error(Y, y)
            cvrmse = np.sqrt(mse) / np.mean(Y)  #ToDo: should the divisor be absolute value? how to handle Y with pos
            # & neg values
            self.scores[f'{on}_set_hourly'].cvrmse = cvrmse
            self.scores[f'{on}_set_hourly'].ndbe = (Y.sum() - y.sum()) / Y.sum()
        Y_daily = Y.resample('d').mean().dropna()
        y_daily = y.resample('d').mean().dropna()
        self.scores[f'{on}_set_daily'] = Scores()
        rsq_daily = r2_score(Y_daily, y_daily)
        self.scores[f'{on}_set_daily'].rsq = rsq_daily
        mse_daily = mean_squared_error(Y_daily, y_daily)
        cvrmse_daily = np.sqrt(mse_daily) / np.mean(y_daily)
        self.scores[f'{on}_set_daily'].cvrmse = cvrmse_daily
        self.scores[f'{on}_set_daily'].ndbe = (Y_daily.sum() - y_daily.sum()) / Y_daily.sum()

    def prediction_metrics(self):
        """

        @return:
        """
        F = self.performance_actual / self.performance_pred
        t = 1
        n = len(self.Y.train)  # ToDo: check if this holds for models requiring train/test split
        m = len(self.y.pred)

        U = t * (1.26 * self.scores.cvrmse / F * np.sqrt((n + 2) / (n * m)))
        #ToDo: check this in general

        self.scores.savings_uncertainty = U
        self.scores.fsu = U / (self.energy_savings)

    def reporting_metrics(self, granularity='daily'):
        """

        It's important to remember that sums should be calculated not using .sum() but as multiplying the .mean() by
        the length of whatever array. This is to more properly represent sums over intervals where there are data
        gaps or "outages"! NaNs are handled variously in this code and generally do not raise errors.

        :param granularity: (str) - 'hourly' or 'daily'
        :return:
        """
        if not granularity in ['hourly', 'daily']:
            msg = "Granularity argument must be either 'hourly' or 'daily.'"
            raise Exception(msg)
        self.report = {} #ToDo: get rid of this
        self.report['y_predicted'] = safe_sum(self.y.pred)
        self.report['Y_actual'] = safe_sum(self.Y.pred)
        self.report['reduction'] = self.report['Y_actual'] - self.report['y_predicted']
        self.report['pct_reduction'] = self.report['reduction'] / self.report['y_predicted']
        self.report['trainset_daily_Rsq'] = self.scores[f'train_set_{granularity}'].rsq
        self.report['trainset_daily_CVRMSE'] = self.scores[f'train_set_{granularity}'].cvrmse


    def scatterplot(self,
                    x='actual',
                    y='predicted',
                    on='train',
                    alpha=.25):
        try:
            df = self.dataframe[[x, y]]
        except KeyError:
            if on == 'train':
                Y, y = self.Y.train, self.y.train
            elif on == 'test':
                Y, y = self.Y.test, self.y.test
            df = pd.concat([Y, y], axis=1)
            df.columns = ['actual', 'predicted']
        ax = df.plot.scatter(x='actual', y='predicted', alpha=alpha, grid=True)
        plt.axline((0,0), slope=1, linestyle='--', color='gray')
        plt.show()

    def timeplot(self,
                 x='actual',
                 y='predicted',
                 weather_data=None,
                 on='train',
                 alpha=.9):
        """

        :param x:
        :param y:
        :param weather_data: (pd.Series) if not None, must be a Series with a name
        :param on:
        :param alpha:
        :return:
        """
        try:
            df = self.dataframe[[x, y]]
        except KeyError:
            if on == 'train':
                Y, y = self.Y.train, self.y.train
            elif on == 'test':
                Y, y = self.Y.test, self.y.test
            df = pd.concat([Y, y], axis=1)
            df.columns = ['actual', 'predicted']
        if weather_data is not None:
            df = pd.concat([df, weather_data], axis=1)
            df.plot(alpha=alpha, grid=True, secondary_y=weather_data.name)
        else:
            df.plot(alpha=alpha, grid=True)

    def dayplot(self,
                x='actual',
                y='predicted',
                on='train',
                weather_data=None
                ):
        try:
            df = self.dataframe[[x, y]]
        except KeyError:
            if on == 'train':
                Y, y = self.Y.train, self.y.train
            elif on == 'test':
                Y, y = self.Y.test, self.y.test
            df = pd.concat([Y, y], axis=1)
            df.columns = ['actual', 'predicted']
        if weather_data is not None:
            df = pd.concat([df, weather_data], axis=1)
        df = df.resample('d').mean()
        df.index = df.index.date
        if weather_data is not None:
            df.plot.bar(rot=90, grid=True, secondary_y=weather_data.name)
        else:
            df.plot.bar(rot=90, grid=True)


    def check_zeroes(self):
        '''A function for checking % dependent variable zeros found in the performance period against same % of
        performance period, and flag if substantially different, or if either is substantially high. May indicate
        shortage of data, prolonged shutdown, etc.

        @return:
        '''
        # ToDo

    def check_sticky(self):
        return

    def normalize(self):
        # function to reg.predict onto weather normalization data set
        pass

    def add_exceptions(self, holidays=[], exceptions=[]):
        '''Holidays will become an added binary factor, and exceptions will be dropped from training set.

        :param holidays:
        :param exceptions:
        :return:
        '''
        from src.config import holidays_list, exceptions_list
        holidays, exceptions = holidays_list, exceptions_list
        X, Y = self.X.data, self.Y.data
        X['holiday'] = 0.0
        for holiday in holidays:
            X['holiday'][holiday[0]:holiday[1]] = 1.0
        for exception in exceptions:
            X = X[(X.index < exception[0]) | (X.index > exception[1])]
            Y = Y[(Y.index < exception[0]) | (Y.index > exception[1])]
        self.X.data, self.Y.data = X, Y

class TOWT(Model):
    """Class for performing time-of-week-and-temperature regression and storing results as class attributes.

    """
    #ToDo: add interactive number of temp coefficients (LBNL suggests 6).
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs
        )
        self.type = 'towt'
        self.temp_bins = None

    def bin_temps(self, num=6):
        """Per LBNL

        @param num:
        @return:
        """
        pass

    def add_TOWT_features(
            self,
            df_=None,
            bins=6,
            temp_col='temperature_2m'
    ):
        """Based on LBNL-4944E: Time of Week & Temperature Model outlined in Quantifying Changes in Building Electricity Use
        ... (2011, Mathieu et al). Given a time-series dataframe with outdoor air temperature column 'temp, this function
        returns dataframe with 168 columns with boolean (0 or 1) for each hour of the week, plus 6 binned temperature
        columns. Each column is intended to be a feature or independent variable of an ordinary least squares regression
        model to determine hourly energy usage as a function of time of week and temperature.

        @param df: (pandas.DataFrame) must have datetime index and at least column 'temp'
        @param n_bins: (int) number of temperature bins. Per LBNL recommendation, default is 6.
        @return: (pandas.DataFrame) augmented with features as new columns. original temperature column is dropped.
        """
        # add time of week features
        if df_ is None:
            df = self.dataframe
        else:
            df = df_
        TOW = df.index.weekday * 24 + df.index.hour
        TOWdf = pd.DataFrame(0, columns=TOW.unique(), index=df.index)
        for index, row in TOWdf.iterrows():
            tow = index.weekday() * 24 + index.hour
            TOWdf[tow].loc[index] = 1
        labels = ['h' + str(x) for x in TOWdf.columns]
        TOWdf.columns = labels
        df = pd.concat([df, TOWdf], axis=1)
        df.dropna(inplace=True)

        # break temp into bins
        if type(bins) == np.int:
            n_bins = bins
            min_temp, max_temp = np.floor(df[temp_col].min()), np.ceil(df[temp_col].max())
            #ToDo: need floor and ceiling arguments? Or can we not use floats, or are floats problematic?
            bin_size = (max_temp - min_temp) / (n_bins)
            temp_bins = np.arange(min_temp, max_temp, bin_size)
            temp_bins = list(np.append(temp_bins, max_temp))
            labels, labels_index = TOxT_column_labels(n_bins)
            df['temp_bin'] = pd.cut(df[temp_col], bins, labels=labels_index)
            self.temp_bins = temp_bins
        elif bins == 'from train':
            # This handles cases where the range of test data may exceed range of train data.
            temp_bins = self.temp_bins
            n_bins = len(temp_bins) - 1
            labels, labels_index = self.TOWT_column_labels(n_bins)
            old_min_temp, old_max_temp = temp_bins[0], temp_bins[-1]
            min_temp = np.floor(df[temp_col].min())
            #ToDo: need floor and ceiling arguments? Or can we not use floats, or are floats problematic?
            temp_bins[0] = min_temp
            df['temp_bin'] = pd.cut(df[temp_col], temp_bins, labels=labels_index)
            df.fillna(0, inplace=True)
        temp_df = pd.DataFrame(columns=labels_index, index=df.index)
        bin_deltas = list(np.array(temp_bins[1:]) - np.array(temp_bins[:-1]))
        for index, row in df.iterrows():
            colname = np.int(row['temp_bin'])
            left_cols = list(np.arange(0, colname, 1))
            write_list = bin_deltas[:colname]
            temp_row = temp_df.loc[index]
            temp_row.loc[left_cols] = write_list
            temp_df.loc[index] = temp_row
            bin_bottom = temp_bins[colname]
            temp_df[colname].loc[index] = row[temp_col] - bin_bottom
        if bins == 'from train':
            adj_amt = old_min_temp - min_temp
            temp_df[0] -= adj_amt

        temp_df.fillna(0, inplace=True)
        temp_df.columns = labels
        joined_df = pd.concat([df.drop(columns=[temp_col, 'temp_bin']), temp_df], axis=1)
        dense_df = joined_df.dropna()
        na_df = df[~df.index.isin(dense_df.index)]
        if len(na_df > 0):
            print(f'Dropped {len(na_df)} rows of NaN values from dataframe before storing X and Y values.')
            print(f'Dropped dataframe: \n {na_df}')
        self.Y.data = dense_df[self.Y_col]
        self.X.data = dense_df.drop(columns=self.Y_col)

        return joined_df

    def truncate_baseline(self, before=None, after=None):
        """This is listed under the TOWT class here because not all models will use train and test sets as identical. Many model types should not.

        :param before:
        :param after:
        :return:
        """
        #ToDo: must happen before train_test_split, right?
        if before is None:
            before = self.train_start
        if after is None:
            after = self.train_end
        if self.X.train is None:
            self.X.train = self.X.data
        if self.Y.train is None:
            self.Y.train = self.Y.data
        X_train = self.X.train.truncate(before=before, after=after)
        Y_train = self.Y.train.truncate(before=before, after=after)
        self.X.train = X_train
        self.Y.train = Y_train

    def run(self, on='train', start=None, end=None):
        X, bins = None, None
        if on == 'train':
            X, Y = self.X.train, self.Y.train
            bins = 6 #ToDo: call this out
        elif on == 'test':
            X, Y = self.X.test, self.Y.test
            bins = 'from train'
        elif on == 'predict': #ToDo: add hard stop so you cannot cast prediction onto baseline period
            X = self.X.truncate(start, end)
            Y = self.Y.truncate(start, end)
            bins = 'from train'
        elif on == 'normalize':
            X = self.X.norm
            bins = 'from train'
        X = self.add_TOWT_features(X, bins=bins)
        if on in {'test', 'predict', 'normalize'}:
            X = X[self.X.train.columns]  # ToDo: raise error if perf period too short to have all week-hour factors
        if on == 'train':
            reg = LinearRegression().fit(X, Y)
        else:
            reg = self.reg
        #ToDO: below won't work without add_TOW features first eh? maybe do some error catching
        #ToDo: also add exception or notification for truncating baseline
        y = reg.predict(X)
        # y = pd.DataFrame(y, index=X.index, columns=['predicted'])
        y = pd.Series(y, index=X.index, name='kW modeled')
        y[y < 0] = 0
        if on == 'train':
            self.X.train, self.y.train, self.Y.train = X, y, Y
            self.reg = reg
        elif on == 'test':
            self.X.test, self.y.test, self.Y.test = X, y, Y
        elif on == 'predict':
            #ToDo: refactor / break out under a new function called prediction metrics or something
            self.X.pred, self.y.pred, self.Y.pred = X, y, Y
            self.kWh_performance_actual = Y.sum()
            self.kWh_performance_pred = y.sum()
            self.energy_savings = self.kWh_performance_pred - self.kWh_performance_actual
            self.pct_savings = 100 * self.energy_savings / self.kWh_performance_actual
            # self.annualized_savings = (y.mean() - Y.mean())*8760 #Todo: not how you do this
        elif on == 'normalize':
            self.y.norm = y

    def predict(self, time_frame):
        """

        :param time_frame:
        :return:
        """
        before, after = time_frame.tuple[0], time_frame.tuple[1]
        if self.reg is None: # or self.clf is None:
            msg = 'Model is being asked to test and has no reg or clf attribute. Need to train model before testing.'
            raise Exception(msg)
        if self.X.pred is None:
            self.X.pred = self.X.data.truncate(before, after)
            if len(self.X.pred) == 0:
                print(f'Requesting additional weather data from open meteo from {before} to {after} for model '
                      f'prediction.')
                s_weather = open_meteo_get(self.location, time_frame)
                df_weather = pd.DataFrame(s_weather)
                df = self.add_TOWT_features(df_weather)
                self.X.pred = df
        if self.Y.pred is not None:
            self.Y.pred = self.Y.data.truncate(before, after)
            #ToDO: error-checking to make sure the lengths of X and Y are the same here
        y_pred = self.reg.predict(self.X.pred)
        self.y.pred = pd.Series(y_pred, index=self.X.pred.index, name='predicted')

    def predict_recursive(self, X=None, Y=None):
        pass

class TODT(Model):
    """

    """
    # def __init__(
    #         self,
    #         df,
    #         Y_col='kW',
    #         X_col='OAT',
    #         weekend=False,
    # ):
    #     self.temp_bins = None
    #     self.data = None
    #     self.weekend = weekend,
    #     self.features = False
    #     if isinstance(df, pd.DataFrame):
    #         super().__init__(dataset=df)
    #         try:
    #             self.set_from_df(df, Y_col, X_col)
    #         except KeyError:
    #             pass
    #         self.Y_col, self.X_col = Y_col, X_col
    #     else:
    #         raise Exception('TODT model requires pandas DataFrame as argument.')
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs
        )
        self.type = 'todt'
        self.temp_bins = None
        if 'weekend' in kwargs:
            self.weekend = kwargs['weekend']
        else:
            self.weekend = False

    def add_TODT_features(
            self,
            df_=None,
            bins=6,
            temp_col='temperature_2m'
    ):
        """See TODT class method. Similar but only uses 24 hour-wise time factors per day rather than 168 per week.

        @param df: (pandas.DataFrame) must have datetime index and at least column 'temp'
        @param n_bins: (int) number of temperature bins. Per LBNL recommendation, default is 6.
        @return: (pandas.DataFrame) augmented with features as new columns. original temperature column is dropped.
        """
        # add time of week features
        if df_ is None:
            df = self.dataframe
        else:
            df = df_
        TOD = df.index.hour
        TODdf = pd.DataFrame(0.0, columns=TOD.unique(), index=df.index)
        for index, row in TODdf.iterrows():
            tod = index.hour
            TODdf[tod].loc[index] = 1.0
        labels = ['h' + str(x) for x in TODdf.columns]
        TODdf.columns = labels
        df = df.join(TODdf)
        df.dropna(inplace=True)
        if bins == None:
            return df

        # break temp into bins
        if type(bins) == np.int:
            n_bins = bins
            min_temp, max_temp = np.floor(df[temp_col].min()), np.ceil(df[temp_col].max())
            #ToDo: need floor and ceiling arguments? Or can we not use floats, or are floats problematic?
            bin_size = (max_temp - min_temp) / (n_bins)
            temp_bins = np.arange(min_temp, max_temp, bin_size)
            temp_bins = list(np.append(temp_bins, max_temp))
            labels, labels_index = TOxT_column_labels(n_bins)
            df['temp_bin'] = pd.cut(df[temp_col], bins, labels=labels_index)
            self.temp_bins = temp_bins
        elif bins == 'from train':
            # This handles cases where the range of test data may exceed range of train data.
            temp_bins = self.temp_bins
            n_bins = len(temp_bins) - 1
            labels, labels_index = TOxT_column_labels(n_bins)
            old_min_temp, old_max_temp = temp_bins[0], temp_bins[-1]
            min_temp = np.floor(df[temp_col].min())
            #ToDo: need floor and ceiling arguments? Or can we not use floats, or are floats problematic?
            temp_bins[0] = min_temp
            df['temp_bin'] = pd.cut(df[temp_col], temp_bins, labels=labels_index)
            df.fillna(0, inplace=True)
        temp_df = pd.DataFrame(columns=labels_index, index=df.index)
        bin_deltas = list(np.array(temp_bins[1:]) - np.array(temp_bins[:-1]))
        for index, row in df.iterrows():
            colname = np.int(row['temp_bin'])
            left_cols = list(np.arange(0, colname, 1))
            write_list = bin_deltas[:colname]
            temp_row = temp_df.loc[index]
            temp_row.loc[left_cols] = write_list
            temp_df.loc[index] = temp_row
            bin_bottom = temp_bins[colname]
            temp_df[colname].loc[index] = row[temp_col] - bin_bottom
        temp_df[0] = pd.to_numeric(temp_df[0])
        if bins == 'from train':
            adj_amt = old_min_temp - min_temp
            temp_df[0] -= adj_amt

        temp_df.fillna(0, inplace=True) #ToDo: not safe?
        temp_df.columns = labels
        joined_df = pd.concat([df.drop(columns=[temp_col, 'temp_bin']), temp_df], axis=1)
        if self.weekend == True:
            joined_df['weekend'] = 0.0
            joined_df['weekend'][joined_df.index.dayofweek > 4] = 1.0
        self.features = True
        dense_df = joined_df.dropna()
        na_df = df[~df.index.isin(dense_df.index)]
        if len(na_df > 0):
            print(f'Dropped {len(na_df)} rows of NaN values from dataframe before storing X and Y.')
            print(f'Dropped dataframe: \n {na_df}')
        try:
            self.Y.data = dense_df[self.Y_col]
            self.X.data = dense_df.drop(columns=self.Y_col)
        except KeyError:
            # This catches the case where the Y column was not passed into the model data (and allows function to
            # proceed)
            pass
        if df_ is None:
            self.dataframe = joined_df
        else:
            return joined_df

    def predict(self, time_frame):
        """Includes methods for

        :param time_frame:
        :return:
        """
        before, after = time_frame.tuple[0], time_frame.tuple[1]
        if self.reg is None: # or self.clf is None:
            msg = 'Model is being asked to test and has no reg or clf attribute. Need to train model before testing.'
            raise Exception(msg)
        if self.X.pred is None:
            self.X.pred = self.X.data.truncate(before, after)
            if len(self.X.pred) == 0:
                print(f'Requesting additional weather data from open meteo from {before} to {after} for model '
                      f'prediction. \n')
                s_weather = open_meteo_get(self.location, time_frame.tuple)
                df_weather = pd.DataFrame(s_weather)
                df = self.add_TODT_features(df_weather)
                self.X.pred = df
        if self.Y.pred is not None:
            Y_pred = self.Y.pred.truncate(before, after)
        # resample Y to match X (or default to hourly if there is trouble)
        freq = pd.infer_freq(self.X.pred.index)
        if not isinstance(freq, str):
            freq = 'H'
        self.Y.pred = Y_pred.resample(freq).mean()
        # error-checking to make sure the lengths of X and Y are the same here

        # ensure column order of X.pred aligns with column order from the training set
        self.X.pred = self.X.pred[list(self.reg.feature_names_in_)]
        y_pred = self.reg.predict(self.X.pred)
        self.y.pred = pd.Series(y_pred, index=self.X.pred.index, name='predicted')

        #ToDo: refactor this to be packaged for both TODT and TOWT

class SimpleOLS(Model):
    """

    """
    def __init__(self, *args, **kwargs):
        try:
            if args:
                if type(args[0]) is Model:
                    self.__dict__ = args[0].__dict__.copy()
            else: #ToDo: clean this up; it is redundant with else clause a few lines below
                super().__init__(
                    *args, **kwargs
                )
        except IndexError:
            super().__init__(
                *args, **kwargs
            )
        self.type = 'simple_ols'

    def run(self, on='train', start=None, end=None):
        if on == 'train':
            if start is None and end is None:
                X = pd.DataFrame(self.X)
                Y = pd.DataFrame(self.Y)
                reg = LinearRegression().fit(X, Y)
                y = reg.predict(X)
                y = pd.DataFrame(y, index=x.index).rename(columns={0: 'kWh_predicted'})
                self.X.train, self.y.train, self.Y.train = x, y, Y
                self.Y.test, self.y.test = self.Y.train, self.y.train
                self.reg = reg
        elif on == 'predict':
            x_pred = pd.DataFrame(self.X.pred)
            y = self.reg.predict(x_pred)
            self.y.pred = pd.DataFrame(y, index=x_pred.index)

class TreeTODT(Model):
    """

    """
    def __init__(self, *args, **kwargs):
        try:
            if args:
                if type(args[0]) is Model:
                    self.__dict__ = args[0].__dict__.copy()
                else: #ToDo: clean this up; it is redundant with else clause a few lines below
                    super().__init__(
                        *args, **kwargs
                    )
        except IndexError:
            super().__init__(
                *args, **kwargs
            )
        self.type = 'tree_todt'
        self.temp_bins = None
        self.reg_colnames = None

    def TOWT_column_labels(self, n_bins):
        """helper function

        :param n_bins:
        :return:
        """
        labels = np.arange(1, n_bins + 1)
        labels_index = [np.int(x - 1) for x in labels]
        labels = ['t' + str(x) for x in labels]
        return labels, labels_index

    def add_TODT_features(self, df=None, time_bins=144, look_back_hrs=1, temp_bins=6):
        """Loosely based on LBNL TOWT model, except uses Decision Tree Regression rather than

        @param df: (pandas.DataFrame) must have datetime index and at least column 'temp'
        @param n_bins: (int) number of temperature bins. Per LBNL recommendation, default is 6.
        @return: (pandas.DataFrame) augmented with features as new columns. original temperature column is dropped.
        """
        # add time of day features
        if df is None:
            df = self.df_trimmed
        df['TOD'] = df.index.hour

        # break temp into bins
        labels_index = []
        if type(temp_bins) == np.int:
            n_bins = temp_bins
            min_temp, max_temp = np.floor(df['temp'].min()), np.ceil(df['temp'].max())
            # ToDo: need floor and ceiling arguments? Or can we not use floats, or are floats problematic?
            bin_size = (max_temp - min_temp) / (n_bins)
            temp_bins = np.arange(min_temp, max_temp, bin_size)
            temp_bins = list(np.append(temp_bins, max_temp))
            labels, labels_index = self.TOWT_column_labels(n_bins)
            df['temp_bin'] = pd.cut(df['temp'], temp_bins, labels=labels_index)
            self.temp_bins = temp_bins
        elif temp_bins == 'from train':
            # This handles cases where the range of test data may exceed range of train data.
            temp_bins = self.temp_bins
            n_bins = len(temp_bins) - 1
            old_min_temp, old_max_temp = temp_bins[0], temp_bins[-1]
            min_temp = np.floor(df['temp'].min())
            # ToDo: need floor and ceiling arguments? Or can we not use floats, or are floats problematic?
            # ToDo: the below line should only apply if the new bin min is LOWER than the old one.
            # temp_bins[0] = min_temp
            labels, labels_index = self.TOWT_column_labels(n_bins)
            df['temp_bin'] = pd.cut(df['temp'], temp_bins, labels=labels_index)
            # df.fillna(0, inplace=True)
        temp_df = pd.DataFrame(columns=labels_index, index=df.index)
        bin_deltas = list(np.array(temp_bins[1:]) - np.array(temp_bins[:-1]))
        for index, row in df.iterrows():
            colname = np.int(row['temp_bin'])
            left_cols = list(np.arange(0, colname, 1))
            write_list = bin_deltas[:colname]
            temp_row = temp_df.loc[index]
            temp_row.loc[left_cols] = write_list
            temp_df.loc[index] = temp_row
            bin_bottom = temp_bins[colname]
            temp_df[colname].loc[index] = row['temp'] - bin_bottom
# ToDo: this doesn't get triggered because we overwrote temp_bins!
        if temp_bins == 'from train':
            adj_amt = old_min_temp - min_temp
            temp_df[0] -= adj_amt
        temp_df.fillna(0, inplace=True)
        temp_df.columns = labels
        joined_df = pd.concat([df.drop(columns=['temp', 'temp_bin']), temp_df], axis=1)
        self.df_joined = joined_df

        return joined_df

    def add_shifted_features(self, colname, df=None):
        '''

        :return:
        '''
        if df is None:
            df = self.df_joined
        shifted_colname = colname + '_prior'
        rolling_colname = colname + '_rolling'
        df[shifted_colname] = df[colname].shift(1)
        df[shifted_colname + '2'] = df[colname].shift(2)
        df[shifted_colname + '3'] = df[colname].shift(3)
        df[shifted_colname + '4'] = df[colname].shift(4)
        df['diff1'] = df[colname] - df[shifted_colname]
        df['diff2'] = df[shifted_colname] - df[shifted_colname + '2']
        df['diff3'] = df[shifted_colname + '2'] - df[shifted_colname + '3']
        df['diff4'] = df[shifted_colname + '3'] - df[shifted_colname + '4']
        df[rolling_colname] = df[colname].rolling(6).mean()
        self.df_joined = df.copy()
        # drop nans resulting from the shifts in the x and Y properties only.
        df.dropna(inplace=True)
        self.X = df.drop(columns=colname)
        self.Y = df[colname]

        return df

    def train_test_split(self):
        '''

        :return:
        '''
        test_size = .5
        x_train, x_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=test_size)
        #ToDo: double check above line is safe as far as splitting both x and Y.
        self.X.train, self.X.test, self.Y.train, self.Y.test = x_train, x_test, Y_train, Y_test

    def ensemble_tree(self, run='train', tree_feature_colnames=None):
        '''

        :return:
        '''
        if tree_feature_colnames is None:
            tree_feature_colnames = [
                'TOD',
                # 'HP_outdoor_prior',
                # 'HP_outdoor_prior2',
                # 'HP_outdoor_prior3',
                'diff1',
                'diff2',
                'diff3',
                'diff4',
                # 'HP_outdoor_rolling'
            ]
        Xa = self.X.drop(columns=tree_feature_colnames)
        reg = LinearRegression().fit(xa, self.Y)
        ya = reg.predict(xa)
        ya = pd.DataFrame(ya, index=self.X.index)
        Xb = ya.join(self.X[tree_feature_colnames])
        test_size = .5
        # x.train, x.test, Y.train, Y.test = train_test_split(xb, self.Y, test_size=test_size)
        # treereg = DecisionTreeRegressor().fit(x.train, self.Y.train)
        gbreg = HistGradientBoostingRegressor().fit(Xb, self.Y)
        yb = gbreg.predict(Xb)
        self.reg = gbreg
        self.reg_colnames = tree_feature_colnames
        self.y.test = pd.DataFrame(yb, index=Xb.index)
        if run == 'predict':
            pass
            # Xa_future

    def gradient_boost(self):
        gb_feature_colnames = [
            'TOD',
            'HP_outdoor_prior',
            'HP_outdoor_prior2',
            'HP_outdoor_prior3',
            'diff1',
            'diff2',
            'diff3',
            'diff4',
            'HP_outdoor_rolling'
        ]
        X = self.X[gb_feature_colnames]
        categorical_columns = [
            'TOD',
            # 'diff1',
            # 'diff2',
            # 'diff3',
            # 'diff4'
        ]
        ordinal_encoder = OrdinalEncoder()
        gbrt_pipeline = make_pipeline(
            ColumnTransformer(
                transformers=[
                    ('categorical', ordinal_encoder, categorical_columns),
                ],
                remainder='passthrough', verbose_feature_names_out=False,
            ),
            HistGradientBoostingRegressor(
                categorical_features=categorical_columns,
            ),
        )

        # gbrt_pipeline.set_output(transform='pandas')

        cv_results = cross_validate(
            gbrt_pipeline, x, self.Y
        )
        y = gbrt_pipeline.predict(x)
        y = pd.DataFrame(y)


    def run(self, on='train', start=None, end=None):
        X, bins = None, None
        if on == 'train':
            X, Y = self.X.train, self.Y.train
            bins = 6 #ToDo: call this out
        elif on == 'test':
            X, Y = self.X.test, self.Y.test
            bins = 'from train'
        elif on == 'predict': #ToDo: add hard stop so you cannot cast prediction onto baseline period
            X = self.X.truncate(start, end)
            Y = self.Y.truncate(start, end)
            bins = 'from train'
        elif on == 'normalize':
            X = self.X.norm
            bins = 'from train'
        if on in {'test', 'predict', 'normalize'}:
            X = X[self.X.train.columns]  # ToDo: raise error if perf period too short to have all week-hour factors
        if on == 'train':
            self.ensemble_tree()
            # self.gradient_boost()
        else:
            reg = self.reg
        #ToDO: below won't work without add_TOW features first eh? maybe do some error catching
        #ToDo: also add exception or notification for truncating baseline
        y = reg.predict(x)
        # y = pd.DataFrame(y, index=X.index, columns=['predicted'])
        y = pd.Series(y, index=x.index, name='kW modeled')
        y[y < 0] = 0
        if on == 'train':
            self.X.train, self.y.train, self.Y.train = x, y, Y
            self.reg = reg
        elif on == 'test':
            self.X.test, self.y.test, self.Y.test = x, y, Y
        elif on == 'predict':
            #ToDo: refactor / break out under a new function called prediction metrics or something
            self.X.pred, self.y.pred, self.Y.pred = x, y, Y
            self.kWh_performance_actual = Y.sum()
            self.kWh_performance_pred = y.sum()
            self.energy_savings = self.kWh_performance_pred - self.kWh_performance_actual
            self.pct_savings = 100 * self.energy_savings / self.kWh_performance_actual
            # self.annualized_savings = (y.mean() - Y.mean())*8760 #Todo: not how you do this
        elif on == 'normalize':
            self.y.norm = y