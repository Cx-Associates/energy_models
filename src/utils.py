import sys, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate
import matplotlib as mpl
import matplotlib.pyplot as plt
from .open_meteo import open_meteo_get

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
    """

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
    """A child class of DataSet with attributes to use in modeling.

    """
    def __init__(self, data=None, **kwargs):
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
        """

        :param location:
        :return:
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
        sets, e.g. acceptable for ordinary least squares linear regression.

        :return:
        """
        before, after = self.time_frames['baseline'].tuple[0], self.time_frames['baseline'].tuple[1]
        X = self.X.data.truncate(before=before, after=after)
        Y = self.Y.data.truncate(before=before, after=after)  #ToDo: figure out how to handle
        reg = LinearRegression().fit(X, Y)
        y_pred = reg.predict(X)
        self.reg = reg
        self.y.test = pd.Series(data=y_pred, index=Y.index, name='predicted')
        self.Y.test, self.Y.train = Y, Y
        self.X.test, self.X.train = X, X
        self.score()

    def score(self):
        """

        :return:
        """
        if self.frequency == 'hourly':
            self.scores['hourly'] = Scores()
            rsq = r2_score(self.Y.test, self.y.test)
            self.scores['hourly'].rsq = rsq
            mse = mean_squared_error(self.Y.test, self.y.test)
            cvrmse = np.sqrt(mse) / np.mean(self.Y.test)
            self.scores['hourly'].cvrmse = cvrmse
            self.scores['hourly'].ndbe = (self.Y.test.sum() - self.y.test.sum()) / self.Y.test.sum()
        Ytest_daily = self.Y.test.resample('d').mean()
        ytest_daily = self.y.test.resample('d').mean()
        self.scores['daily'] = Scores()
        rsq_daily = r2_score(Ytest_daily, ytest_daily)
        self.scores['daily'].rsq = rsq_daily
        mse_daily = mean_squared_error(Ytest_daily, ytest_daily)
        cvrmse_daily = np.sqrt(mse_daily) / np.mean(ytest_daily)
        self.scores['daily'].cvrmse = cvrmse_daily
        self.scores['daily'].ndbe = (Ytest_daily.sum() - ytest_daily.sum()) / Ytest_daily.sum()

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

    def scatterplot(self,
                    x='actual',
                    y='predicted',
                    alpha=.25):
        try:
            df_scatter = self.dataframe[[x, y]]
        except KeyError:
            df_scatter = pd.concat([self.Y.test, self.y.test], axis=1)
            df_scatter.columns = ['actual', 'predicted']
        ax = df_scatter.plot.scatter(x='actual', y='predicted', alpha=alpha, grid=True)
        plt.axline((0,0), slope=1, linestyle='--', color='gray')
        plt.show()

    def timeplot(self,
                 x='actual',
                 y='predicted',
                 weather=False,
                 alpha=.9):
        try:
            df = self.dataframe[[x, y]]
        except KeyError:
            df = pd.concat([self.Y.test, self.y.test], axis=1)
            df.columns = ['actual', 'predicted']
        if weather == True:
            df['OAT'] = self.dataframe['temperature_2m']
            df.plot(alpha=alpha, grid=True, secondary_y='OAT')
        else:
            df.plot(alpha=alpha, grid=True)

    def dayplot(self,
                x='actual',
                y='predicted',
                weather=False
                ):
        try:
            df = self.dataframe[[x, y]]
        except KeyError:
            df = pd.concat([self.Y.test, self.y.test], axis=1)
            df.columns = ['actual', 'predicted']
        if weather == True:
            df['OAT'] = self.dataframe['temperature_2m']
        df = df.resample('d').mean()
        df.index = df.index.date
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

    def add_TOWT_features(self, df, bins=6, temp_col='temperature_2m'):
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
        X_train = self.X.truncate(before=before, after=after)
        Y_train = self.Y.truncate(before=before, after=after)
        self.X.train = X_train
        self.X.test = X_train
        self.Y.train = Y_train
        self.Y.test = Y_train

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
            temp_col=None
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
        if temp_col is None:
            temp_col = self.X_col
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
            print(f'Dropped {len(na_df)} rows of NaN values from dataframe before storing X and Y values.')
            print(f'Dropped dataframe: \n {na_df}')
        self.Y.data = dense_df[self.Y_col]
        self.X.data = dense_df.drop(columns=self.Y_col)
        if df_ is None:
            self.dataframe = joined_df
        else:
            return joined_df


    def add_exceptions(self, holidays=[], exceptions=[]):
        pass

    def test(self):
        pass

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