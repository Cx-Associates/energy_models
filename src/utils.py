import sys, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

sys.path.append('../src')

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


def add_TOWT_features(df, n_bins=6):
    """Based on LBNL-4944E: Time of Week & Temperature Model outlined in Quantifying Changes in Building Electricity Use
    ... (2011, Mathieu et al). Given a time-series dataframe with outdoor air temperature column 'temp, this function
    returns dataframe with 168 columns with boolean (0 or 1) for each hour of the week, plus 6 binned temperature
    columns. Each column is intended to be a feature or independent variable an ordinary least squares regression
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
    df = df.join(TOWdf)
    df.dropna(inplace=True)

    # break temp into bins
    min_temp, max_temp = np.floor(df['temp'].min()), np.ceil(df['temp'].max())
    bin_size = (max_temp - min_temp) / (n_bins)
    bins = np.arange(min_temp, max_temp, bin_size)
    bins = list(np.append(bins, max_temp))
    labels = np.arange(1, n_bins + 1)
    labels_index = [np.int(x - 1) for x in labels]
    labels = ['t' + str(x) for x in labels]
    df['temp_bin'] = pd.cut(df['temp'], bins, labels=labels_index)
    temp_df = pd.DataFrame(columns=labels_index, index=df.index)
    for index, row in df.iterrows():
        colname = np.int(row['temp_bin'])
        bin_bottom = bins[colname]
        temp_df[colname].loc[index] = row['temp'] - bin_bottom #ToDo: revisit with normalizing
    temp_df.fillna(0, inplace=True)
    temp_df.columns = labels
    joined_df = pd.concat([df.drop(columns=['temp', 'temp_bin']), temp_df], axis=1)
    return joined_df


class Dataset:
    """Generic dataset class comprising an energy time-series and a weather time-series.

    """
    #ToDO: update energy to use "energy" and not "kWh" in case MMBtu data from gas etc. is enabled

    def __init__(
            self,
            energy_filepath=None,
            weather_filepath=None,
            df=None
    ):
        self.name = None
        self.energy_series = None
        self.temperature_series = None
        self.normalized_temperature_series = None
        self.joined_df = None
        self.sparse_df = None
        self.display_start = None
        self.display_end = None

        if df is not None:
            pass
        else:
            df = pd.read_csv(energy_filepath, index_col=[0], parse_dates=True)
            # s_energy = df['kWh'].resample('h').mean()
            # ToDo: write function to auto-recognize frequency. instead, resampling to hourly below regardless
            s_energy = df['kWh']
            self.energy_series = s_energy.dropna()

            try:
                df = read_weather_data(weather_filepath, 'temp')
                s_temp = df['temp']
            except pd.errors.EmptyDataError: #ToDo: no idea why I'm getting this
                df = pd.read_csv(weather_filepath, index_col=[0], parse_dates=True)
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

        self.full_df = df
        df.dropna(inplace=True)
        self.trimmed_df = df

        self.temperature_series = df['temp']
        self.energy_series = df['kWh']

    def check_zeroes(self):
        return

    def check_sticky(self):
        return


class Modelset(Dataset):
    """A child class of DataSet with attributes to use in modeling.

    """

    def __init__(self, *args, x='temperature', y='energy', **kwargs):
        try:
            if type(args[0]) is Dataset:
                self.__dict__ = args[0].__dict__.copy()
            else: #ToDo: clean this up; it is redundant with else clause a few lines below
                super().__init__(
                    **kwargs
                )
        except IndexError:
            # super().__init__(
            #     args[0],
            #     args[1]
            # )
            super().__init__(
                **kwargs
            )
        self.baseline_start = None
        self.baseline_end = None
        self.bp_heating = None
        self.bp_cooling = None
        self.performance_start = None
        self.performance_end = None
        self.X = None
        self.X_train = None
        self.X_test = None
        self.X_pred = None
        self.X_norm = None
        self.Y = None
        self.Y_train = None
        self.Y_test = None
        self.Y_pred = None
        self.reg = None
        self.clf = None
        self.y = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_norm = None
        self.kWh_performance_actual = None
        self.kWh_performance_pred = None
        self.rsq = None
        self.cvrmse = None
        self.savings_uncertainty = None
        self.fsu = None

        if x == 'temperature':
            self.X = pd.DataFrame(self.temperature_series)
        if y == 'energy':
            self.Y = self.energy_series

    def set_balance_point(self, cooling=None, heating=None):
        s = balance_point_transform(self.X['temp'], cooling)
        self.X_train = self.X.copy()
        self.X_train['temp_bp'] = s
        self.X_test['temp_bp'] = s

    def clear_balance_points(self):
        if 'temp_bp' in self.X.columns:
            self.X.drop(columns='temp_bp', inplace=True)
        try:
            if 'temp_bp' in self.X_train.columns:
                self.X_train.drop(columns='temp_bp', inplace=True)
        except AttributeError:
            pass
        self.bp_cooling = None

    def truncate_baseline(self, before=None, after=None):
        #ToDo: must happen before train_test_split, right?
        X_train = self.X.truncate(before=before, after=after)
        Y_train = self.Y.truncate(before=before, after=after)
        self.X_train = X_train
        self.X_test = X_train
        self.Y_train = Y_train
        self.Y_test = Y_train

    def score(self):
        """

        :return:
        """
        rsq = r2_score(self.Y_test, self.y_test)
        self.rsq = rsq

        mse = mean_squared_error(self.Y_test, self.y_test)
        cvrmse = np.sqrt(mse) / np.mean(self.Y_test)
        self.cvrmse = cvrmse

        self.ndbe = (self.Y_test.sum() - self.y_test.sum()) / self.Y_test.sum()

    def prediction_metrics(self):
        """

        @return:
        """
        F = self.kWh_performance_actual / self.kWh_performance_pred
        t = 1
        n = len(self.Y_train)  # ToDo: check if this holds for models requiring train/test split
        m = len(self.y_pred)

        U = t * (1.26 * self.cvrmse / F * np.sqrt((n + 2) / (n * m)))
        #ToDo: check this in general

        self.savings_uncertainty = U
        self.fsu = U / (self.energy_savings)

    def plot_modelset(self):
        df_scatter = pd.concat([self.Y_train, self.y_test], axis=1)
        df_scatter.plot.scatter('kWh', 'predicted', alpha=.2)

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


class TOWT(Modelset):
    """Class for performing time-of-week-and-temperature regression and storing results as class attributes.

    """
    #ToDo: add interactive number of temp coefficients (LBNL suggests 6).
    def __init__(self, *args, **kwargs):
        try:
            if type(args[0]) is Modelset:
                self.__dict__ = args[0].__dict__.copy()
            else: #ToDo: clean this up; it is redundant with else clause a few lines below
                super().__init__(
                    **kwargs
                )
        except IndexError:
            # super().__init__(
            #     args[0],
            #     args[1]
            # )
            super().__init__(
                **kwargs
            )
        self.X_train, self.X_test = self.X.copy(), self.X.copy()
        self.Y_train, self.Y_test = self.Y.copy(), self.Y.copy()
        self.type = 'towt'

    def bin_temps(self, num=6):
        """Per LBNL

        @param num:
        @return:
        """
        pass

    def run(self, on='train', start=None, end=None):
        #ToDo: badly needs refacotring and rethikning on if the reindex is even necessary
        if on == 'train':
            X, Y = self.X_train, self.Y_train
            # Y.dropna(inplace=True)
            # X = X.reindex(Y.index)
            # X.dropna(inplace=True)
            # Y = Y.reindex(X.index)
        elif on == 'test':
            X, Y = self.X_test, self.Y_test
            # Y.dropna(inplace=True)
            # X = X.reindex(Y.index)
            # X.dropna(inplace=True)
            # Y = Y.reindex(X.index)
        elif on == 'predict': #ToDo: add hard stop so you cannot cast prediction onto baseline period
            X = self.X.truncate(start, end)
            Y = self.Y.truncate(start, end)
            # Y.dropna(inplace=True)
            # X = X.reindex(Y.index)
            # X.dropna(inplace=True)
            # Y = Y.reindex(X.index)
        elif on == 'normalize':
            X = self.X_norm
        X = add_TOWT_features(X)
        if on in {'test', 'predict'}:
            X = X[self.X_train.columns]  # ToDo: raise error if perf period too short to have all week-hour factors
        if self.bp_cooling is not None:
            X['temp'] = X.pop('temp_bp')
        if on == 'train':
            reg = LinearRegression().fit(X, Y)
        else:
            reg = self.reg
        #ToDO: below won't work without add_TOW features first eh? maybe do some error catching
        #ToDo: also add exception or notification for truncating baseline
        y = reg.predict(X)
        # y = pd.DataFrame(y, index=X.index, columns=['predicted'])
        y = pd.Series(y, index=X.index, name='kW modeled')
        if on == 'train':
            self.X_train, self.y_train, self.Y_train = X, y, Y
            self.reg = reg
        elif on == 'test':
            self.X_test, self.y_test, self.Y_test = X, y, Y
        elif on == 'predict':
            self.X_pred, self.y_pred, self.Y_pred = X, y, Y
            self.kWh_performance_actual = Y.sum()
            self.kWh_performance_pred = y.sum()
            self.energy_savings = self.kWh_performance_pred - self.kWh_performance_actual
            self.pct_savings = 100 * self.energy_savings / self.kWh_performance_actual
            # self.annualized_savings = (y.mean() - Y.mean())*8760 #Todo: not how you do this
        elif on == 'normalize':
            self.y_norm = y