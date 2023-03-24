import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from energy_models.src.utils import TOWT, Dataset, Modelset
from src.st_utils import make_TOWT
from src.data_config import dict_config

# for interactive plotting while debugging in PyCharm
plt.interactive(True)
mpl.use('TkAgg')

dataset_name = 'Devtown (Anywhere, USA)'

# directories
dir_here = os.path.abspath(os.path.dirname(__file__))
dir_grandparent = os.path.dirname(os.path.dirname(dir_here))
dir_data = os.path.join(dir_grandparent, 'data')

config_parameters_dict = dict_config[dataset_name]
config_parameters_dict['energy_filepath'] = os.path.join(dir_data, dict_config[dataset_name]['energy_data'])
config_parameters_dict['weather_filepath'] = os.path.join(dir_data, dict_config[dataset_name]['weather_data'])


model = TOWT(**config_parameters_dict)
# model.set_balance_point(cooling=55)
model.run('train')
model.run('test')
model.score()
model.run('normalize') #ToDo: where does x_norm come from?

model.run('predict', start='2017-11-01', end='2017-12-31')

model.Y_pred.plot()
model.y_pred.plot(grid=True)

pass