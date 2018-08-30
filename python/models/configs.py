import json
from sklearn.model_selection import train_test_split
import uuid
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import csv
from colorama import Fore, init, Back, Style
# Colorama settings
# init(autoreset=True)
from enum import Enum
from .data_generator import DataGenerator
from sklearn.preprocessing import Normalizer, StandardScaler, normalize

class Norm(Enum):
    NONE = 1
    STANDARD = 2
    NORM_SAMPLES = 3
    NORM_FEATURES = 4
    MINMAX = 5

class StructureTemplate(object):
    def get_dict(self):
        raise NotImplementedError('Method get_dict not implemented in class.')

class TrainConfig(StructureTemplate):
    def __init__(self, train_data_config={}, train_data_analysis={}, valid_data_analysis={},
                 # test_data_config={}, test_data_analysis={},
                 network_config={}, train_result={}, validation_result={}, test_configs=[]):
        self.id = str(uuid.uuid4())
        self.train_data_config = train_data_config
        self.network_config = network_config
        self.train_data_analysis = train_data_analysis
        self.valid_data_analysis = valid_data_analysis
        self.train_result = train_result
        self.validation_result = validation_result

    def dump_to_json(self, fname):
        # Preprocess data
        data = {self.id: {'data_config': self.train_data_config.__dict__,
                          'test_data_config': self.test_data_config.__dict__,
                          'network_config': self.network_config.__dict__,
                          'train_result': self.train_result.__dict__}}
        # Dump data
        if not os.path.isfile(fname) or os.stat(fname).st_size == 0:  # If file does not exist or is empty
            a = [data]
            with open(fname, 'w') as f:
                json.dump(a, f)
        else:
            with open(fname, 'r') as f:
                feeds = json.load(f)
            with open(fname, 'w') as f:
                feeds.append(data)
                json.dump(feeds, f)


    def dump_images(self, history):
        if not isinstance(history, dict):
            history = history.history

        # Check if validation values are present:
        if 1 in [1 for key, value in history.items() if 'val' in key.lower()]:
            val = True
        else:
            val = False

        # Only select keys that do not start with 'val'
        filtered_history = dict((key, value) for key, value in history.items() if not key.startswith('val'))

        for key in filtered_history.keys():
            pyplot.plot(history[key])
            if val == True:
                pyplot.plot(history['val_{}'.format(key)])

            pyplot.title(self.id)
            pyplot.ylabel(key)
            pyplot.xlabel('epoch')

            if val == True:
                pyplot.legend(['train', 'validation'], loc='upper right')
            else:
                pyplot.legend(['train'], loc='upper right')

            fname = os.path.join('output', 'images', '{}_{}.png'.format(self.id, key))
            pyplot.savefig(fname)
            pyplot.close()


    def get_dict(self):
        return self.__dict__


    def dump_to_csv(self, fname):
        csv_header = []

        big_dict = {}

        for key, val in self.get_dict().items():
            if key == 'id':
                csv_header.append('id')
                big_dict = {key: val}
            else:
                if key == 'valid_data_analysis':
                    new_dict = val.get_dict(valid=True)
                else:
                    if hasattr(val, 'get_dict'):
                        new_dict = val.get_dict()
                    else:
                        new_dict = {}

                big_dict = {**big_dict, **new_dict}

        file_exists = os.path.isfile(fname)
        with open(fname, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(big_dict.keys()), delimiter=';')
            if not file_exists:
                writer.writeheader()
            writer.writerow(big_dict)
            print(Fore.YELLOW)
            print(big_dict)
            print(Style.RESET_ALL)


class TestConfig(StructureTemplate):

    def __init__(self, model_id=None, test_data_config=None, test_data_analysis=None, test_result=None):
        self.id = str(uuid.uuid4())
        self.model_id = model_id
        self.test_data_config = test_data_config
        self.test_data_analysis = test_data_analysis
        self.test_result = test_result


    def get_dict(self):
        return self.__dict__


    def dump_to_csv(self, fname):
        csv_header = []

        big_dict = {}

        # TODO: Check types of elements to handle cases such as 'id' and 'model_id' (which have no get_dict() method) automatically
        for key, val in self.get_dict().items():
            if key == 'id':
                csv_header.append('id')
                big_dict = {key: val}
            elif key == 'model_id':
                csv_header.append('model_id')
                big_dict = {**big_dict, **{key: val}}
            else:
                if hasattr(val, 'get_dict'):
                    new_dict = val.get_dict()
                else:
                    new_dict = {}
                big_dict = {**big_dict, **new_dict}

        file_exists = os.path.isfile(fname)
        with open(fname, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(big_dict.keys()), delimiter=';')
            if not file_exists:
                writer.writeheader()
            writer.writerow(big_dict)
            print(Fore.YELLOW)
            print(big_dict)
            print(Style.RESET_ALL)


class DataConfig(StructureTemplate):

    def __init__(self, main_data=None, method_name=None, norm=Norm.NONE, atk_index=None, subset_size=None, n_atk_subsets=None,
                 range_atk=None, timestep=16, c=0.2, verbose=True, ratio=1, random=0, P=None, A=None, atk_function=None):
        self.data_generator = DataGenerator(main_data)
        self.method_name = method_name
        self.norm = norm
        self.atk_index = atk_index
        self.subset_size = subset_size
        self.n_atk_subsets = n_atk_subsets
        self.timestep = timestep
        self.c = c
        self.verbose = verbose
        self.range_atk = range_atk
        self.ratio = ratio
        self.random = random
        self.P = P
        self.A = A
        self.atk_function = atk_function


    def print(self):
        print("method_name   : {}".format(self.method_name))
        print("norm          : {}".format(self.norm))
        print("atk_index     : {}".format(self.atk_index))
        print("subset_size   : {}".format(self.subset_size))
        print("n_atk_subsets : {}".format(self.n_atk_subsets))
        print("timestep      : {}".format(self.timestep))
        print("c             : {}".format(self.c))
        print("verbose       : {}".format(self.verbose))
        print("range_atk     : {}".format(self.range_atk))


    def retrieve_data_set(self):
        rd = getattr(self.data_generator, self.method_name)(self)
        return rd


    def get_dict(self):
        exclude = ['data_generator']
        return {k:v for k, v in self.__dict__.items() if k not in exclude}


class KerasNetworkConfig(StructureTemplate):

    def __init__(self, num_input=None, timestep=None, num_hidden1=None, num_hidden2=None, num_output=None, batch_size=None,
                 epochs=None, dropout=None, early_stopping=False, csvlogger=False, tensorboard=False,
                 checkpoint=False):
        self.num_input = num_input
        self.timestep = timestep
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        self.num_output = num_output
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.early_stopping = early_stopping
        self.csvlogger = csvlogger
        self.tensorboard = tensorboard
        self.checkpoint = checkpoint


    def get_dict(self):
        return self.__dict__

class TfNetworkConfig(StructureTemplate):

    def __init__(self, num_input=None, timestep=None, num_hidden=None, num_output=None, batch_size=None,
                 epochs=None, orig_decay=None, max_lr_epoch=None, lr_given=None, dropout_given=None, tolerance=None,
                 display_step=None, hidden_layers=None, train_stop=None, val_loss_improv=0):
        self.num_input = num_input
        self.timestep = timestep
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.orig_decay = orig_decay
        self.max_lr_epoch = max_lr_epoch
        self.lr_given = lr_given
        self.dropout_given = dropout_given
        self.tolerance = tolerance
        self.display_step = display_step
        self.train_stop = train_stop
        self.val_loss_improv = val_loss_improv


    def get_dict(self):
        return self.__dict__


class Analyzer:
    def X_y_data(data):
        X = data['X']
        y = data['y']

        X_shape = X.shape
        y_shape = y.shape
        msmts_total = y.size
        msmts_atk = np.count_nonzero(y)
        msmts_non_atk = np.count_nonzero(y == 0)
        msmts_p_atk = msmts_atk / msmts_total
        rows_total = len(y)
        rows_atk = np.count_nonzero(np.count_nonzero(y, axis=1))
        rows_non_atk = np.count_nonzero(np.count_nonzero(y, axis=1) == 0)
        rows_p_atk = rows_atk / rows_total
        mean_atk_msmts_per_row = np.mean(np.count_nonzero(y, axis=1))

        return EvalValues(
            X_shape = X_shape,
            y_shape = y_shape,
            msmts_total = msmts_total,
            msmts_atk = msmts_atk,
            msmts_non_atk = msmts_non_atk,
            msmts_p_atk = msmts_p_atk,
            rows_total = rows_total,
            rows_atk = rows_atk,
            rows_non_atk = rows_non_atk,
            rows_p_atk = rows_p_atk,
            mean_atk_msmts_per_row=mean_atk_msmts_per_row
        )


class EvalValues(StructureTemplate):
    def __init__(self, X_shape=None, y_shape=None, msmts_total=None, msmts_atk=None, msmts_non_atk=None, msmts_p_atk=None, rows_total=None,
                 rows_atk=None, rows_non_atk=None, rows_p_atk=None, mean_atk_msmts_per_row=None):
        self.X_shape = X_shape
        self.y_shape = y_shape
        self.msmts_total = msmts_total
        self.msmts_atk = msmts_atk
        self.msmts_non_atk = msmts_non_atk
        self.msmts_p_atk = msmts_p_atk
        self.rows_total = rows_total
        self.rows_atk = rows_atk
        self.rows_non_atk = rows_non_atk
        self.rows_p_atk = rows_p_atk
        self.mean_atk_msmts_per_row = mean_atk_msmts_per_row


    def get_dict(self, valid=False):
        if valid == True:
            return {'V_{}'.format(k): v for k, v in self.__dict__.items()}
        else:
            return self.__dict__


class Result(StructureTemplate):
    def __init__(self, train_time=None, stopped_epoch=None, model_eval_values=None):
        self.train_time = train_time
        self.model_eval_values = model_eval_values
        self.stopped_epoch = stopped_epoch


    def get_dict(self):
        return {**{k:v for k, v in self.__dict__.items() if not k == 'model_eval_values'}, **self.model_eval_values}


class TestResult(StructureTemplate):
    def __init__(self, model_eval_values=None):
        self.model_eval_values = model_eval_values


    def get_dict(self):
        return {**{k: v for k, v in self.__dict__.items() if not k == 'model_eval_values'}, **self.model_eval_values}