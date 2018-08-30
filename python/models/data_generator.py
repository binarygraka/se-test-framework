import numpy as np
from random import randint
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler, Normalizer
import models.configs
import random
from sklearn.model_selection import train_test_split

class DataGenerator:

    def __init__(self, data):
        # os.path.join('..', 'data', 'main_load_data.pkl')
        if not data:
            raise ValueError("Invalid argument.")
        elif not isinstance(data, dict):
            raise ValueError("Data is not of type dictionary.")
        self._data = data

    # @property
    # def data(self):
    #     print("Getting value")
    #     return self._data
    #
    # @data.setter
    # def data(self, value):
    #     if not value:
    #         raise ValueError("No data has been specified.")
    #     self._data = value


    def create_fdi_X3_y1(self, config, data=""):
        """
        Generates FDI data.
        Each element in the config.atk_index gets attacked by a probability of 0.5.
        If at least one element in each config.timestep got attacked, the label is 1. Otherwise the label is 0.

        @type z: ndarray(num_data, num_input)
        @param z: Measurements
        @type H: ndarray(num_input, _, num_data)
        @param H: Measurement Jacobian Matrix
        @type config.atk_index: array
        @param config.atk_index: Indices of config.timestep that get attacked by a chance of 50%
        @type config.norm: boolean
        @param config.norm: Normalize using sklearn.normalize, which sets every vector of the sample to unit config.norm
        @type config.timestep: int
        @param config.timestep: Size in which z gets grouped (default: 16)
        @type config.c: float
        @param config.c: Error that gets introduced to the state estimation variables (default: 0.2)

        @rtype X: ndarray(num_data, config.timestep, num_input)
        @return X: Data
        @rtype y: ndarray(num_data, 1)
        @return y: Label
        """

        if not config and not isinstance(config, models.config.DataConfig):
            raise ValueError("models.config.DataConfig is invalid.")

        if not data and not isinstance(data, dict):
            data = self._data
        z = data['z']
        H = data['H']

        usable = int(len(z) / config.timestep)
        z_len = len(z[1])

        y = np.zeros((usable, 1))
        X = np.zeros((usable, config.timestep, z_len))

        for t in iter(range(usable)):
            temp = np.zeros((config.timestep, z_len))

            for step in iter(range(config.timestep)):
                idx = (t * config.timestep) + step
                temp[step, :] = z[idx, :]

                if step in config.atk_index:
                    if np.random.randint(2):

                        n_se = H[:, :, idx].shape[1]

                        xx = np.zeros((n_se, 1))
                        xx[10:11] = config.c

                        z_a = z[idx, :] + np.transpose(np.dot(H[:, :, idx], xx))
                        temp[step, :] = z_a
                        y[t] = 1
                    else:
                        temp[step, :] = z[idx, :]
                        y[t] = 0
            if config.norm is True:
                X[t, :, :] = normalize(temp)
            else:
                X[t, :, :] = temp

        if config.verbose:
            print("Max: {}, min: {}".format(np.amax(X), np.amin(X)))
            print("X shape: {}".format(X.shape))
            print("y shape: {}".format(y.shape))

        return X, y


    def create_fdi_X3_y1_se(self, config, data=""):
        """
        Generates FDI data.
        Each element in the config.atk_index gets attacked by a probability of 0.5.
        If at least one element in each config.timestep got attacked, the label is 1. Otherwise the label is 0.

        @type z: ndarray(num_data, num_input)
        @param z: Measurements
        @type H: ndarray(num_input, _, num_data)
        @param H: Measurement Jacobian Matrix
        @type config.atk_index: array
        @param config.atk_index: Indices of config.timestep that get attacked by a chance of 50%
        @type config.norm: boolean
        @param config.norm: Normalize using sklearn.normalize, which sets every vector of the sample to unit config.norm
        @type config.subset_size: int
        @param config.subset_size: Defines in how many parts the state estimation variables should be grouped
        @type config.n_atk_subsets: int
        @param config.n_atk_subsets: Defines how many of the subset parts should be attacked. Cannot be higher than the number of subsets.
        @type config.timestep: int
        @param config.timestep: Size in which z gets grouped (default: 16)
        @type config.c: float
        @param config.c: Error that gets introduced to the state estimation variables (default: 0.2)

        @rtype X: ndarray(num_data, config.timestep, num_input)
        @return X: Data
        @rtype y: ndarray(num_data, 1)
        @return y: Label
        """

        if not config and not isinstance(config, models.config.DataConfig):
            raise ValueError("models.config.DataConfig is invalid.")

        if not data and not isinstance(data, dict):
            data = self._data
        z = data['z']
        H = data['H']

        usable = int(len(z) / config.timestep)
        z_len = len(z[1])

        y = np.zeros((usable, config.subset_size))
        X = np.zeros((usable, config.timestep, z_len))
        S = np.zeros((usable, H[:, :, 1].shape[1]))

        for t in iter(range(usable)):
            temp = np.zeros((config.timestep, z_len))

            for step in iter(range(config.timestep)):
                idx = (t * config.timestep) + step
                temp[step, :] = z[idx, :]

                if step in config.atk_index:
                    if np.random.randint(2):

                        n_se = H[:, :, idx].shape[1]

                        # Attack only subset of state variables
                        se_w = int(H[:, :, idx].shape[1] / config.subset_size)  # Integer sets
                        se_remainder = H[:, :, idx].shape[1] % config.subset_size  # Remainder
                        # se_attacked = [randint(0, config.subset_size - 1), randint(0, config.subset_size - 1)]  # Attacked subsets
                        se_attacked = [randint(0, config.subset_size - 1) for _ in range(config.n_atk_subsets)]

                        se_atk_index = np.zeros((n_se, 1))
                        for x in range(config.subset_size):
                            if x in se_attacked:  # Attacked set
                                if x == config.subset_size - 1:  # Last element
                                    se_atk_index[x * se_w:(x * se_w) + se_w + se_remainder] = config.c
                                else:
                                    se_atk_index[x * se_w:(x * se_w) + se_w] = config.c

                        z_a = z[idx, :] + np.transpose(np.dot(H[:, :, idx], se_atk_index))
                        temp[step, :] = z_a
                        y[t, se_attacked] = 1
                        S[t, :] = np.transpose(se_atk_index)
                    else:
                        temp[step, :] = z[idx, :]
                        y[t] = 0
            if config.norm is True:
                X[t, :, :] = normalize(temp)
            else:
                X[t, :, :] = temp

        if config.verbose:
            print("Max: {}, min: {}, after".format(np.amax(X), np.amin(X)))
            print("X shape: {}".format(X.shape))
            # print("S shape: {}".format(S.shape))
            print("y shape: {}".format(y.shape))

        return X, y#, S


    def create_fdi_X3_y1_se_window(self, config, data=""):
        rd = self.create_fdi_X2_y1_se(config)

        if config.ratio != 1:
            X_train, y_train = self.create_fdi_window_from_fdi_X2_y1_se(rd['X_train'], rd['y_train'], config)
            X_test, y_test = self.create_fdi_window_from_fdi_X2_y1_se(rd['X_test'], rd['y_test'], config)
        else:
            X_train, y_train = self.create_fdi_window_from_fdi_X2_y1_se(rd['X_train'], rd['y_train'], config)
            X_test = None
            y_test = None

        return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'c': rd['c'], 'a': rd['a']}


    def create_fdi_X3_yn(self, config, data=""):
        """
        Generates FDI data.
        At every set of config.timestep, a random number that is part of config.range_atk gets chosen which determines, how many measurements will get attacked.
        E.g. config.range_atk=[3,6] --> This means that for every config.timestep between 3 and 6 measurements get attacked.
        The label is 1 for every element that has been attacked and 0 otherwise.

        @type z: ndarray(num_data, num_input)
        @param z: Measurements
        @type H: ndarray(num_input, _, num_data)
        @param H: Measurement Jacobian Matrix
        @type config.range_atk: array
        @param config.range_atk: Range for number of measurements that get attacked in every set of config.timestep
        @type config.norm: boolean
        @param config.norm: Normalize using sklearn.normalize, which sets every vector of the sample to unit config.norm
        @type config.timestep: int
        @param config.timestep: Size in which z gets grouped (default: 16)
        @type config.c: float
        @param config.c: Error that gets introduced to the state estimation variables (default: 0.2)

        @rtype X: ndarray(num_data, config.timestep, num_input)
        @return X: Data
        @rtype y: ndarray(num_data, config.timestep)
        @return y: Label
        """

        if not config and not isinstance(config, models.config.DataConfig):
            raise ValueError("models.config.DataConfig is invalid.")

        if not data and not isinstance(data, dict):
            data = self._data
        z = data['z']
        H = data['H']

        usable = int(len(z) / config.timestep)
        z_len = len(z[1])

        y = np.zeros((usable, config.timestep))
        X = np.zeros((usable, config.timestep, z_len))

        for t in iter(range(usable)):
            temp = np.zeros((config.timestep, z_len))
            n_atk_msmts = randint(config.range_atk[0], config.range_atk[1])
            config.atk_index = [randint(0, config.timestep - 1) for _ in range(n_atk_msmts)]

            for step in iter(range(config.timestep)):
                idx = (t * config.timestep) + step
                temp[step, :] = z[idx, :]

                if step in config.atk_index:
                    n_se = H[:, :, idx].shape[1]

                    z_a = z[idx, :] + np.transpose(np.dot(H[:, :, idx], (config.c * np.ones((n_se, 1)))))
                    temp[step, :] = z_a
                    y[t, step] = 1
                else:
                    temp[step, :] = z[idx, :]

            if config.norm is True:
                X[t, :, :] = normalize(temp)
            else:
                X[t, :, :] = temp

        if config.verbose:
            print("Max: {}, min: {}, after".format(np.amax(X), np.amin(X)))
            print("X shape: {}".format(X.shape))
            print("y shape: {}".format(y.shape))

        return X, y


    def create_fdi_X2_y1(self, config, data=""):
        """
          Generates FDI data.
          Each element in the config.atk_index gets attacked by a probability of 0.5.
          If at least one element in each config.timestep got attacked, the label is 1. Otherwise the label is 0.

          Different to create_fdi_X3_y1, X and y are not grouped into arrays of size config.timestep, but are continuous.

          @type z: ndarray(num_data, num_input)
          @param z: Measurements
          @type H: ndarray(num_input, _, num_data)
          @param H: Measurement Jacobian Matrix
          @type config.atk_index: array
          @param config.atk_index: Indices of config.timestep that get attacked by a chance of 50%
          @type config.norm: boolean
          @param config.norm: Normalize using sklearn.normalize, which sets every vector of the sample to unit config.norm
          @type config.timestep: int
          @param config.timestep: Size in which z gets grouped (default: 16)
          @type config.c: float
          @param config.c: Error that gets introduced to the state estimation variables (default: 0.2)

          @rtype X: ndarray(num_data*config.timestep, num_input)
          @return X: Data
          @rtype y: ndarray(num_data*config.timestep, 1)
          @return y: Label
          """

        if not config and not isinstance(config, models.config.DataConfig):
            raise ValueError("models.config.DataConfig is invalid.")

        if not data and not isinstance(data, dict):
            data = self._data
        z = data['z']
        H = data['H']

        usable = int(len(z) / config.timestep)
        z_len = len(z[1])

        y = np.zeros((usable * config.timestep, 1))
        X = np.zeros((usable * config.timestep, z_len))

        for t in iter(range(usable)):
            temp = np.zeros((config.timestep, z_len))

            for step in iter(range(config.timestep)):
                idx = (t * config.timestep) + step
                X[idx, :] = z[idx, :]

                if step in config.atk_index:
                    if np.random.randint(2):

                        n_se = H[:, :, idx].shape[1]

                        z_a = z[idx, :] + np.transpose(np.dot(H[:, :, idx], (config.c * np.ones((n_se, 1)))))
                        X[idx, :] = z_a
                        y[idx] = 1
                    else:
                        X[idx, :] = z[idx, :]
                        y[idx] = 0
                else:
                    y[idx] = 0
        if config.norm is True: X = normalize(X)

        if config.verbose:
            print("Max: {}, min: {}, after".format(np.amax(X), np.amin(X)))
            print("X shape: {}".format(X.shape))
            print("y shape: {}".format(y.shape))

        return X, y


    def create_fdi_X2_y1_se(self, config, data=""):
        """
        Generates FDI data.
        Each element in the config.atk_index gets attacked by a probability of 0.5.
        If at least one attack in each timespan of size timestep (e.g. span of 16) is present, the label is 1. Otherwise the label is 0.

        Different to create_fdi_X3_y1, X and y are not grouped into arrays of size config.timestep, but are continuous.

        @type z: ndarray(num_data, num_input)
        @param z: Measurements
        @type H: ndarray(num_input, _, num_data)
        @param H: Measurement Jacobian Matrix
        @type config.atk_index: array
        @param config.atk_index: Indices of config.timestep that get attacked by a chance of 50%
        @type config.norm: boolean
        @param config.norm: Normalize using sklearn.normalize, which sets every vector of the sample to unit config.norm
        @type config.timestep: int
        @param config.timestep: Size in which z gets grouped (default: 16)
        @type config.c: float
        @param config.c: Error that gets introduced to the state estimation variables (default: 0.2)
        @type config.P: int (0 <= number <= 1)
        @param config.P: Amount of timesteps that get attacked. E.g., 0.1 means that 10% of all timesteps get attacked. atk_index is useless if P is specified
        @type config.atk_function: int
        @param config.atk_function: 0: generate_atk_subset_msmt, 1: generate_prob_distrib_msmt_all, 2: generate_prob_distrib_msmt

        @rtype X: ndarray(num_data*config.timestep, num_input)
        @return X: Data
        @rtype y: ndarray(num_data*config.timestep, 1)
        @return y: Label
        """

        if not config and not isinstance(config, models.config.DataConfig):
            raise ValueError("models.config.DataConfig is invalid.")

        if not data and not isinstance(data, dict):
            data = self._data
        z = data['z']
        H = data['H']

        usable = int(len(z) / config.timestep)
        z_len = len(z[1])

        y = np.zeros((usable * config.timestep, config.subset_size))
        X = np.zeros((usable * config.timestep, z_len))
        Z = np.zeros((usable * config.timestep, z_len))
        S = np.zeros((usable, H[:, :, 1].shape[1]))
        a = np.zeros((usable * config.timestep, z_len))
        c = np.zeros((usable * config.timestep, H[:, :, 1].shape[1]))

        for t in iter(range(usable)):

            for step in iter(range(config.timestep)):
                idx = (t * config.timestep) + step
                X[idx, :] = z[idx, :]

                if config.P:
                    if random.random() >= (1 - config.P):
                        if config.atk_function == 0:
                            z_a, se_attacked, c_t, a_t = self.generate_atk_subset_msmt(H[:, :, idx], z[idx, :], config.subset_size,
                                                                             config.n_atk_subsets, config.c)
                        elif config.atk_function == 1:
                            z_a, se_attacked, c_t, a_t = self.generate_prob_distrib_msmt_all(H[:, :, idx], z[idx, :], config.subset_size,
                                                                                   config.A)
                        elif config.atk_function == 2:
                            z_a, se_attacked, c_t, a_t = self.generate_prob_distrib_msmt(H[:, :, idx], z[idx, :], config.subset_size,
                                                                                  config.n_atk_subsets, config.A)
                        elif config.atk_function == 3:
                            z_a, se_attacked, c_t, a_t = self.generate_prob_distrib_msmt(H[:, :, idx], z[idx, :], config.subset_size,
                                                                                    config.n_atk_subsets, config.A, mean_zero=True)
                        c[idx, :] = c_t.ravel()
                        a[idx, :] = a_t

                        X[idx, :] = z_a
                        Z[idx, :] = z[idx, :]
                        y[idx, se_attacked] = 1
                else:
                    if step in config.atk_index and np.random.randint(2):
                        if config.atk_function == 0:
                            z_a, se_attacked, c_t, a_t = self.generate_atk_subset_msmt(H[:, :, idx], z[idx, :], config.subset_size, config.n_atk_subsets,
                                                                             config.c)
                        elif config.atk_function == 1:
                            z_a, se_attacked, c_t, a_t = self.generate_prob_distrib_msmt_all(H[:, :, idx], z[idx, :], config.subset_size,
                                                                                   config.n_atk_subbsets, config.A)
                        elif config.atk_function == 2:
                            z_a, se_attacked, c_t, a_t = self.generate_prob_distrib_msmt(H[:, :, idx], z[idx, :], config.subset_size,
                                                                                  config.n_atk_subsets, config.A)
                        elif config.atk_function == 3:
                            z_a, se_attacked, c_t, a_t = self.generate_prob_distrib_msmt(H[:, :, idx], z[idx, :], config.subset_size,
                                                                                    config.n_atk_subsets, config.A, mean_zero=True)
                        c[idx, :] = c_t.ravel()
                        a[idx, :] = a_t

                        X[idx, :] = z_a
                        Z[idx, :] = z[idx, :]
                        y[idx, se_attacked] = 1

        if config.verbose:
            print("Before. Max: {}, min: {}".format(np.amax(X), np.amin(X)))
            print("X shape: {}".format(X.shape))
            print("y shape: {}".format(y.shape))

        if config.ratio != 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=config.ratio, random_state=config.random)

            if config.norm != models.configs.Norm.NONE:
                X_train, X_test = self.normalize_data(config.norm, X_train, X_test)
        else:
            X_train = X
            y_train = y
            X_test = None
            y_test = None
            if config.norm != models.configs.Norm.NONE:
                X_train, X_test = self.normalize_data(config.norm, X_train, X_test)

        if config.verbose:
            print("After. Max: {}, min: {}".format(np.amax(X_train), np.amin(X_train)))

        return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'c': c, 'a': a, 'Z': Z}


    def normalize_data(self, norm, X_train, X_test):
        if norm is models.configs.Norm.STANDARD:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            if X_test is not None: X_test = scaler.transform(X_test)
        elif norm is models.configs.Norm.NORM_SAMPLES:  # Samples are scaled individually, therefore norms do not have to be saved
            scaler = Normalizer()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_train = normalize(X_train, axis=1)
            if X_test is not None: X_test = scaler.transform(X_test)
            # if X_test is not None: X_test = normalize(X_test, axis=1)
        elif norm is models.configs.Norm.NORM_FEATURES:
            X_train, norms = normalize(X_train, return_norm=True, axis=0)
            if X_test is not None: X_test = X_test / norms
        elif norm is models.configs.Norm.MINMAX:
            scaler = MinMaxScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            if X_test is not None: X_test = scaler.transform(X_test)

        return X_train, X_test


    def generate_prob_distrib_msmt_all(self, H, z, subset_size, A):
        '''
        Attack all subsets, such that c forms a normal distribution with mean and std from the measurement (Yan16a, Ozay16)
        Scaled by A

        :param H:
        :param z:
        :param A:
        :return:
        '''
        n_se = H.shape[1]
        se_attacked = range(subset_size)

        c = np.random.normal(np.mean(z), np.std(z), n_se) * A

        a = np.transpose(np.dot(H, c))
        z_a = z + a

        return z_a, se_attacked, c, a


    def generate_atk_subset_msmt(self, H, z, subset_size, n_atk_subsets, c):
        n_se = H.shape[1]

        # Attack only subset of state variables
        se_w = int(H.shape[1] / subset_size)  # Integer sets
        se_remainder = H.shape[1] % subset_size  # Remainder
        # se_attacked = [randint(0, subset_size - 1) for _ in range(n_atk_subsets)]
        se_attacked = np.random.choice(subset_size, n_atk_subsets, replace=False)

        se_atk_index = np.zeros((n_se, 1))
        for x in range(subset_size):
            if x in se_attacked:  # Attacked set
                if x == subset_size - 1:  # Last element
                    se_atk_index[x * se_w:(x * se_w) + se_w + se_remainder] = c
                else:
                    se_atk_index[x * se_w:(x * se_w) + se_w] = c

        a = np.transpose(np.dot(H, se_atk_index))
        z_a = z + a
        return z_a, se_attacked, se_atk_index, a


    def generate_prob_distrib_msmt(self, H, z, subset_size, n_atk_subsets, A, mean_zero=False):
        n_se = H.shape[1]

        # Attack only subset of state variables
        se_w = int(H.shape[1] / subset_size)  # Integer sets
        se_remainder = H.shape[1] % subset_size  # Remainder
        # se_attacked = [randint(0, subset_size - 1) for _ in range(n_atk_subsets)]
        se_attacked = np.random.choice(subset_size, n_atk_subsets, replace=False)

        c = np.zeros((n_se, 1))
        for x in range(subset_size):
            if x in se_attacked:  # Attacked set
                if mean_zero:
                    z_mean = 0
                else:
                    z_mean = np.mean(z)
                if x == subset_size - 1:  # Last element
                    c[x * se_w:(x * se_w) + se_w + se_remainder] = np.multiply(np.random.normal(z_mean, np.std(z), (se_w + se_remainder, 1)), A)
                else:
                    c[x * se_w:(x * se_w) + se_w] = np.multiply(np.random.normal(z_mean, np.std(z), (se_w, 1)), A)

        a = np.transpose(np.dot(H, c))
        z_a = z + a
        return z_a, se_attacked, c, a



    def create_fdi_X2_yn(self, config, data=""):
        """
        Generates FDI data.
        At every set of config.timestep, a random number that is part of config.range_atk gets chosen which determines, how many measurements will get attacked.
        E.g. config.range_atk=[3,6] --> This means that for every config.timestep between 3 and 6 measurements get attacked.
        The label is 1 for every element that has been attacked and 0 otherwise.

        Different to create_fdi_X3_yn, X and y are not grouped into arrays of size config.timestep, but are continuous.

        @type z: ndarray(num_data, num_input)
        @param z: Measurements
        @type H: ndarray(num_input, _, num_data)
        @param H: Measurement Jacobian Matrix
        @type config.range_atk: array
        @param config.range_atk: Range for number of measurements that get attacked in every set of config.timestep
        @type config.norm: boolean
        @param config.norm: Normalize using sklearn.normalize, which sets every vector of the sample to unit config.norm
        @type config.timestep: int
        @param config.timestep: Size in which z gets grouped (default: 16)
        @type config.c: float
        @param config.c: Error that gets introduced to the state estimation variables (default: 0.2)

        @rtype X: ndarray(num_data*config.timestep,, num_input)
        @return X: Data
        @rtype y: ndarray(num_data*config.timestep, config.timestep)
        @return y: Label
        """

        if not config and not isinstance(config, models.config.DataConfig):
            raise ValueError("models.config.DataConfig is invalid.")

        if not data and not isinstance(data, dict):
            data = self._data
        z = data['z']
        H = data['H']

        usable = int(len(z) / config.timestep)
        z_len = len(z[1])

        y = np.zeros((usable * config.timestep, 1))
        X = np.zeros((usable * config.timestep, z_len))

        for t in iter(range(usable)):

            n_atk_msmts = randint(config.range_atk[0], config.range_atk[1])
            config.atk_index = [randint(0, config.timestep - 1) for _ in range(n_atk_msmts)]

            for step in iter(range(config.timestep)):
                idx = (t * config.timestep) + step
                X[idx, :] = z[idx, :]

                if step in config.atk_index:
                    n_se = H[:, :, idx].shape[1]

                    z_a = z[idx, :] + np.transpose(np.dot(H[:, :, idx], (config.c * np.ones((n_se, 1)))))
                    X[idx, :] = z_a
                    y[idx] = 1
                else:
                    X[idx, :] = z[idx, :]
        if config.norm is True: X = normalize(X)

        if config.verbose:
            print("Max: {}, min: {}, after".format(np.amax(X), np.amin(X)))
            print("X shape: {}".format(X.shape))
            print("y shape: {}".format(y.shape))

        return X, y


    def create_fdi_window_from_fdi_X2_y1(self, X, y, config):
        """
        Takes an array of size (num_data*config.timestep, num_input).
        Then it rolls a sliding window of size config.timestep over the array and extracts all batches.
        The total number of extracted batches will be (num_data * config.timestep) - (config.timestep - 1).

        @type X: ndarray(num_data*config.timestep, num_input)
        @param X: Measurements
        @type y: ndarray(num_input*config.timestep, 1)
        @param y: Measurement Jacobian Matrix
        @type config.timestep: int
        @param config.timestep: Size in which z gets grouped (default: 16)

        @rtype X_new: ndarray((num_data * config.timestep) - (config.timestep - 1), num_input)
        @return X_new: Data
        @rtype y_new: ndarray((num_data * config.timestep) - (config.timestep - 1), 1)
        @return y_new: Label
        """

        if not config and not isinstance(config, models.config.DataConfig):
            raise ValueError("models.config.DataConfig is invalid.")

        X_re = X

        z_len = X.shape[1]

        X_re_len = X_re.shape[0]
        X_new = np.zeros((X_re_len - (config.timestep - 1), config.timestep, z_len))
        y_new = np.zeros((X_re_len - (config.timestep - 1), 1))

        for t in iter(range(X_re_len - (config.timestep - 1))):
            for i in iter(range(config.timestep)):
                X_new[t, i, :] = X_re[t + i, :]
                if i == (config.timestep - 1):
                    # Last step, check label
                    y_new[t] = y[t + i]

        if config.verbose:
            print("X shape: {}".format(X_new.shape))
            print("y shape: {}".format(y_new.shape))

        return X_new, y_new


    def create_fdi_window_from_fdi_X2_y1_se(self, X, y, config):
        """
        Takes an array of size (num_data*config.timestep, num_input).
        Then it rolls a sliding window of size config.timestep over the array and extracts all batches.
        The total number of extracted batches will be (num_data * config.timestep) - (config.timestep - 1).

        @type X: ndarray(num_data*config.timestep, num_input)
        @param X: Measurements
        @type y: ndarray(num_input*config.timestep, 1)
        @param y: Measurement Jacobian Matrix
        @type config.timestep: int
        @param config.timestep: Size in which z gets grouped (default: 16)

        @rtype X_new: ndarray((num_data * config.timestep) - (config.timestep - 1), num_input)
        @return X_new: Data
        @rtype y_new: ndarray((num_data * config.timestep) - (config.timestep - 1), 1)
        @return y_new: Label
        """

        if not config and not isinstance(config, models.config.DataConfig):
            raise ValueError("models.config.DataConfig is invalid.")

        X_re = X

        z_len = X.shape[1]

        X_re_len = X_re.shape[0]
        X_new = np.zeros((X_re_len - (config.timestep - 1), config.timestep, z_len))
        y_new = np.zeros((X_re_len - (config.timestep - 1), y.shape[1]))

        for t in iter(range(X_re_len - (config.timestep - 1))):
            for i in iter(range(config.timestep)):
                X_new[t, i, :] = X_re[t + i, :]
                if i == (config.timestep - 1):
                    # Last step, check label
                    y_new[t] = y[t + i]

        if config.verbose:
            print("X shape: {}".format(X_new.shape))
            print("y shape: {}".format(y_new.shape))

        return X_new, y_new


    def create_fdi_window_from_fdi_X2_yn(self, X, y, config):
        """
        Takes an array of size (num_data*config.timestep, num_input).
        Then it rolls a sliding window of size config.timestep over the array and extracts all batches.
        The total number of extracted batches will be (num_data * config.timestep) - (config.timestep - 1).

        @type X: ndarray(num_data*config.timestep, num_input)
        @param X: Measurements
        @type y: ndarray(num_input*config.timestep, 1)
        @param y: Measurement Jacobian Matrix
        @type config.timestep: int
        @param config.timestep: Size in which z gets grouped (default: 16)

        @rtype X_new: ndarray((num_data * config.timestep) - (config.timestep - 1), num_input)
        @return X_new: Data
        @rtype y_new: ndarray((num_data * config.timestep) - (config.timestep - 1), config.timestep)
        @return y_new: Label
        """

        if not config and not isinstance(config, models.config.DataConfig):
            raise ValueError("models.config.DataConfig is invalid.")

        X_re = X

        z_len = X.shape[1]

        X_re_len = X_re.shape[0]
        X_new = np.zeros((X_re_len - (config.timestep - 1), config.timestep, z_len))
        y_new = np.zeros((X_re_len - (config.timestep - 1), config.timestep))

        for t in iter(range(X_re_len - (config.timestep - 1))):
            for i in iter(range(config.timestep)):
                X_new[t, i, :] = X_re[t + i, :]
                # Last step, check label
                y_new[t, i] = y[t + i]

        if config.verbose:
            print("X shape: {}".format(X_new.shape))
            print("y shape: {}".format(y_new.shape))

        return X_new, y_new