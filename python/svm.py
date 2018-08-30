import pickle
from os.path import join
import numpy as np
import seaborn as sns; sns.set()
from models.configs import DataConfig, Analyzer, Norm
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
import time
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
import os
import csv
import math

with open(join('..', 'data', 'main_load_data.pkl'), 'rb') as f:
    power_grid_data = pickle.load(f)

Ps = [0.5]
subset_sizes = [28,29]
norms = [Norm.MINMAX]
atk_styles = [{'atk_function': 2, 'A': 0.1, 'c': 0}]

for atk_style in atk_styles:
    for norm in norms:
        for P in Ps:
            for subset_size in subset_sizes:
                n_atk_subsets = math.ceil(subset_size / 2)
                train_data_config = DataConfig(main_data=power_grid_data, method_name='create_fdi_X2_y1_se', norm=norm,
                                               subset_size=subset_size, n_atk_subsets=n_atk_subsets, P=P,
                                               timestep=16, c=atk_style['c'], random=0, ratio=0.7, atk_function=atk_style['atk_function'],
                                               A=atk_style['A'])


                rd = train_data_config.retrieve_data_set()

                Xtrain = rd['X_train']
                ytrain = rd['y_train']
                Xtest = rd['X_test']
                ytest = rd['y_test']

                n = 10000
                n_train = int(n * 0.7)
                n_test = int(n * 0.3)

                Xtrain = Xtrain[0:n_train]
                ytrain = ytrain[0:n_train]
                Xtest = Xtest[n:n + n_test]
                ytest = ytest[n:n + n_test]

                model = OneVsRestClassifier(SVC(kernel="rbf"))

                scores = ['f1']

                # parameters = {
                #     "estimator__C": [1,10,100,1000,10000,100000],
                #     "estimator__gamma": [0.01, 0.1, 1, 10],
                # }
                parameters = {
                    "estimator__C": [10000],
                    "estimator__gamma": [0.1],
                }

                # for score in scores:
                #     print("# Tuning hyper-parameters for %s" % score)
                #     print()
                #
                #     clf = GridSearchCV(model, parameters, cv=5,
                #                        scoring='%s_macro' % score)
                #     clf.fit(Xtrain, ytrain)
                #
                #     print("Best parameters set found on development set:")
                #     print()
                #     print(clf.best_params_)
                #     print()
                #     print("Grid scores on development set:")
                #     print()
                #     means = clf.cv_results_['mean_test_score']
                #     stds = clf.cv_results_['std_test_score']
                #     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                #         print("%0.3f (+/-%0.03f) for %r"
                #               % (mean, std * 2, params))
                #     print()
                #
                #     print("Detailed classification report:")
                #     print()
                #     print("The model is trained on the full development set.")
                #     print("The scores are computed on the full evaluation set.")
                #     print()
                #     y_true, y_pred = ytest, clf.predict(Xtest)
                #     print(classification_report(y_true, y_pred))
                #     print()

                grid = GridSearchCV(model, parameters)

                start = time.time()
                grid.fit(Xtrain, ytrain)

                print(grid.best_params_)

                model = grid.best_estimator_
                yfit = model.predict(Xtest)

                print(time.time() - start)
                print(classification_report(ytest, yfit))

                result = {}
                result['train_time'] = time.time() - start
                result['f1'] = f1_score(ytest, yfit, average='weighted')
                result['aoc_mac'] = roc_auc_score(ytest, yfit)
                result['aoc_mic'] = roc_auc_score(ytest, yfit, average='micro')
                result['n_train'] = len(Xtrain)
                result['n_test'] = len(Xtest)
                result['C'] = grid.best_params_['estimator__C']
                result['gamma'] = grid.best_params_['estimator__gamma']
                result['P'] = P
                result['subset_size'] = subset_size
                result['n_atk_subsets'] = n_atk_subsets
                result['c'] = atk_style['c']
                result['A'] = atk_style['A']
                result['tp'] = np.count_nonzero(yfit * ytest)
                result['tn'] = np.count_nonzero((yfit - 1) * (ytest - 1))
                result['fp'] = np.count_nonzero(yfit * (ytest - 1))
                result['fn'] = np.count_nonzero((yfit - 1) * ytest)
                result['norm'] = norm
                result['atk_function'] = atk_style['atk_function']

                analyzed = Analyzer.X_y_data({'X': Xtest, 'y': ytest}).get_dict()

                print(result)

                csv_header = []
                fname = join('output', 'svm_libido.csv')

                result = {**result, **analyzed}

                file_exists = os.path.isfile(fname)
                with open(fname, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=list(result.keys()), delimiter=';')
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(result)
