from __future__ import print_function

import tensorflow as tf
from models.configs import DataConfig, TfNetworkConfig, TrainConfig, Analyzer, TestConfig, Result, Norm
import os
import pickle
import numpy as np
from os.path import join
import time
from colorama import Fore, init, Back, Style
from sklearn.metrics import roc_auc_score
import sys
import math
from sklearn.preprocessing import StandardScaler, Normalizer


class Model(object):
    def __init__(self, id=None, network_conf=None, is_training=None, train_data=None, validation_data=None,
                 test_data=None):
        self.id = id
        self.is_training = is_training

        if self.is_training:
            self.train_data = train_data
            self.validation_data = validation_data
            self.num_input = network_conf.num_input
            self.num_hidden = network_conf.num_hidden
            self.num_output = network_conf.num_output
            self.hidden_layers = network_conf.hidden_layers
            self.epochs = network_conf.epochs
            self.timestep = network_conf.timestep
            self.batch_size_given = network_conf.batch_size
            self.display_step = network_conf.display_step
            self.orig_decay = network_conf.orig_decay
            self.max_lr_epoch = network_conf.max_lr_epoch
            self.lr_given = network_conf.lr_given
            self.dropout_given = network_conf.dropout_given
            self.tolerance = network_conf.tolerance
            self.train_stop = network_conf.train_stop
            self.val_loss_improv = network_conf.val_loss_improv

            tf.add_to_collection('num_input', self.num_input)
            tf.add_to_collection('num_hidden', self.num_hidden)
            tf.add_to_collection('num_output', self.num_output)
            tf.add_to_collection('hidden_layers', self.hidden_layers)

            # Graph input
            self.x = tf.placeholder("float", [None, self.timestep, self.num_input], name='x')
            self.y = tf.placeholder("float", [None, self.num_output], name='y')
            self.batch_size = tf.placeholder(tf.int64, name='batch_size')
            self.dropout = tf.placeholder_with_default(1.0, shape=(), name='dropout')

            # Input pipeline
            train_dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y)).apply(
                tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat(self.epochs)
            validate_dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y)).apply(
                tf.contrib.data.batch_and_drop_remainder(self.batch_size))
            iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
            features, labels = iter.get_next()
            self.train_init_op = iter.make_initializer(train_dataset, name='train_init_op')
            self.validation_init_op = iter.make_initializer(validate_dataset)

            self.train_data = (train_data['X'], train_data['y'])
            self.validation_data = (validation_data['X'], validation_data['y'])

            # Define weights
            weights = {
                'out': tf.Variable(tf.random_normal([self.num_hidden, self.num_output]))
            }
            biases = {
                'out': tf.Variable(tf.random_normal([self.num_output]))
            }

            # Set up model
            inputs = tf.nn.dropout(features, self.dropout, name='inputs')

            self.init_state = tf.placeholder(tf.float32, [self.hidden_layers, 2, None, self.num_hidden], name='init_state')

            state_per_layer_list = tf.unstack(self.init_state, axis=0)
            rnn_tuple_state = tuple(
                [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
                 for idx in range(self.hidden_layers)]
            )

            def lstm_cell(n_hidden, dropout):
                cell = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
                return cell

            if self.hidden_layers > 0:
                cell = tf.contrib.rnn.MultiRNNCell(
                    [lstm_cell(self.num_hidden, self.dropout) for _ in range(self.hidden_layers)],
                    state_is_tuple=True)

            output, self.state = tf.nn.dynamic_rnn(cell, features, dtype=tf.float32, initial_state=rnn_tuple_state)

            # Extract last timestep of output
            output = tf.transpose(output, [1, 0, 2])
            output = tf.gather(output, int(output.get_shape()[0]) - 1)

            with tf.name_scope('Model'):
                # Make prediction
                self.logits = tf.matmul(output, weights['out']) + biases['out']

            with tf.name_scope('Loss'):
                # Calculate cost/loss
                self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=labels), name='cost')

            self.learning_rate = tf.Variable(0.0, trainable=False, name='learning_rate')

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
            with tf.name_scope('Optimizer'):
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.train.get_or_create_global_step(), name='train_op')

            self.new_lr = tf.placeholder(tf.float32, shape=[], name='new_lr')
            self.lr_update = tf.assign(self.learning_rate, self.new_lr, name='lr_update')

            # Predict
            self.prediction = tf.nn.sigmoid(self.logits, name='prediction')

            # Metrics for multilabel binary classification
            self.predicted_labels = tf.cast(tf.less(tf.constant(0.5), self.prediction), tf.float32, name='predicted_labels')
            self.correct_pred = tf.cast(tf.equal(self.predicted_labels, labels), tf.int32, name='correct_pred')
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name='accuracy')

            self.tp = tf.add(tf.count_nonzero(self.predicted_labels * labels), tf.constant(0, dtype=tf.int64), name='tp')  # Add 0, because otherwise this tensor is not available at model restore, why o'ever
            self.tn = tf.add(tf.count_nonzero((self.predicted_labels - 1) * (labels - 1)), tf.constant(0, dtype=tf.int64), name='tn')
            self.fp = tf.add(tf.count_nonzero((self.predicted_labels * (labels - 1))), tf.constant(0, dtype=tf.int64), name='fp')
            self.fn = tf.add(tf.count_nonzero(((self.predicted_labels - 1) * labels)), tf.constant(0, dtype=tf.int64), name='fn')
            self.precision = tf.divide(self.tp, tf.add(self.tp, self.fp), name='precision')
            self.labels = labels

            self.recall = tf.divide(self.tp, tf.add(self.tp, self.fn), name='recall')

            self.f1 = tf.multiply(tf.constant(2, dtype=tf.float64),
                                  tf.divide(tf.multiply(self.precision, self.recall), tf.add(self.precision, self.recall)), name='f1')

            # Tensorboard metrics
            tf.summary.scalar("precision", self.precision)
            tf.summary.scalar("recall", self.recall)
            tf.summary.scalar("f1_score", self.f1)
            tf.summary.scalar("loss", self.cost)
            tf.summary.scalar("accuracy", self.accuracy)
            tf.summary.scalar("learning_rate", self.learning_rate)
            self.merged_summary_op = tf.summary.merge_all()
        else:
            self.test_data = test_data

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})

    def train(self):
        # Parameters
        global_start_time = time.time()
        num_batches = int(self.train_data[0].shape[0] / self.batch_size_given)
        validation_cost = []

        history = {'cost': [], 'val_cost': [], 'acc': [], 'val_acc': [], 'f1': [], 'val_f1': []}

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()
        saver = tf.train.Saver()

        try:

            # Start training
            with tf.Session() as sess:
                # Run the initializer
                sess.run(init)
                sess.run(init_local)

                # Tensorboard
                writer_val = tf.summary.FileWriter(join('output', 'tensorboard', '{}'.format(self.id), 'val'))
                writer_train = tf.summary.FileWriter(join('output', 'tensorboard', '{}'.format(self.id), 'train'))

                val_loss_stop_count = 1
                train_loss_stop_count = 1

                # Loop over epochs
                for epoch in range(self.epochs):
                    # Set state
                    current_state = np.zeros((self.hidden_layers, 2, self.batch_size_given, self.num_hidden))

                    # Set training data
                    sess.run(self.train_init_op, feed_dict={self.x: self.train_data[0], self.y: self.train_data[1],
                                                            self.batch_size: self.batch_size_given})

                    # Set learning rate decay
                    new_lr_decay = self.orig_decay ** max(epoch + 1 - self.max_lr_epoch, 0.0)
                    self.assign_lr(sess, self.lr_given * new_lr_decay)

                    # Loop over batches
                    for batch in range(num_batches):
                        t_cost, _, current_state, t_acc, t_prec, t_recall, t_f1_score, t_tp, t_tn, t_fp, t_fn, t_summary, t_lr, t_dropout, t_predicted_labels, t_labels = \
                            sess.run([self.cost, self.train_op, self.state, self.accuracy, self.precision, self.recall,
                                      self.f1, self.tp, self.tn, self.fp, self.fn, self.merged_summary_op,
                                      self.learning_rate, self.dropout, self.predicted_labels, self.labels],
                                     feed_dict={self.init_state: current_state, self.dropout: self.dropout_given})

                        writer_train.add_summary(t_summary, (epoch * num_batches) + batch)

                        t_auc_macro = 0
                        t_auc_micro = 0
                        try:
                            t_auc_macro = roc_auc_score(t_labels, t_predicted_labels)
                        except ValueError:
                            print(sys.exc_info()[0])
                        try:
                            t_auc_micro = roc_auc_score(t_labels, t_predicted_labels, average='micro')
                        except ValueError:
                            print(sys.exc_info()[0])

                        if batch % self.display_step == 0 or batch == 1:
                            # Calculate metrics per batch
                            train_eval_values = {'t_cost': t_cost, 't_lr': t_lr, 't_acc': t_acc, 't_tp': t_tp, 't_tn': t_tn,
                                                 't_fp': t_fp, 't_fn': t_fn, 't_prec': t_prec, 't_recall': t_recall, 't_f1': t_f1_score,
                                                 't_dropout': t_dropout, 't_auc_macro': t_auc_macro, 't_auc_micro': t_auc_micro}
                            print(
                                "Step: {:5d} | Accuracy: {:.4f} | LR: {:.8f} | Cost: {:.6f} | tp: {:5d} | tn: {:5d} | fp: {:5d} | fn: {:5d} | "
                                "Prec: {:.4f} | Recall: {:.4f} | F1: {:.4f} | Drop: {:.3f} | AUC Mac: {:.4f} | AUC Mic: {:.4f}".format(
                                    batch, t_acc, t_lr, t_cost, t_tp, t_tn, t_fp, t_fn, t_prec, t_recall, t_f1_score, t_dropout, t_auc_macro, t_auc_micro))

                    ############################################ EPOCH FINISHED ############################################
                    print("Epoch Finished {}/{}! \n".format(epoch + 1, self.epochs))
                    writer_train.add_summary(t_summary, ((epoch + 1) * num_batches))
                    writer_train.flush()

                    # Start validation
                    current_state = np.zeros((self.hidden_layers, 2, len(self.validation_data[0]), self.num_hidden))
                    sess.run(self.validation_init_op,
                             feed_dict={self.x: self.validation_data[0], self.y: self.validation_data[1],
                                        self.batch_size: len(self.validation_data[0])})

                    v_cost, v_acc, v_prec, v_recall, v_f1_score, v_tp, v_tn, v_fp, v_fn, v_summary, v_dropout, v_predicted_labels = \
                        sess.run([self.cost, self.accuracy, self.precision, self.recall,
                                  self.f1, self.tp, self.tn, self.fp, self.fn, self.merged_summary_op, self.dropout, self.predicted_labels],
                                 feed_dict={self.init_state: current_state})
                    writer_val.add_summary(v_summary, ((epoch + 1) * num_batches))
                    writer_val.flush()

                    v_auc_macro = 0
                    v_auc_micro = 0
                    try:
                        v_auc_macro = roc_auc_score(self.validation_data[1], v_predicted_labels)
                    except ValueError:
                        print(sys.exc_info()[0])
                    try:
                        v_auc_micro = roc_auc_score(self.validation_data[1], v_predicted_labels, average='micro')
                    except ValueError:
                        print(sys.exc_info()[0])


                    validation_eval_values = {'v_cost': v_cost, 'v_acc': v_acc, 'v_tp': v_tp, 'v_tn': v_tn, 'v_fp': v_fp,
                                              'v_fn': v_fn, 'v_prec': v_prec, 'v_recall': v_recall, 'v_f1': v_f1_score,
                                              'v_dropout': v_dropout, 'v_auc_macro': v_auc_macro, 'v_auc_micro': v_auc_micro}

                    print(
                        "VALIDATION: \nStep: {:5d} | Accuracy: {:.4f} | Cost: {:.6f} | tp: {:5d} | tn: {:5d} | fp: {:5d} | fn: {:5d} | Prec: {:.4f} "
                        "| Recall: {:.4f} | F1: {:.4f} | Drop: {:.3f} | AUC Mac: {:.4f} | AUC Mic: {:.4f} \n".format(
                            epoch, v_acc, v_cost, v_tp, v_tn, v_fp, v_fn, v_prec, v_recall, v_f1_score, v_dropout, v_auc_macro, v_auc_micro))

                    print(repr(self.validation_data[1][0:5]))
                    print("---------------------------")
                    print()
                    print(repr(v_predicted_labels[0:5]))

                    # Add metrics to history
                    history['cost'].append(t_cost)
                    history['val_cost'].append(v_cost)
                    history['acc'].append(t_acc)
                    history['val_acc'].append(v_acc)
                    history['f1'].append(t_f1_score)
                    history['val_f1'].append(v_f1_score)

                    # Save if validation cost increased
                    if epoch != 0:
                        print("Debug: val must be smaller than {}".format(validation_cost[-1] - self.val_loss_improv))
                        if v_cost < (validation_cost[-1] - self.val_loss_improv):
                            print("Validation cost improved from {:.6f} to {:.6f}".format(validation_cost[-1], v_cost))
                            save_path = saver.save(sess, join('.', 'tf_models', '{}.ckpt'.format(self.id)))
                            print("Model saved to file: {}\n".format(save_path))
                            val_loss_stop_count = 1
                        else:
                            print(
                                "Validation cost did not improve from {:.6f} to {:.6f} for the {:2d} time".format(validation_cost[-1],
                                                                                                                  v_cost, val_loss_stop_count))
                            if val_loss_stop_count >= self.tolerance:
                                print("Stop training because validation cost did not improve for {} times.".format(val_loss_stop_count))
                                break
                            val_loss_stop_count += 1

                    if t_cost <= self.train_stop:
                        if train_loss_stop_count >= self.tolerance:
                            print("Stop training because training loss is lower than training stop loss for {} times.".format(
                                train_loss_stop_count))
                            break
                        train_loss_stop_count += 1

                    validation_cost.append(v_cost)

        except KeyboardInterrupt:
            print("Interrupted!")

        print("Optimization Finished!")
        train_time = "{0:.2f}".format(time.time() - global_start_time)
        train_result = Result(stopped_epoch=epoch + 1, model_eval_values={**train_eval_values, **{'t_train_time': train_time}})
        validation_result = Result(stopped_epoch=epoch + 1, model_eval_values={**validation_eval_values, **{'v_train_time': train_time}})
        # TODO: Handle train_time only in train_result and ignore in validation_result

        return history, train_result, validation_result

    def test(self):
        '''
        Warning: This method loads the entire test data set into memory and does not iterate over it batch-wise.
        If you face a problem with the memory, a batch-wise method has to be implemented.

        :return:
        '''

        saver = tf.train.import_meta_graph(join('tf_models', '{}.ckpt.meta'.format(self.id)))

        # Start testing
        with tf.Session() as sess:
            graph = tf.get_default_graph()
            self.x = graph.get_tensor_by_name("x:0")
            self.y = graph.get_tensor_by_name("y:0")
            self.batch_size = graph.get_tensor_by_name("batch_size:0")
            self.num_input = tf.get_collection('num_input')[0]
            self.hidden_layers = tf.get_collection('hidden_layers')[0]
            self.num_hidden = tf.get_collection('num_hidden')[0]

            self.test_data = (self.test_data['X'], self.test_data['y'])

            self.train_init_op = graph.get_operation_by_name('train_init_op')

            self.init_state = graph.get_tensor_by_name("init_state:0")
            self.cost = graph.get_tensor_by_name("Loss/cost:0")
            self.accuracy = graph.get_tensor_by_name("accuracy:0")
            self.precision = graph.get_tensor_by_name("precision:0")
            self.recall = graph.get_tensor_by_name("recall:0")
            self.f1 = graph.get_tensor_by_name("f1:0")
            self.tp = graph.get_tensor_by_name("tp:0")
            self.tn = graph.get_tensor_by_name("tn:0")
            self.fp = graph.get_tensor_by_name("fp:0")
            self.fn = graph.get_tensor_by_name("fn:0")
            self.dropout = graph.get_tensor_by_name("dropout:0")

            # Run the initializer
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            load_path = saver.restore(sess, join('tf_models', '{}.ckpt'.format(self.id)))
            print("Model loaded from file: {}".format(load_path))

            current_state = np.zeros((self.hidden_layers, 2, len(self.test_data[0]), self.num_hidden))

            sess.run(self.train_init_op,
                     feed_dict={self.x: self.test_data[0], self.y: self.test_data[1],
                                self.batch_size: len(self.test_data[0])})

            print("Start testing.")
            cost, acc, prec, recall, f1, tp, tn, fp, fn, dropout, correct_pred = \
                sess.run([self.cost, self.accuracy, self.precision, self.recall, self.f1, self.tp, self.tn, self.fp, self.fn, self.dropout,
                          self.correct_pred], feed_dict={self.init_state: current_state})

            print(Fore.LIGHTGREEN_EX)
            print("Test cost: {}, acc: {}, prec: {}, recall: {}, f1: {}, dropout: {}, aoc: {}".format(cost, acc, prec, recall, f1, dropout,
                                                                                                      roc_auc_score(correct_pred, self.test_data[1])))
            print(Style.RESET_ALL)
            print("Testing Finished!")

            test_eval_values = {'cost': cost, 'acc': acc, 'tp': tp, 'tn': tn, 'fp': fp,
                                      'fn': fn, 'prec': prec, 'recall': recall, 'f1': f1,
                                      'dropout': dropout, 'auc': roc_auc_score(correct_pred, self.test_data[1])}

            return Result(model_eval_values=test_eval_values)

############################################################################################################################################
################################################################### MAIN ###################################################################
############################################################################################################################################

def main():
    train = True

    with open(join('..', 'data', 'main_load_data.pkl'), 'rb') as f:
        power_grid_data = pickle.load(f)

    if train:
        ############## PART 1 #################

        train_config_list = []
        percentages = [0.05, 0.1, 0.3, 0.5]
        subset_sizes = [1,4,8,12,16,20,24,28,29]
        norms = [Norm.STANDARD]
        atk_styles = [{'atk_function': 2, 'A': 0.1, 'c': 0}]

        for atk_style in atk_styles:
            for norm in norms:
                for percentage in percentages:
                    for subset_size in subset_sizes:
                        n_atk_subsets = math.ceil(subset_size/2)

                        test_config_list = []
                        train_data_config = DataConfig(main_data=power_grid_data, method_name='create_fdi_X3_y1_se_window', norm=norm,
                                                       subset_size=subset_size, n_atk_subsets=n_atk_subsets, c=atk_style['c'],
                                                       timestep=16, ratio=0.7, P=percentage, atk_function=atk_style['atk_function'], A=atk_style['A'])
                        tf_network_config = TfNetworkConfig(num_input=41, timestep=16, num_hidden=200, num_output=subset_size,
                                                            batch_size=256, epochs=200, orig_decay=0.93, max_lr_epoch=1,
                                                            hidden_layers=2, lr_given=0.001, dropout_given=0.7, tolerance=2, display_step=5,
                                                            train_stop=0.001, val_loss_improv=0.001)

                        # Sum them up
                        train_config = TrainConfig(train_data_config=train_data_config, network_config=tf_network_config)

                        train_config_dict = {'train_config': train_config, 'test_config_list': test_config_list}
                        train_config_list.append(train_config_dict)

        print("{} different configs.".format(len(train_config_list)))

        for config in train_config_list:

            # TRAINING FIRST

            # Generate/retrieve the data
            train_data = {}
            validation_data = {}
            rd = config['train_config'].train_data_config.retrieve_data_set()

            #train_data['X'] = rd['X_train']
            #train_data['y'] = rd['y_train']
            #validation_data['X'] = rd['X_test']
            #validation_data['y'] = rd['y_test']

            n = 10000
            n_train = int(n * 0.7)
            n_test = int(n * 0.3)

            train_data['X'] = rd['X_train'][0:n_train]
            train_data['y'] = rd['y_train'][0:n_train]
            validation_data['X'] = rd['X_test'][n:n + n_test]
            validation_data['y'] = rd['y_test'][n:n + n_test]

            print(train_data['y'][0:10])
            # Analyse the data sets
            config['train_config'].train_data_analysis = Analyzer.X_y_data(train_data)
            config['train_config'].valid_data_analysis = Analyzer.X_y_data(validation_data)

            # Run network / evaluate the model
            tf.reset_default_graph()
            m1 = Model(id=config['train_config'].id, is_training=True,
                       network_conf=config['train_config'].network_config,
                       train_data=train_data,
                       validation_data=validation_data)
            history, config['train_config'].train_result, config['train_config'].validation_result = m1.train()

            # Save everything to CSV file
            config['train_config'].dump_to_csv(os.path.join('output', 'tf_10kfinal.csv'))

            # Create and save images
            config['train_config'].dump_images(history)

            # # TEST SECOND
            #
            # for test_config in config['test_config_list']:
            #     test_data = {}
            #     test_data['X'], test_data['y'] = test_config.test_data_config.retrieve_single_dataset()
            #
            #     test_config.test_data_analysis = Analyzer.X_y_data(test_data)
            #
            #     # test_config.test_result = m1.test(test_data=test_data)
            #     tf.reset_default_graph()
            #     test_config.test_result = Model(id=test_config.model_id, is_training=False, test_data=test_data).test()
            #
            #     test_config.dump_to_csv(join('output', 'tf_test_timesteps.csv'))

        ############## PART 2 #################

        # train_config_list = []
        # # subset_sizes = [5]
        # timesteps = [1, 5, 16, 32, 48, 64]
        #
        # for timestep in timesteps:
        #     test_config_list = []
        #     train_data_config = DataConfig(main_data=power_grid_data, method_name='create_fdi_X3_y1_se_window', norm=True,
        #                                    subset_size=29, n_atk_subsets=29,
        #                                    timestep=timestep, c=0.2, random=1, ratio=0.7, P=0.1)
        #     tf_network_config = TfNetworkConfig(num_input=41, timestep=timestep, num_hidden=200, num_output=29,
        #                                         batch_size=1280, epochs=200, orig_decay=0.93, max_lr_epoch=1,
        #                                         hidden_layers=2, lr_given=0.001, dropout_given=0.6, tolerance=4, display_step=5,
        #                                         train_stop=0.001)
        #
        #
        #     # Sum them up
        #     train_config = TrainConfig(train_data_config=train_data_config, network_config=tf_network_config)
        #
        #     train_config_dict = {'train_config': train_config, 'test_config_list': test_config_list}
        #     train_config_list.append(train_config_dict)
        #
        # for config in train_config_list:
        #
        #     # TRAINING FIRST
        #
        #     # Generate/retrieve the data
        #     train_data = {}
        #     validation_data = {}
        #     train_data['X'], validation_data['X'], train_data['y'], validation_data[
        #         'y'] = config['train_config'].train_data_config.retrieve_splitted_dataset()
        #
        #     # Analyse the data sets
        #     config['train_config'].train_data_analysis = Analyzer.X_y_data(train_data)
        #     config['train_config'].valid_data_analysis = Analyzer.X_y_data(validation_data)
        #
        #     # Run network / evaluate the model
        #     tf.reset_default_graph()
        #     m1 = Model(id=config['train_config'].id, is_training=True,
        #                network_conf=config['train_config'].network_config,
        #                train_data=train_data,
        #                validation_data=validation_data)
        #     history, config['train_config'].train_result, config['train_config'].validation_result = m1.train()
        #
        #     # Save everything to CSV file
        #     config['train_config'].dump_to_csv(os.path.join('output', 'tf_train_var_perc_timestep.csv'))
        #
        #     # Create and save images
        #     config['train_config'].dump_images(history)
        #
        #     # TEST SECOND
        #
        #     for test_config in config['test_config_list']:
        #         test_data = {}
        #         test_data['X'], test_data['y'] = test_config.test_data_config.retrieve_single_dataset()
        #
        #         test_config.test_data_analysis = Analyzer.X_y_data(test_data)
        #
        #         # test_config.test_result = m1.test(test_data=test_data)
        #         tf.reset_default_graph()
        #         test_config.test_result = Model(id=test_config.model_id, is_training=False, test_data=test_data).test()
        #
        #         test_config.dump_to_csv(join('output', 'tf_test_96.csv'))

        ############## PART 4 #################

        # train_config_list = []
        # subset_sizes = [1, 4, 8, 12, 16, 20, 24, 28, 29]
        # # subset_sizes = [1]
        #
        # for subset_size in subset_sizes:
        #     test_config_list = []
        #     train_data_config = DataConfig(main_data=power_grid_data, method_name='create_fdi_X3_y1_se_window', norm=True,
        #                                    atk_index=[0], subset_size=subset_size, n_atk_subsets=subset_size,
        #                                    timestep=1, c=2, random=1, ratio=0.7)
        #     tf_network_config = TfNetworkConfig(num_input=41, timestep=1, num_hidden=200, num_output=subset_size,
        #                                         batch_size=1280, epochs=200, orig_decay=0.93, max_lr_epoch=1,
        #                                         hidden_layers=2, lr_given=0.001, dropout_given=0.6, tolerance=4, display_step=5,
        #                                         train_stop=0.001)
        #     test_data_config_1 = DataConfig(main_data=power_grid_data, method_name='create_fdi_X3_y1_se_window', norm=True,
        #                                   atk_index=[0], subset_size=subset_size, n_atk_subsets=subset_size,
        #                                   timestep=1, c=0.2, random=1)
        #     test_data_config_2 = DataConfig(main_data=power_grid_data, method_name='create_fdi_X3_y1_se_window', norm=True,
        #                                   atk_index=[0], subset_size=subset_size, n_atk_subsets=subset_size,
        #                                   timestep=1, c=0.2, random=1)
        #     test_data_config_3 = DataConfig(main_data=power_grid_data, method_name='create_fdi_X3_y1_se_window', norm=True,
        #                                   atk_index=[0], subset_size=subset_size, n_atk_subsets=subset_size,
        #                                   timestep=1, c=2, random=1)
        #
        #     # Sum them up
        #     train_config = TrainConfig(train_data_config=train_data_config, network_config=tf_network_config)
        #
        #     test_config_list.append(TestConfig(model_id=train_config.id, test_data_config=test_data_config_1))
        #     test_config_list.append(TestConfig(model_id=train_config.id, test_data_config=test_data_config_2))
        #     test_config_list.append(TestConfig(model_id=train_config.id, test_data_config=test_data_config_3))
        #
        #     train_config_dict = {'train_config': train_config, 'test_config_list': test_config_list}
        #     train_config_list.append(train_config_dict)
        #
        # for config in train_config_list:
        #
        #     # TRAINING FIRST
        #
        #     # Generate/retrieve the data
        #     train_data = {}
        #     validation_data = {}
        #     train_data['X'], validation_data['X'], train_data['y'], validation_data[
        #         'y'] = config['train_config'].train_data_config.retrieve_splitted_dataset()
        #
        #     # Analyse the data sets
        #     config['train_config'].train_data_analysis = Analyzer.X_y_data(train_data)
        #     config['train_config'].valid_data_analysis = Analyzer.X_y_data(validation_data)
        #
        #     # Run network / evaluate the model
        #     tf.reset_default_graph()
        #     m1 = Model(id=config['train_config'].id, is_training=True,
        #                network_conf=config['train_config'].network_config,
        #                train_data=train_data,
        #                validation_data=validation_data)
        #     history, config['train_config'].train_result, config['train_config'].validation_result = m1.train()
        #
        #     # Save everything to CSV file
        #     config['train_config'].dump_to_csv(os.path.join('output', 'tf_train04_3.csv'))
        #
        #     # Create and save images
        #     config['train_config'].dump_images(history)
        #
        #     # TEST SECOND
        #
        #     for test_config in config['test_config_list']:
        #         test_data = {}
        #         test_data['X'], test_data['y'] = test_config.test_data_config.retrieve_single_dataset()
        #
        #         test_config.test_data_analysis = Analyzer.X_y_data(test_data)
        #
        #         # test_config.test_result = m1.test(test_data=test_data)
        #         tf.reset_default_graph()
        #         test_config.test_result = Model(id=test_config.model_id, is_training=False, test_data=test_data).test()
        #
        #         test_config.dump_to_csv(join('output', 'tf_test04_3.csv'))

    ############################################## COMPLETE DIFFERENT TEST PART #####################################################
    else:
        test_config_list = []

        test_data = {}
        model_ids = ['6542fb76-58dd-42cb-983e-edfc4b782fb6']
        subset_sizes = [1]

        for model_id in model_ids:
            for subset_size in subset_sizes:
                test_data_config = DataConfig(main_data=power_grid_data, method_name='create_fdi_X3_y1_se', norm=True,
                                           atk_index=[0], subset_size=subset_size, n_atk_subsets=subset_size,
                                           timestep=1, c=0.2, random=1, ratio=0.7)

                test_config_list.append(TestConfig(model_id=model_id, test_data_config=test_data_config))

        for config in test_config_list:
            test_data['X'], test_data['y'] = config.test_data_config.retrieve_single_dataset()

            config.test_data_analysis = Analyzer.X_y_data(test_data)

            config.test_result = Model(id=config.model_id, is_training=False, test_data=test_data).test()

            config.dump_to_csv(join('output', 'lala.csv'))


if __name__ == '__main__':
    main()
