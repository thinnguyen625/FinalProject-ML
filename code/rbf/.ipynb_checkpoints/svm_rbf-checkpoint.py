import os
import time
import gzip
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from itertools import repeat
from pickle import dump, load
from multiprocessing import Pool

class cfg:
    '''
    Common used infomation:
    -----
    Attributes:
        - _gammas: the gammas to initialize each RBF kernel model
        - _wd: the current working directory (must contains mnist.pkl.gz)
        - _res_path: the path to the result folder (contains best model files and .csv files for each C and also analysis images)
    '''
    _gammas = [0.0005, 0.005, 0.025, 0.05, 0.25, 0.5]
    _wd = os.getcwd()
    _res_path = os.path.join(_wd, 'res')
    if not os.path.exists(_res_path):
        os.mkdir(_res_path)

def single_fit(params):
    '''
    Fit a single RBF kernel model to the train_X - train_Y data
    -----
    Params:
        - params: a 3-element tuple (single_rbf object, train_X, train_Y)
    -----
    Returns:
        - single_rbf object
    '''
    single_rbf, train_X, train_Y = params
    start = time.time()
    single_rbf.model.fit(train_X, train_Y)
    single_rbf.fit_time = time.time() - start
    return single_rbf

def single_score(params):
    '''
    Predict on a data part (train or valid) using the 'model' attribute of single_rbf object
    -----
    Params:
        - params: a 4-element tuple (single_rbf object, _X, _Y, phase)
    -----
    Returns:
        - single_rbf object
    
    '''
    single_rbf, _X, _Y, phase = params
    err = 1 - single_rbf.model.score(_X, _Y)
    if phase == 'tr':
        single_rbf.tr_err = err
    elif phase == 'va':
        single_rbf.va_err = err
        
    return single_rbf

class single_rbf:
    '''
    Container of an RBF kernel model and its details
    -----
    Attributes:
        - model: the SVC (RBF kernel) object
        - name: 'rbf_' + str(C) + '_' + str(gamma) + '_' + decision_function_shape
        - path: res_path + r'/' + self.name + '.pkl'
        - fit_time: fit time of each model
        - tr_err: training error
        - va_err: validating error
        - is_best_va: whether this is the best model so far in the validating phase for an individual C
    '''
    def __init__(self, C, gamma, decision_function_shape, res_path):
        self.model = SVC(kernel='rbf', C=C, gamma=gamma, decision_function_shape=decision_function_shape)
        self.name = 'rbf_' + str(C) + '_' + str(gamma) + '_' + decision_function_shape
        self.path = res_path + r'/' + self.name + '.pkl'
        self.fit_time = None
        self.tr_err = None
        self.va_err = None
        self.is_best_va = False

class multi_rbf:
    '''
    Container of all the single RBF kernel models of an individual C
    -----
    Attributes:
        - C: the C param to pass to all the models
        - gammas: the gamma params to pass to each of the models (default: inherit from cfg)
        - decision_function_shape: the classification strategy (default: 'ovo')
        - wd: the working directory containing mnist.pkl.gz (default: inherit from cfg)
        - res_path: the path to the result model (default: inherit from cfg)
        - n_jobs: number of subprocess to run (default: 2; by now, this code only accepts n_jobs = 2)
        - name: 'rbf_' + str(self.C) + '_' + self.decision_function_shape
        - single_rbf_list: the list of single_rbf instances
        - best_va_idx: index in single_rbf_list of the model which has best validating error for an individual C
    '''
    def __init__(self, C, gammas=cfg._gammas, decision_function_shape='ovo', wd=cfg._wd, res_path=cfg._res_path, n_jobs=2):
        self.C = C
        self.gammas = gammas
        self.decision_function_shape = decision_function_shape
        self.wd = wd
        self.res_path = res_path
        self.n_jobs = n_jobs if n_jobs == 2 else 2
        self.name = 'rbf_' + str(self.C) + '_' + self.decision_function_shape

        self.single_rbf_list = []
        for gamma in gammas:
            self.single_rbf_list.append(single_rbf(self.C, gamma, self.decision_function_shape, self.res_path))

        self.best_va_idx = None

    def run(self, n_samples, train_X, train_Y, val_X, val_Y):
        '''
        Fit 2 models at a time then score on the training and validating set and update the best models so far for an individual C
            -----
            Params:
                - n_samples: the number of samples to be used for training, validating (must less than val_Y.shape[0])
                - train_X, train_Y: training set
                - val_X, val_Y: validation set
            -----
            Returns:
                - self
        '''
        print('Fitting %d RBF SVM models with C = %f' % (len(self.gammas), self.C))
        for i in range(0, len(self.gammas), self.n_jobs):
            print(' Fitting %d models...' % self.n_jobs)
            pool1 = Pool(self.n_jobs)
            self.single_rbf_list[i], self.single_rbf_list[i + 1] = pool1.map_async(single_fit,
                                                                    zip([self.single_rbf_list[i], self.single_rbf_list[i + 1]],
                                                                        repeat(train_X[:n_samples, :]), repeat(train_Y[:n_samples]))).get()
            pool1.terminate()
            print('     -', self.single_rbf_list[i].name, 'in', self.single_rbf_list[i].fit_time, 'seconds')
            print('     -', self.single_rbf_list[i + 1].name, 'in', self.single_rbf_list[i + 1].fit_time, 'seconds')
            #------------
            print(' Scoring...')
            pool2 = Pool(self.n_jobs)
            self.single_rbf_list[i], self.single_rbf_list[i + 1] = pool2.map_async(single_score,
                                                                            zip([self.single_rbf_list[i], self.single_rbf_list[i + 1]],
                                                                                repeat(val_X[:n_samples, :]), repeat(val_Y[:n_samples]),
                                                                                ['va'] * self.n_jobs)).get()
            pool2.terminate()
            print('     -', self.single_rbf_list[i].name + ': va_err %f' % self.single_rbf_list[i].va_err)
            print('     -', self.single_rbf_list[i + 1].name + ': va_err %f' % self.single_rbf_list[i + 1].va_err)
            
            self.update_best(i, i + 1)

            pool3 = Pool(self.n_jobs)
            self.single_rbf_list[i], self.single_rbf_list[i + 1] = pool3.map_async(single_score,
                                                                            zip([self.single_rbf_list[i], self.single_rbf_list[i + 1]],
                                                                                repeat(train_X[:n_samples, :]), repeat(train_Y[:n_samples]),
                                                                                ['tr'] * self.n_jobs)).get()
            pool3.terminate()
            print('     -', self.single_rbf_list[i].name + ': tr_err %f' % self.single_rbf_list[i].tr_err)
            print('     -', self.single_rbf_list[i + 1].name + ': tr_err %f' % self.single_rbf_list[i + 1].tr_err)
            
            print(' --- --- ---')
            self.make_df()

        print('Best validate:', self.single_rbf_list[self.best_va_idx].name)
        return self

    def update_best(self, i, j):
        '''
        Compare models' result and update the best model so far for an individual C
        -----
        Params:
            - i, j: index of the single_rbf object in self.single_rbf_list
        '''
        i_va_err = self.single_rbf_list[i].va_err
        j_va_err = self.single_rbf_list[j].va_err
        
        #The first run
        if j <= 1:
            self.best_va_idx = i if i_va_err <= j_va_err else j
            va_model = self.single_rbf_list[self.best_va_idx].model
            va_path = self.single_rbf_list[self.best_va_idx].path
            self.single_rbf_list[self.best_va_idx].is_best_va = True
            dump(va_model, open(va_path, 'wb'))
            return

        prev_best_va_err = self.single_rbf_list[self.best_va_idx].va_err
        
        better_va_idx = i if i_va_err <= j_va_err else j
        cur_best_va_idx = better_va_idx if self.single_rbf_list[better_va_idx].va_err <= prev_best_va_err else self.best_va_idx
            
        if cur_best_va_idx == self.best_va_idx:
            return
        else:
            if os.path.exists(self.single_rbf_list[self.best_va_idx].path):
                os.remove(self.single_rbf_list[self.best_va_idx].path)
            if not os.path.exists(self.single_rbf_list[cur_best_va_idx].path):
                model = self.single_rbf_list[cur_best_va_idx].model
                path = self.single_rbf_list[cur_best_va_idx].path
                dump(model, open(path, 'wb'))
            self.single_rbf_list[self.best_va_idx].is_best_va = False
            self.single_rbf_list[cur_best_va_idx].is_best_va = True
            self.best_va_idx = cur_best_va_idx

    def make_df(self):
        '''
        Make a pandas DataFrame from the results after training all the models for an individual C
        '''
        df_data = {'gamma': self.gammas, 'fit_time': [],
                    'tr_err': [], 'va_err': [],
                    'best_va': [],
                    'name': []}
        for s in self.single_rbf_list:
            df_data['fit_time'].append(s.fit_time)
            df_data['tr_err'].append(s.tr_err)
            df_data['va_err'].append(s.va_err)
            df_data['best_va'].append(s.is_best_va)
            df_data['name'].append(s.name)
        df = pd.DataFrame(data=df_data)
        df.to_csv(self.res_path + r'/' + self.name + '.csv', index=False)
        