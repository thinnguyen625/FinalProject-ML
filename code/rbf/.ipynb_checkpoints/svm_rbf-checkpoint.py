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
    print('     -', single_rbf.name, 'in', single_rbf.fit_time, 'seconds')
    return single_rbf

def single_predict(params):
    '''
    Predict on all the three data parts (train, valid, test) using the model attribute of single_rbf object
    -----
    Params:
        - params: a 7-element tuple (single_rbf object, tr_X, tr_Y, va_X, va_Y, te_X, te_Y)
    -----
    Returns:
        - single_rbf object
    
    '''
    single_rbf, tr_X, tr_Y, va_X, va_Y, te_X, te_Y = params
    single_rbf.tr_err = 1 - single_rbf.model.score(tr_X, tr_Y)
    single_rbf.va_err = 1 - single_rbf.model.score(va_X, va_Y)
    single_rbf.te_err = 1 - single_rbf.model.score(te_X, te_Y)
    print('     -', single_rbf.name + ': tr_err %f'%single_rbf.tr_err + ', va_err %f'%single_rbf.va_err + ', te_err %f'%single_rbf.te_err)
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
        - te_err: testing error
        - is_best_tr: whether this is the best model so far in the training phase for an individual C
        - is_best_te: whether this is the best model so far in the testing phase for an individual C
    '''
    def __init__(self, C, gamma, decision_function_shape, res_path):
        self.model = SVC(kernel='rbf', C=C, gamma=gamma, decision_function_shape=decision_function_shape)
        self.name = 'rbf_' + str(C) + '_' + str(gamma) + '_' + decision_function_shape
        self.path = res_path + r'/' + self.name + '.pkl'
        self.fit_time = None
        self.tr_err = None
        self.va_err = None
        self.te_err = None
        self.is_best_tr = False
        self.is_best_te = False

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
        - train_X, train_Y, val_X, val_Y, test_X, test_Y: the data parts
        - single_rbf_list: the list of single_rbf instances
        - best_tr_idx: index in single_rbf_list of the model which has best training error for an individual C
        - best_tr_idx: index in single_rbf_list of the model which has best testing error for an individual C
    '''
    def __init__(self, C, gammas=cfg._gammas, decision_function_shape='ovo', wd=cfg._wd, res_path=cfg._res_path, n_jobs=2):
        self.C = C
        self.gammas = gammas
        self.decision_function_shape = decision_function_shape
        self.wd = wd
        self.res_path = res_path
        self.n_jobs = n_jobs if n_jobs == 2 else 2
        self.name = 'rbf_' + str(self.C) + '_' + self.decision_function_shape

        self.train_X, self.train_Y = None, None
        self.val_X, self.val_Y = None, None
        self.test_X, self.test_X = None, None
        self.read_mnist()

        self.single_rbf_list = []
        for gamma in gammas:
            self.single_rbf_list.append(single_rbf(self.C, gamma, self.decision_function_shape, self.res_path))

        self.best_tr_idx = None
        self.best_te_idx = None

    def run(self, n_samples):
        '''
        Fit 2 models at a time then predict on the whole dataset and update the best models so far for an individual C
            -----
            Params:
                - n_samples: the number of samples to be used for training, validating and testing (must less than val_Y.shape[0] and test_Y.shape[0])
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
                                                                    repeat(self.train_X[:n_samples, :]), repeat(self.train_Y[:n_samples]))).get()
            pool1.terminate()
            #------------
            print(' Predicting...')
            pool2 = Pool(self.n_jobs)
            self.single_rbf_list[i], self.single_rbf_list[i + 1] = pool2.map_async(single_predict,
                                                                    zip([self.single_rbf_list[i], self.single_rbf_list[i + 1]],
                                                                    repeat(self.train_X[:n_samples, :]), repeat(self.train_Y[:n_samples]),
                                                                    repeat(self.val_X[:n_samples, :]), repeat(self.val_Y[:n_samples]),
                                                                    repeat(self.test_X[:n_samples, :]), repeat(self.test_Y[:n_samples]))).get()
            pool2.terminate()

            better_tr_idx, better_te_idx = self.get_better_idx(i, i + 1)
            self.update_best(better_tr_idx, better_te_idx)
            print(' --- --- ---')
        print('Best train:', self.single_rbf_list[self.best_tr_idx].name)
        print('Best test:', self.single_rbf_list[self.best_te_idx].name)
        self.make_df()
        return self

    def get_better_idx(self, i, j):
        '''
        Compare the result of 2 models
        If equal, prefer the model which has smaller gamma
            -----
            Params:
                - i: index of single_rbf instance in self.single_rbf_list
                - j: index of the other single_rbf to be compared to
            -----
            Returns:
                - better_tr_idx: either i or j
                - better_te_idx: either i or j (can be the same as better_tr_idx)
        '''
        better_tr_idx = None
        if self.single_rbf_list[i].tr_err <= self.single_rbf_list[j].tr_err:
            better_tr_idx = i
        else:
            better_tr_idx = j

        better_te_idx = None
        if self.single_rbf_list[i].te_err <= self.single_rbf_list[j].te_err:
            better_te_idx = i
        else:
            better_te_idx = j
        return better_tr_idx, better_te_idx

    def update_best(self, tr_candidate_idx, te_candidate_idx):
        '''
        Compare models' result and update the best model so far for an individual C
        -----
        Params:
            - tr_candidate_idx: the candidate's index to be compared to the best model in training so far
            - te_candidate_idx: the candidate's index to be compared to the best model in testing so far
        '''
        #The first run
        #Save both the models
        #Warning: I forgot to update is_best_tr and is_best_te status (will be fixed later)
        if tr_candidate_idx <= 1 or te_candidate_idx <= 1:
            self.best_tr_idx = tr_candidate_idx
            tr_model = self.single_rbf_list[tr_candidate_idx].model
            tr_path = self.single_rbf_list[tr_candidate_idx].path
            dump(tr_model, open(tr_path, 'wb'))
            
            self.best_te_idx = te_candidate_idx
            te_model = self.single_rbf_list[te_candidate_idx].model
            te_path = self.single_rbf_list[te_candidate_idx].path
            dump(te_model, open(te_path, 'wb'))
            return

        #Compare the candidate for the best training model so far
        #If True:
        #Delete the model file of the current best if the candidate is better and save the candidate model
        #Change is_best_tr status for candidate to True and for the current best to False
        #Update best_tr_idx
        if self.single_rbf_list[tr_candidate_idx].tr_err <= self.single_rbf_list[self.best_tr_idx].tr_err:
            if os.path.exists(self.single_rbf_list[self.best_tr_idx].path):
                os.remove(self.single_rbf_list[self.best_tr_idx].path)
            if not os.path.exists(self.single_rbf_list[tr_candidate_idx].path):
                model = self.single_rbf_list[tr_candidate_idx].model
                path = self.single_rbf_list[tr_candidate_idx].path
                dump(model, open(path, 'wb'))
            
            self.single_rbf_list[self.best_tr_idx].is_best_tr = False
            self.single_rbf_list[tr_candidate_idx].is_best_tr = True
            self.best_tr_idx = tr_candidate_idx
            
        #Compare the candidate for the best testing model so far
        #If True:
        #Delete the model file of the current best if the candidate is better and save the candidate model
        #Change is_best_te status for candidate to True and for the current best to False
        #Update best_te_idx
        if self.single_rbf_list[te_candidate_idx].te_err <= self.single_rbf_list[self.best_te_idx].te_err:
            if os.path.exists(self.single_rbf_list[self.best_te_idx].path):
                os.remove(self.single_rbf_list[self.best_te_idx].path)
            if not os.path.exists(self.single_rbf_list[te_candidate_idx].path):
                model = self.single_rbf_list[te_candidate_idx].model
                path = self.single_rbf_list[te_candidate_idx].path
                dump(model, open(path, 'wb'))

            self.single_rbf_list[self.best_te_idx].is_best_te = False
            self.single_rbf_list[te_candidate_idx].is_best_te = True
            self.best_te_idx = te_candidate_idx
    
    def make_df(self):
        '''
        Make a pandas DataFrame from the results after training all the models for an individual C
        '''
        df_data = {'gamma': self.gammas, 'fit_time': [],
                    'tr_err': [], 'va_err': [], 'te_err': [],
                    'best_tr': [], 'best_te': [],
                    'name': []}
        for s in self.single_rbf_list:
            df_data['fit_time'].append(s.fit_time)
            df_data['tr_err'].append(s.tr_err)
            df_data['va_err'].append(s.va_err)
            df_data['te_err'].append(s.te_err)
            df_data['best_tr'].append(s.is_best_tr)
            df_data['best_te'].append(s.is_best_te)
            df_data['name'].append(s.name)
        df = pd.DataFrame(data=df_data)
        df.to_csv(self.res_path + r'/' + self.name + '.csv', index=False)

    def read_mnist(self):
        '''
        Read the mnist.pkl.gz
        '''
        mnist_file = self.wd + r'/mnist.pkl.gz'
        f = gzip.open(mnist_file, 'rb')
        train_data, val_data, test_data = load(f, encoding='latin1')
        f.close()    
        self.train_X, self.train_Y = train_data
        self.val_X, self.val_Y = val_data
        self.test_X, self.test_Y = test_data
        