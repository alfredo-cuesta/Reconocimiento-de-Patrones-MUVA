import pandas as pd
import numpy as np


def mnist_features (data, theta=0.1):
    # data: dataframe
    #   rows = examples
    #   columns = raw attributes, none is label + no column names
    # theta: parameter of the feature extraction
    # features extracted: 
    #   'width','W_max1','W_max2','W_max3',
    #   'height','H_max1','H_max2','H_max3',
    #   'area','w_vs_h'
    #
    features = np.zeros([data.shape[0], 10]) #<- allocate memory with zeros
    data = data.values.reshape([data.shape[0],28,28]) 
    #-> axis 0: id of instance, axis 1: width(cols) , axis 2: height(rows)
    for k in range(data.shape[0]):
        #..current image 
        x = data[k,:,:]
        #--width feature
        sum_cols = x.sum(axis=0) #<- axis0 of x, not of data!!
        indc = np.argwhere(sum_cols > theta * sum_cols.max())
        col_3maxs = np.argsort(sum_cols)[-3:] 
        features[k,0] = indc[-1] - indc[0]
        features[k,1:4] = col_3maxs
        #--height feature
        sum_rows = x.sum(axis=1) #<- axis1 of x, not of data!!
        indr = np.argwhere(sum_rows > theta * sum_rows.max())
        features[k,4] = indr[-1] - indr[0]
        row_3maxs = np.argsort(sum_rows)[-3:] 
        features[k,5:8] = row_3maxs
    #--area
    features[:,8] = features[:,0] * features[:,4]
    #--ratio W/H
    features[:,9] = features[:,0] / features[:,4]
    col_names = ['width','W_max1','W_max2','W_max3','height','H_max1','H_max2','H_max3','area','w_vs_h']
    #
    return pd.DataFrame(features,columns = col_names) 

def join_features_labels(X0,X1):
    # X0, X1: dataframes
    # returns a dataframe with X0 and X1 together and 
    #   a new column 'label' 
    #   = 0 for rows from X0
    #   = 1 for rows from X1
    #
    Y0 = pd.DataFrame(np.zeros(X0.shape[0]),columns=['label'])
    XY0 = pd.concat([X0,Y0],axis=1)
    Y1 = pd.DataFrame(np.ones(X1.shape[0]),columns=['label'])
    XY1 = pd.concat([X1,Y1],axis=1)
    return pd.concat([XY0,XY1],axis=0,ignore_index=True)

def jitter(X,sigma=0.2):
    random_sign = (-1.)**np.random.randint(1,high=3,size=X.shape)
    return X + np.random.normal(0,sigma,size=X.shape)*random_sign 

def single_stratified_split(X,Y,test_size=.2, random_state=1234):
    # X is the dataframe with examples (rows) and attributes (columns)
    # Y is the dataframe with labels
    # test_size is the percentage of X separated; default is 0.2
    # random_state is a seed for pseudorandom generation
    # returns 
    #   X_train, Y_train = dataframes of (1-test_size)% of the X and Y
    #   X_test, Y_test = dataframes of test_size% of the X and Y
    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    split_ix = splitter.split(X,Y)
    for train_ix, test_ix in split_ix:
        X_train = X.loc[train_ix].reset_index(drop=True)
        Y_train = Y.loc[train_ix].reset_index(drop=True)
        X_test  = X.loc[test_ix].reset_index(drop=True)
        Y_test  = Y.loc[test_ix].reset_index(drop=True)
    return X_train, Y_train, X_test, Y_test

