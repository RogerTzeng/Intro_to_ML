# -*- coding: utf-8 -*-

"""
===================== PLEASE WRITE HERE =====================
- Practice using SVM and Decision Tree classifier
- Use SVM and Decision Tree classifier to train a model to classify three class of Iris plants, and plot the Decision boundary surface.
- 曾靖桐 Date:2021.11.15
===================== PLEASE WRITE HERE =====================
"""

# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
import matplotlib.pyplot as plt


# Load data file and return two numpy arrays, including x: features and 
# y: labels
def load_data(path):
    """
    - load data and assign the features and labels
    """
    print('Loading data...')
    
    data=pd.read_csv(path, sep=",", header= None)
        
    # Print the number of samples
    n_samples = len(data)
    print('Number of samples:', n_samples)
    
    # Split the data into features and labels
    x = data.values[:, :-1].tolist()
    y = data.values[:,-1].tolist()
    
    
    return data, x, y 


# Split the data into training set and testing set
def split_dataset(x, y, testset_portion):
    print('Split dataset.')
    """
    - split the data  into a training set and a testing 
    set according to the 'testset_portion'. That is, the testing set will 
    account for 'testset_portion' of the overall data. You may use the function
    'sklearn.model_selection.train_test_split'.    
    """
    # ===================== PLEASE WRITE HERE =====================
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testset_portion, random_state=42)
    
    # ===================== PLEASE WRITE HERE =====================
    
    return x_train, x_test, y_train, y_test    

# Train Decision tree classifier on x_train and y_train
def train_DT(x_train, y_train, depth):
    print('Start training.')
    """
    - use the function 'sklearn.DecisionTreeClassifier' to train and fit
    a classifier.    
    """    
    # ===================== PLEASE WRITE HERE =====================
    
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    clf.fit(x_train, y_train)
    
    # ===================== PLEASE WRITE HERE =====================    
        
    return clf

# Train SVM classifier on x_train and y_train
def train_SVM(x_train, y_train, C, Gamma):
    print('Start training.')
    """
    - use the function 'sklearn.svm.SVC' to train and fit a classifier
    with "rbf" kernel.    
    """    
    # ===================== PLEASE WRITE HERE =====================
    
    clf = SVC(C=C, kernel='rbf', gamma=Gamma)
    clf.fit(x_train, y_train)
    
    # ===================== PLEASE WRITE HERE =====================    
        
    return clf

# Use the trained classifier to test on x_test
def test(clf, x_test):
    print('Start testing...')
    """
    - use the trained classifier to predict the classes on x_test
    
    """
    # ===================== PLEASE WRITE HERE =====================   
    
    y_pred = clf.predict(x_test)

    # ===================== PLEASE WRITE HERE =====================
    
    return y_pred

def plot_tree(clf, feature_names, labels):
    print('tree diagram')
    """
    - the output of decision tree is intuitive to understand and can be easily 
    visualised
    - you can use sklearn.tree.plot_tree to plot the tree diagram
    """
    # ===================== PLEASE WRITE HERE =====================
    
    tree.plot_tree(clf, feature_names=feature_names, class_names=labels)
        
    # ===================== PLEASE WRITE HERE =====================
    
    

def plot_decision_boundary(clf, X, y):
    """
    -plot the decision boundary surface with different Gamma and regularization
    variable(C)
    """
    # ===================== PLEASE WRITE HERE =====================
    
    fig, ax = plt.subplots()
    # title for the plots
    title = ('Decision Boundary when C=500')
    # Set-up grid for plotting.
    X0, X1 = X[:, 0], X[:, 1]

    x_min, x_max = X0.min() - 1, X0.max() + 1
    y_min, y_max = X1.min() - 1, X1.max() + 1
    h = (x_max - x_min)/1000
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel('sepal width')
    ax.set_xlabel('sepal legnth')
    plt.xlim(xx.min(), xx.max())
    ax.set_title(title)
    
    # ===================== PLEASE WRITE HERE =====================


# Main
if __name__=='__main__':
    # Some parameters
    path = 'iris.data'
    testset_portion = 0.2
    
    # Load data
    data, x, y = load_data(path)
    feature_names=["sepal_length","sepal_width","petal_length","petal_width"]
    labels = np.unique(np.array(y))
    
    # Encode the labels from string to integer
    lb = preprocessing.LabelEncoder()
    lb.fit(labels)
    y=lb.transform(y)
    
    # Preprocessing
    x_train, x_test, y_train, y_test = split_dataset(x, y, testset_portion)
    
    ############################################################
    ######## decision tree Classification model ################
    ## Estimate performance on unseen data
    # Set hyperparameter
    Depth=3
    
    # Train and test
    print("Training and Testing for Decision Tree:")
    clf_DT = train_DT(x_train, y_train, Depth)
    y_pred_DT = test(clf_DT, x_test)

    # get Accuracy
    acc_DT = accuracy_score(y_test, y_pred_DT)
    print('\nAccuracy Decision Tree:', round(acc_DT, 3))
    
    #get the confusion matrix
    confusion_mat_DT = confusion_matrix(y_test, y_pred_DT)
    print('\nConfusion Matrix Decision Tree:', confusion_mat_DT)
    
    ## Analysis on training behaviour
    # plot tree diagram 
    """
    to understand how the algorithm has behaved, we have to visualize the 
    splits of the decision tree
    try with different value of depth parameter
    """
    plot_tree(clf_DT, feature_names, labels)
    
    
    ############################################################
    ######## SVM Classification model ##########################
    ## Estimate performance on unseen data
    
    # Set hyperparameter
    C=500
    Gamma=1
    
    # train and test
    print("Training and Testing for SVM:")
    clf_SVM = train_SVM(x_train, y_train, C, Gamma)
    y_pred_SVM = test(clf_SVM, x_test)

    # get Accuracy
    acc_SVM = accuracy_score(y_test, y_pred_SVM)
    print('\nAccuracy SVM with gamma:{} and C:{} is'.format(Gamma, C), round(acc_SVM, 3))
    
    #get the Confusion matrix
    confusion_mat_SVM = confusion_matrix(y_test, y_pred_SVM)
    print('\nConfusion Matrix SVM with gamma:{} and C:{} is'.format(Gamma, C), confusion_mat_SVM)
     
    ## Analysis 
    # plot Decision boundary surface 
    """
    to visualize the boundaries created by 
    In decision surface plot, you can consider only 2 feature at a time
    you can consider first two feature("sepal legnth" and "sepal width") and 
    train the model
    do the analysis by changing the Gamma and C parameters of the svm.svc 
    classifier
    """
    x_train_DB=np.array(x_train)[:,:2]
    x_test_DB=np.array(x_test)[:,:2]

    clf_SVM_DB = train_SVM(x_train_DB, y_train, C, Gamma)
    plot_decision_boundary(clf_SVM_DB, x_test_DB, y_test)