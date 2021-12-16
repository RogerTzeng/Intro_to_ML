"""
===================== PLEASE WRITE HERE =====================
- The title of this script.
- A brief explanation of this script, e.g. the purpose of this script, what
can this script achieve or solve, what algorithms are used in this script...

- Author:曾靖桐 Date:2021.12.08
===================== PLEASE WRITE HERE =====================
"""

# Import libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm


# Load data file and return two numpy arrays, i.e. x, y
# x: (n_samples, 24)
# y: (n_samples, )
def load_data(path):
    # ===================== PLEASE WRITE HERE =====================

    print('Load data.')
    data = np.genfromtxt(path, delimiter=',')
    x = data[1:, 0:-1]
    y = data[1:, -1].astype(int)

    # ===================== PLEASE WRITE HERE =====================

    return x, y

def preprocessing(x, y, random_state=1):
    # Split dataset
    # ===================== PLEASE WRITE HERE =====================

    train_ratio = 0.8
    validation_ratio = 0.1
    test_ratio = 0.1
    x_train0, x_test0, y_train, y_test = train_test_split(x, y, test_size=1-train_ratio, random_state = random_state)
    x_val0, x_test0, y_val, y_test = train_test_split(x_test0, y_test, test_size=test_ratio/(test_ratio + validation_ratio))
    
    # ===================== PLEASE WRITE HERE =====================


    # Feature scaling
    # ===================== PLEASE WRITE HERE =====================

    scaler = StandardScaler().fit(x_train0)
    x_train = scaler.transform(x_train0)
    x_val = scaler.transform(x_val0)
    x_test = scaler.transform(x_test0)

    # ===================== PLEASE WRITE HERE =====================

    return x_train, x_val, x_test, y_train, y_val, y_test

def train(x_train, x_val, y_train, y_val, layers, n_epochs):
    print('Hidden layers:', layers)
    print('# epochs:', n_epochs)

    # Initialize arrays, which will be used to store loss and acc at each epoch
    train_loss = np.zeros(n_epochs)
    train_acc = np.zeros(n_epochs)
    val_loss = np.zeros(n_epochs)
    val_acc = np.zeros(n_epochs)

    # Define MLP classifier
    clf = MLPClassifier(hidden_layer_sizes=layers, random_state=2,
                        max_iter=1, warm_start=True)

    for i in tqdm(range(n_epochs)):
        # Training
        # ===================== PLEASE WRITE HERE =====================

        clf.fit(x_train, y_train)

        # ===================== PLEASE WRITE HERE =====================


        # Evaluation on the training set
        # ===================== PLEASE WRITE HERE =====================

        

        # ===================== PLEASE WRITE HERE =====================


        # Evaluation on the validation set
        # ===================== PLEASE WRITE HERE =====================

        

        # ===================== PLEASE WRITE HERE =====================

    return clf, train_loss, train_acc, val_loss, val_acc

# def evaluation(clf, x, y):
#     # Get predictions
#     # ===================== PLEASE WRITE HERE =====================

    

#     # ===================== PLEASE WRITE HERE =====================


#     # Scores
#     # ===================== PLEASE WRITE HERE =====================
#     # Recall of each class
#     recalls =

#     # Accuracy
#     acc =

#     # Unweighted Average Recall (UAR)
#     uar =

#     # Confusion matrix
#     cf_matrix =

#     # ===================== PLEASE WRITE HERE =====================

#     return recalls, acc, uar, cf_matrix

def plot_curve(train, val, title, legend_loc):
    plt.plot(train, color='c', label='Train')
    plt.plot(val, color='m', label='Val')
    plt.title(title, fontsize=14)
    plt.xlabel('Epochs')
    plt.legend(loc=legend_loc)
    plt.grid()
    plt.show()

def plot_heatmap(cf_matrix):
    ax = sns.heatmap(cf_matrix, annot=True, fmt="d")
    ax.set_title('Confusion Matrix on Validation set')
    ax.set_ylabel('Ground Truth')
    ax.set_xlabel('Prediction')
    plt.show()


# Main
if __name__=='__main__':
    # Load data
    x, y = load_data('data.csv')

    # Preprocessing
    x_train, x_val, x_test, y_train, y_val, y_test = preprocessing(x, y)

    # Training
    hidden_layer_sizes = (16, 16, 16)
    n_epochs = 100
    clf, train_loss, train_acc, val_loss, val_acc = train(
        x_train, x_val, y_train, y_val, hidden_layer_sizes, n_epochs)

    # Loss and accuracy curve
    plot_curve(train_loss, val_loss, 'Log Loss', 'upper right')
    plot_curve(train_acc, val_acc, 'Accuracy', 'upper left')

    # Evaluation on validation set
    print('======== Validation Set ========')
    # Loss
    print('Loss:', round(val_loss[-1], 3))
    # # Recall, accuracy, UAR
    # recalls, acc, uar, cf_matrix = evaluation(clf, x_val, y_val)
    # print('Recalls:', np.round(recalls, 3))
    # print('Accuracy', round(acc, 3))
    # print('UAR:', round(uar, 3))
    # # Confusion matrix
    # plot_heatmap(cf_matrix)

    # Evaluation on testing set
    """
    print('======== Testing Set ========')
    recalls, acc, uar, cf_matrix = evaluation(clf, x_test, y_test)
    print('Recalls:', np.round(recalls, 3))
    print('Accuracy', round(acc, 3))
    print('UAR:', round(uar, 3))
    plot_heatmap(cf_matrix)
    """
