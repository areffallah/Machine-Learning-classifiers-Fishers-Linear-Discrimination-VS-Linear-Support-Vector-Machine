import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.utils import shuffle
import datetime

# Plot Confusion Matrix Function
# https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Code for classifying Normal and Imbalanced Induction Motor dataset
# Constants
No_Attr = 8
Total_No_data = 1000
No_of_Train_Data = 600
No_of_attributes = 4

# Load Data for Normal and Imbalanced Induction Motor from csv file
# Finding working directory path
dir_path = os.getcwd()
pd_data_balanced = pd.read_csv(
    dir_path+'//Data//500 Balanced.csv', header=None, low_memory=False)
pd_data_imbalanced = pd.read_csv(
    dir_path+'//Data//500 ImBalanced.csv', header=None, low_memory=False)

# Merging Normal and Imbalanced data into a panda dataframe
pd_data_all = pd.concat([pd_data_balanced , pd_data_imbalanced], axis=0)

# Converting panda dataframe to numpy array
np_data_all = pd_data_all.to_numpy()

# Defining input X and output Y
# Our panda dataframe has 9 columns including 8 attributes and one column for class identification (0: Normal, 1:Imbalanced)
X = np_data_all[0:Total_No_data, 0:No_of_attributes]
Y = np_data_all[0:Total_No_data, 8]

# SVM using SkLearn
# Shuffling index
indexes = shuffle(np.array(range(Total_No_data)))
# Dividing data into test and train data
train_indexes = indexes[0:No_of_Train_Data]
test_indexes = indexes[No_of_Train_Data:Total_No_data]
X_train = X[train_indexes,:]
X_test = X[test_indexes,:]
y_train = Y[train_indexes]
y_test = Y[test_indexes]

# Generating Model
# Create a svm Classifier
clf = svm.SVC(kernel='linear')  # Linear Kernel
#Train the model using the training sets
training_start = datetime.datetime.now()  # Timing the process
clf.fit(X_train, y_train)
training_stop = datetime.datetime.now()  # Timing the process
training_time = training_stop - training_start
#Predict the response for test dataset
test_start = datetime.datetime.now()  # Timing the process
y_pred = clf.predict(X_test)
test_stop = datetime.datetime.now()  # Timing the process
testing_time = test_stop - test_start

# Evaluating the Model
# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Computational time for Training (SVM) = ",
      training_time.microseconds, " microseconds")
print("Computational time for Teasting (SVM) = ",
      testing_time.microseconds, " microseconds")

# Plotting confusion Matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
class_names = ["Normal", "Imbalanced"]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.show()
print("done")
