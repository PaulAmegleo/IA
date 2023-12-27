import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from sklearn import tree 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from joblib import dump

def importdata():
    df = pd.read_csv("Data.csv")
    

    
    return df 
    
# Function to split the dataset 
def splitdataset(df): 

    # Separating the target variable 
    X = df.drop('Movement', axis=1)
    Y = df['Movement']

    # Splitting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.3, random_state = 100) 
    
    return X, Y, X_train, X_test, y_train, y_test 
    
# Function to perform training with giniIndex. 
def train_using_gini(X_train, X_test, y_train): 

    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=5, min_samples_leaf=15) 

    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 
    
# Function to perform training with entropy. 
def train_with_entropy(X_train, X_test, y_train): 

    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 5, min_samples_leaf = 15) 

    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 


# Function to make predictions 
def prediction(X_test, clf_object): 

    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
    
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
    
    print("Confusion Matrix: \n", 
        confusion_matrix(y_test, y_pred)) 
    
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
    
    print("Report : ", 
    classification_report(y_test, y_pred)) 

# Driver code 
def main(): 
    
    # Building Phase 
    data = importdata() 
   
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 
    clf_gini = train_using_gini(X_train, X_test, y_train) 
    clf_entropy = train_with_entropy(X_train, X_test, y_train) 
    class_names = ['Up', 'Down', 'Left', 'Right']
    tree.plot_tree(clf_gini, class_names=class_names)
    plt.show()
    # Operational Phase 
    print("Results Using Gini Index:") 
    
    # Prediction using gini 
    y_pred_gini = prediction(X_test, clf_gini) 
    cal_accuracy(y_test, y_pred_gini) 
    
    print("Results Using Entropy:") 
    # Prediction using entropy 
    y_pred_entropy = prediction(X_test, clf_entropy) 
    cal_accuracy(y_test, y_pred_entropy) 
    dump(clf_entropy, 'modelBinary.joblib')
    
    
# Calling main function 
if __name__=="__main__": 
    main() 