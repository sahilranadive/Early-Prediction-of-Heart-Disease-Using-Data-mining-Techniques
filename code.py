import numpy as np
from sklearn.preprocessing import StandardScaler,normalize
import matplotlib.pyplot as plt
from sklearn import svm
%matplotlib inline
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler
from collections import namedtuple, Counter, defaultdict
from math import log,ceil,sqrt

from sklearn import metrics
from math import exp,log10
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import csv

def plot_confusion_matrix(Y_actual,Y_predicted):
    conf_matrix =  metrics.confusion_matrix(Y_actual,Y_predicted)
    sns.heatmap(conf_matrix, annot=True,  fmt='')

####function to normalize the data##

def norm(Z):
    mean = np.sum(Z,axis=0)/len(Z)
    var = []
    for i in range(Z.shape[1]):
        sum1 = 0
        for j in range(Z.shape[0]):
            sum1 += (Z[j,i] - mean[i])**2
        var.append(1. *sum1/len(Z))

    for i in range(Z.shape[1]):
        for j in range(Z.shape[0]):
            Z[j,i] = (Z[j,i] - mean[i])/sqrt(var[i])
    
    
    
            
    return Z


def accuracy(Y,Y_hat):
                
    sum1 = sum([1 for i in range(len(Y)) if Y[i] == Y_hat[i]])
    return (1.0*sum1/len(Y))*100


data_txt =open("processed.cleveland.data1.txt","r") 
data = data_txt.read()
data = data.split('\n')[:-1]


X = []
Y = []
for i in data:
    
    x_1 = i.split(',')
    temp=map(float,x_1)
    x=tuple(temp)
   
    
    Y.append(float(x[len(x)-1]))
    x = x[0:len(x)-1]   
    
        
    #print type(i)
    X.append(x)
    
X = np.array(X)

Y = np.reshape(np.array(Y),(len(Y),))
Y[Y > 0] = 1 ###changing the lables to 0 and 1

for i in range(X.shape[1]): #####imputing the missing values with mode of the feature
    m =  mode(X[:,i])[0][0]
    for j in range(X.shape[0]):
        if X[j][i] == -9:
            X[j][i] = m


X_norm = norm(X.copy())##normalizing the data

cov_matrix = np.dot(np.transpose(X_norm),X_norm)
eigen= np.linalg.eig(cov_matrix)
eigen_val = eigen[0]
#print eigen_val
eigen_vec = eigen[1]
#print( type(eigen_vec))
#print (eigen_vec)
#print (eigen_vec[0])
total_eigen_val = np.sum(eigen_val)
sum_d = []
for i in range(len(eigen_val)):
    sum_d.append(1. *np.sum(eigen_val[i+1:])/total_eigen_val)
width = 0.35
plt.bar(range(len(eigen_val)),sum_d,width)
plt.show()



W = eigen_vec[:,:]

Z = []
for i in X_norm:
    Z.append(np.dot(np.transpose(W),i))
Z = np.array(Z)

#print ("X's shape: ", X.shape)
#print ("Z's shape: ", Z.shape)
#print ("Y's shape: ", Y.shape)
#print(Z)

colnames = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs','restecg', 'thalach', 'exang','oldpeak','slope','ca','thal']
df = pd.DataFrame(data=X,columns=colnames)
df.describe()

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

#shuffle
Z,Y= shuffle(Z,Y, random_state=0)

# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(Z[:200], Y[:200])

# make predictions
Y_expect_Cart = Y[170:]
Y_predict_Cart= model.predict(Z[170:])
# summarize the fit of the model
print ("Accuracy of the CART is: %s" %(accuracy(Y[170:],Y_predict_Cart)))
print()
print(metrics.classification_report(Y_expect_Cart,Y_predict_Cart))


print ("Confusion Matrix")
print ("================")
plot_confusion_matrix(Y[170:],Y_predict_Cart) 


from id3 import Id3Estimator
from id3 import export_graphviz

#shuffle
Z,Y= shuffle(Z,Y, random_state=0)

# fit a ID3 model to the data
estimator = Id3Estimator()
estimator.fit(Z[:200], Y[:200])
export_graphviz(estimator.tree_, 'tree.dot')

# make predictions
Y_expect_ID3 = Y[170:]
Y_predict_ID3=estimator.predict(Z[170:])

# summarize the fit of the model

print ("Accuracy of the ID3 is: %s" %(accuracy(Y[170:],Y_predict_ID3)))
print()
print(metrics.classification_report(Y[170:],Y_predict_Cart))
print ("Confusion Matrix")
print ("================")
plot_confusion_matrix(Y[170:],Y_predict_ID3)  


from sklearn.ensemble import RandomForestClassifier

#shuffle
Z,Y= shuffle(Z,Y, random_state=0)

# fit a Random Forest model to the data
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
rf.fit(Z[:200], Y[:200])

# make predictions
Y_expect_RF = Y[170:]
Y_predict_RF = rf.predict(Z[170:])


# summarize the fit of the model
print ("Accuracy of the Random Forest Classifier is: %s" %(accuracy(Y[170:],Y_predict_RF)))
print()
print(metrics.classification_report(Y[170:],Y_predict_RF))
print ("Confusion Matrix")
print ("================")
plot_confusion_matrix(Y[170:],Y_predict_RF)  

kappa_score_CART=cohen_kappa_score(Y_expect_Cart,Y_predict_Cart)
kappa_score_ID3=cohen_kappa_score(Y_expect_ID3,Y_predict_ID3)
kappa_score_RF=cohen_kappa_score(Y_expect_RF,Y_predict_RF)
print("kappa score for CART: %s" %(kappa_score_CART))
print("kappa score for ID3: %s" %(kappa_score_ID3))
print("kappa score for Random Forest: %s" %(kappa_score_RF))

mean_absolute_error_CART=metrics.mean_absolute_error(Y_expect_Cart, Y_predict_Cart)
mean_absolute_error_ID3=metrics.mean_absolute_error(Y_expect_ID3, Y_predict_ID3)
mean_absolute_error_RF=metrics.mean_absolute_error(Y_expect_RF, Y_predict_RF)
print(mean_absolute_error_CART)
print(mean_absolute_error_ID3)
print(mean_absolute_error_RF)

root_mean_squared_error_CART=sqrt(metrics.mean_squared_error(Y_expect_Cart, Y_predict_Cart))
root_mean_squared_error_ID3=sqrt(metrics.mean_squared_error(Y_expect_ID3, Y_predict_ID3))
root_mean_squared_error_RF=sqrt(metrics.mean_squared_error(Y_expect_RF, Y_predict_RF))
print(root_mean_squared_error_CART)
print(root_mean_squared_error_ID3)
print(root_mean_squared_error_RF)