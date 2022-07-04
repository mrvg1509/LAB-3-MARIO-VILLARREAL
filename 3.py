
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import retro
 
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import pickle
filename = 'modelo_finalizado1.sav'
filename_XTEST = 'XTest2.sav'
filename_YTEST = 'YTest2.sav'
filename_XTRAIN = 'XTrain2.sav'
filename_YTRAIN = 'YTrain2.sav'
loaded_model = pickle.load(open(filename, 'rb'))
xtest = pickle.load(open(filename_XTEST, 'rb'))
ytest = pickle.load(open(filename_YTEST, 'rb'))
xtrain = pickle.load(open(filename_XTRAIN, 'rb'))
ytrain = pickle.load(open(filename_YTRAIN, 'rb'))

reg = LinearRegression(normalize=True)
reg.fit(xtrain, ytrain)
prediction_linear = reg.predict(xtest) 
plt.scatter(xtest, ytest, color="black")
plt.plot(xtest, ytrain, color="blue", linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()


disp = metrics.plot_confusion_matrix(prediction_linear, xtest, ytest)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()




logi = LogisticRegression(random_state=0)
logi.fit(xtrain, ytrain)
plt.tight_layout()
plt.show()


