#!/usr/bin/env bash

! pip install gym-retro
! pip install imutils
! pip install opencv-python


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import retro

## Este es mi primer algoritmo de aprendizaje supervisado
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split


! python -m retro.import roms

def convert_array_to_decimal(keys):
  ints = keys
  string_ints = [str(int) for int in ints]
  string = "".join(string_ints)
  return int(string, 2)

def convet_decimal_to_array(number):
  string_ints = str(np.base_repr(number)).rjust(12,'0')
  return list(map(int, string_ints))

#Carga de los bk's
movie = retro.Movie(
   'records/SonicTheHedgehog-Genesis-GreenHillZone.Act1-0000000.bk2'
    )
movie.step()

try:
  env = retro.make(
    game= movie.get_game(),
    state=None,
    use_restricted_actions=retro.Actions.ALL,
    players=movie.players,
  )
  env.initial_state = movie.get_state()
  world = np.asarray(env.reset()).reshape(-1)
  target = np.array([0])
  while movie.step():
    img = np.asarray(env.render(mode='rgb_array')).reshape(-1)
    world = np.vstack((world,img))
    keys = []
    for p in range(movie.players):
      for i in range(env.num_buttons):
        keys.append(movie.get_key(i,p))
    keys = [int(elem) for elem in keys]
    number = convert_array_to_decimal(keys)
    target = np.append(target,number)
    ob, rew, done, info = env.step(keys)

except Exception as e:
  print(e)
  env.close()
  
target


clf = svm.SVC(gamma=0.001)
X_train, X_test, Y_train, Y_test = train_test_split(world, target, test_size=0.25, shuffle=True)

clf.fit(X_train, Y_train)


import pickle
filename = 'modelo_finalizado1.sav'
pickle.dump(clf, open(filename, 'wb'))

filename_XTest = 'XTest2.sav'
filename_XTrain = 'XTrain2.sav'
filename_YTest = 'YTest2.sav'
filename_YTrain = 'YTrain2.sav'
pickle.dump(X_test, open(filename_XTest, 'wb'))
pickle.dump(Y_test, open(filename_YTest, 'wb'))
pickle.dump(X_train, open(filename_XTrain, 'wb'))
pickle.dump(Y_train, open(filename_YTrain, 'wb'))



! pip install gym-retro
! pip install imutils
! pip install opencv-python
! python -m retro.import roms
 
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import retro
 
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
 
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


loaded_model = pickle.load(open(filename, 'rb'))
 
disp = metrics.plot_confusion_matrix(loaded_model, xtest, ytest)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
 
plt.show()


import retro

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('seaborn-white')
 
# Procesamento de imagenes
from skimage.transform import resize
from skimage.color import rgb2gray
from IPython import display

def convert_from_decimal_to_array(number):
  string_ints = str(np.base_repr(number)).rjust(12, '0')
  return list(map(int, string_ints))

done = False
try:
  env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')
  observation = env.reset()
  img = plt.imshow(env.render(mode='rgb_array'))
  while not done:
      world = env.render(mode='rgb_array')
      world_data = np.asarray(world).reshape(-1)
      img.set_data(world)
      display.display(plt.gcf())
      display.clear_output(wait=True)
      predicted = loaded_model.predict([world_data])
      action = convert_from_decimal_to_array(predicted[0])
      ob, rew, done, info = env.step(action)
      #print("Action ", action, "Reward ", rew)
except Exception as e:
  print(e)
  env.close()
  
from sklearn.linear_model import LinearRegression
reg = LinearRegression(normalize=True)

reg.fit(xtrain, ytrain)
prediction_linear = reg.predict(xtest) 


disp = metrics.plot_confusion_matrix(prediction_linear, xtest, ytest)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
 
plt.show()

! pip install scikit-image

import retro

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('seaborn-white')
 
# Procesamento de imagenes
from skimage.transform import resize
from skimage.color import rgb2gray
from IPython import display

def convert_from_decimal_to_array(number):
  string_ints = str(np.base_repr(abs(int(number)))).rjust(12, '0')
  return list(map(int, string_ints))

done = False
try:
  env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')
  observation = env.reset()
  img = plt.imshow(env.render(mode='rgb_array'))
  while not done:
      world = env.render(mode='rgb_array')
      world_data = np.asarray(world).reshape(-1)
      img.set_data(world)
      display.display(plt.gcf())
      display.clear_output(wait=True)
      predicted = reg.predict([world_data])
      action = convert_from_decimal_to_array(predicted[0])
      print(action)
      ob, rew, done, info = env.step(action)
      #print("Action ", action, "Reward ", rew)
except Exception as e:
  print(e)
  env.close()


from sklearn.linear_model import LogisticRegression
logi = LogisticRegression(random_state=0)
logi.fit(xtrain, ytrain)


! pip install scikit-image

import retro

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('seaborn-white')
 
# Procesamento de imagenes
from skimage.transform import resize
from skimage.color import rgb2gray
from IPython import display

def convert_from_decimal_to_array(number):
  string_ints = str(np.base_repr(abs(int(number)))).rjust(12, '0')
  return list(map(int, string_ints))

done = False
try:
  env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')
  observation = env.reset()
  img = plt.imshow(env.render(mode='rgb_array'))
  while not done:
      world = env.render(mode='rgb_array')
      world_data = np.asarray(world).reshape(-1)
      img.set_data(world)
      display.display(plt.gcf())
      display.clear_output(wait=True)
      predicted = logi.predict([world_data])
      action = convert_from_decimal_to_array(predicted[0])
      print(action)
      ob, rew, done, info = env.step(action)
      #print("Action ", action, "Reward ", rew)
except Exception as e:
  print(e)
  env.close()
  
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB


gnb = GaussianNB()
bern = BernoulliNB()

gnb.fit(xtrain, ytrain)
bern.fit(xtrain, ytrain)

predicted_bern= bern.predict(xtest)
predicted_gnb = gnb.predict(xtest)


disp = metrics.plot_confusion_matrix(bern, xtest, ytest)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()


! pip install scikit-image

import retro

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('seaborn-white')
 
# Procesamento de imagenes
from skimage.transform import resize
from skimage.color import rgb2gray
from IPython import display

def convert_from_decimal_to_array(number):
  string_ints = str(np.base_repr(abs(int(number)))).rjust(12, '0')
  return list(map(int, string_ints))

done = False
try:
  env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')
  observation = env.reset()
  img = plt.imshow(env.render(mode='rgb_array'))
  while not done:
      world = env.render(mode='rgb_array')
      world_data = np.asarray(world).reshape(-1)
      img.set_data(world)
      display.display(plt.gcf())
      display.clear_output(wait=True)
      predicted = bern.predict([world_data])
      action = convert_from_decimal_to_array(predicted[0])
      print(action)
      ob, rew, done, info = env.step(action)
      #print("Action ", action, "Reward ", rew)
except Exception as e:
  print(e)
  env.close()


from sklearn import tree
arbol = tree.DecisionTreeClassifier()
arbol.fit(xtrain, ytrain)

plt.figure(figsize=(12,12))
tree.plot_tree(arbol)
plt.savefig('tree_high_dpi', dpi=200) 

predicted_arbol = arbol.predict(xtest)

disp = metrics.plot_confusion_matrix(arbol, xtest, ytest)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()


! pip install scikit-image

import retro

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('seaborn-white')
 
# Procesamento de imagenes
from skimage.transform import resize
from skimage.color import rgb2gray
from IPython import display

def convert_from_decimal_to_array(number):
  string_ints = str(np.base_repr(abs(int(number)))).rjust(12, '0')
  return list(map(int, string_ints))

done = False
try:
  env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')
  observation = env.reset()
  img = plt.imshow(env.render(mode='rgb_array'))
  while not done:
      world = env.render(mode='rgb_array')
      world_data = np.asarray(world).reshape(-1)
      img.set_data(world)
      display.display(plt.gcf())
      display.clear_output(wait=True)
      predicted = arbol.predict([world_data])
      action = convert_from_decimal_to_array(predicted[0])
      print(action)
      ob, rew, done, info = env.step(action)
      #print("Action ", action, "Reward ", rew)
except Exception as e:
  print(e)
  env.close()
  

  
