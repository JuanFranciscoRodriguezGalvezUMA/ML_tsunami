# Final Individual Model for Arrival Time Problem

# This script is divided in three parts. The first one is to load the libraries, data, split and scale the data. It must be done.
# The second part is the model topology. You can run it if you want but maybe will not give the same results. Just you can see it.
# The third part is just for load the model instead of running it and check the errors. Also the matrix weights can be extracted from here
# just using ann.get_weights(). This will give you the next information:

# Basically, what you need (for what I know) is just the first part and the third part until you load the model and get the weights matrix.

'''
Quick explanation of the weight matrix is provided in the FinalIndividualModelMaximumHeight.py.
'''










'''
FIRST PART
'''


# Data is provided in the repository

####################################################################################################################
############################################## LIBRARIES  ##########################################################
####################################################################################################################

import numpy as np
import tensorflow as tf
from matplotlib import pyplot


####################################################################################################################
############################################## DATA PREPARATION  ###################################################
####################################################################################################################


#Data in the order we obtained the best results. You can shuffle it to get other results.

dataX = np.loadtxt(r"Directory\DatosAT_X.txt")
datay = np.loadtxt(r"Directory\DatosAT_Y.txt")



# We split the samples en 12.000 for training, 2.000 for validation and 2.000 for test. Random_state = 0 in order to always return the same split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataX, datay, test_size = 0.25, random_state = 0)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = 0)




import joblib

scaler_filename = r"Directory\scaler.save"
scaler2_filename = r"Directory\scaler2.save"
scaler = joblib.load(scaler_filename)
scaler2 = joblib.load(scaler2_filename)

scaler.clip = False #Due to versions incompatibility (https://stackoverflow.com/questions/65635575/attributeerror-minmaxscaler-object-has-no-attribute-clip)
scaler2.clip = False

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

y_train = scaler2.fit_transform(y_train)
y_val = scaler2.transform(y_val)
y_test = scaler2.transform(y_test)














'''
SECOND PART
'''

####################################################################################################################
############################################ MODEL 17 ##############################################################
####################################################################################################################


# MODEL 17. It takes 12975 epochs.

#Callbacks

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2

tf.random.set_seed(0)
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=100, input_shape=(9,), kernel_regularizer=l2(0.0001), activation='tanh'))
ann.add(tf.keras.layers.Dense(units=100, activation='tanh'))
ann.add(tf.keras.layers.Dense(units=100, activation='tanh'))
ann.add(tf.keras.layers.Dense(units=100, activation='tanh'))
ann.add(tf.keras.layers.Dense(units=100, activation='tanh'))
ann.add(tf.keras.layers.Dense(units=6,activity_regularizer = l2(0.00001)))
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
hub = tf.keras.losses.Huber(delta=0.5)
ann.compile(optimizer = opt, loss = hub)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=500)
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.75,patience=180, min_lr=0.0001)
history = ann.fit(X_train, y_train, batch_size = 256, epochs = 13000, validation_data= (X_val,y_val), shuffle = True, callbacks=[es,rlrop])


















'''
THIRD PART
'''

######################################################################################################################
############################################### ERRORS ###############################################################
######################################################################################################################


# As we mentioned, you dont need to run these models, a folder with the h5 file of these models are provided. You can just load them.
# If you have run the model, you dont need to load it.

ann = tf.keras.models.load_model(r"Directory\model_17.h5")

# If you want to see a brief summary of your model, run the line below.

#model.summary()

# To see the matrix weights you need to execute this lines

weights_matrix = ann.get_weights()

# This contains also the bias! 









# To check the errors in each set:


############################################### VALIDATION ###############################################################

# This show us the mean absolute error and the maximum absolute error made in the validation set.

y_pred_val = ann.predict(X_val)
dif_val = abs(scaler2.inverse_transform(y_pred_val) - scaler2.inverse_transform(y_val))
dif_val = [dif_val[i][j] for i in range(len(dif_val)) for j in range(6)]
print(np.mean(dif_val))
print(max(dif_val))


#Validation at each point selected (we selected 6 points)

mean_dif_val = []
max_dif_val = []
y_pred_val = ann.predict(X_val)
for j in range(6):
    dif_val = abs(scaler2.inverse_transform(y_pred_val) - scaler2.inverse_transform(y_val))
    dif_val = [dif_val[i][j] for i in range(len(dif_val))]
    mean_dif_val.append(np.mean(dif_val))
    max_dif_val.append(max(dif_val))
print(mean_dif_val)
print(max_dif_val)




############################################### TEST ###############################################################

# This show us the mean absolute error and the maximum absolute error made in the test set.


y_pred_test = ann.predict(X_test)
dif_test = abs(scaler2.inverse_transform(y_pred_test) - scaler2.inverse_transform(y_test))
dif_test = [dif_test[i][j] for i in range(len(dif_test)) for j in range(6)]
print(np.mean(dif_test))
print(max(dif_test)) 


#Test at each point

mean_dif_test = []
max_dif_test = []
y_pred_test = ann.predict(X_test)
for j in range(6):
    dif_test = abs(scaler2.inverse_transform(y_pred_test) - scaler2.inverse_transform(y_test))
    dif_test = [dif_test[i][j] for i in range(len(dif_test))]
    mean_dif_test.append(np.mean(dif_test))
    max_dif_test.append(max(dif_test))
print(mean_dif_test)
print(max_dif_test)






############################################### TRAINING ###############################################################

# This show us the mean absolute error and the maximum absolute error made in the training set.


y_pred_training = ann.predict(X_train)
dif_training = abs(scaler2.inverse_transform(y_pred_training) - scaler2.inverse_transform(y_train))
dif_training = [dif_training[i][j] for i in range(len(dif_training)) for j in range(6)]
print(np.mean(dif_training))
print(max(dif_training))



#Training at each point

mean_dif_train = []
max_dif_train = []
y_pred_train = ann.predict(X_train)
for j in range(6):
    dif_train = abs(scaler2.inverse_transform(y_pred_train) - scaler2.inverse_transform(y_train))
    dif_train = [dif_train[i][j] for i in range(len(dif_train))]
    mean_dif_train.append(np.mean(dif_train))
    max_dif_train.append(max(dif_train))
print(mean_dif_train)
print(max_dif_train)








# We can also plot the errors graphic (only if you have run the model, not loaded).

pyplot.subplot(111)
pyplot.title('Error', pad=-40)
pyplot.plot(history.history['loss'], label='Training') 
pyplot.plot(history.history['val_loss'], label='Validation') 
pyplot.legend()
pyplot.show()