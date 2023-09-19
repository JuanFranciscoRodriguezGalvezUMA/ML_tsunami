######################################################################################################################
####################################### RUNNING EXAMPLES #############################################################
######################################################################################################################


# To run one example, we need:

import tensorflow as tf
import numpy as np
import joblib


# And now to load the scalers

scaler_filename = r"Directory\scaler.save"
scaler2_filename = r"Directory\scaler2.save"
scaler = joblib.load(scaler_filename)
scaler2 = joblib.load(scaler2_filename)

scaler.clip = False #Due to versions incompatibility (https://stackoverflow.com/questions/65635575/attributeerror-minmaxscaler-object-has-no-attribute-clip)
scaler2.clip = False


# We load the models we are going to use. We can add more if we want. This list and the weights list below can be save and load.

model_1 = tf.keras.models.load_model(r"Directory\model_13.h5")
model_2 = tf.keras.models.load_model(r"Directory\model_16.h5")
model_3 = tf.keras.models.load_model(r"Directory\model_19.h5")
model_4 = tf.keras.models.load_model(r"Directory\model_23.h5")
models = list()


models.append(model_1)
models.append(model_2)
models.append(model_3)
models.append(model_4)




#Evaluate ensemble
def ensemble_predictions(members, weights, datos):
    yhats = [model.predict(datos) for model in members]
    n = len(models)
    results = weights[0]*yhats[0]
    for i in range(1,n):
        results += weights[i]*yhats[i]
    return results


weights = [0.28412398, 0.35871758, 0.29530805, 0.06185039]  #These are the best weights we found.







#JUST FOR ONE EXAMPLE

# Once this is done, we apply the normalization to the input values (Okada parameters), we use ann.predict() to predict the result, and we
# reconvert the result by using scaler2.inverse_transform.


ejemplo = scaler.transform([[-9.719567, 35.887791, 10.789185, 122.200684, 40.049072, 47.265747, 38.707275, 84.171143, 4.780341]])
print(scaler2.inverse_transform(ensemble_predictions(models,weights,ejemplo)))


# This is only to compare. It is datay[0] because ejemplo is the first row in "DatosMH_X". You can try others. Load datay.
datay = np.loadtxt(r"Directory\DatosMH_Y.txt")
print(datay[0])





# FOR ALL THE SETS

######################################################################################################################
############################################### ERRORS ###############################################################
######################################################################################################################

# To check errors in the sets, we need to load them

dataX = np.loadtxt(r"Directory\DatosMH_X.txt")
datay = np.loadtxt(r"Directory\DatosMH_Y.txt")


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataX, datay, test_size = 0.25, random_state = 0)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = 0)

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

y_train = scaler2.fit_transform(y_train)
y_val = scaler2.transform(y_val)
y_test = scaler2.transform(y_test)



############################################### VALIDATION ###############################################################

# This show us the mean absolute error and the maximum absolute error made in the validation set.


y_pred_val = ensemble_predictions(models, weights, X_val)
dif_val = abs(scaler2.inverse_transform(y_pred_val) - scaler2.inverse_transform(y_val))
dif_val = [dif_val[i][j] for i in range(len(dif_val)) for j in range(6)]
print(np.mean(dif_val))
print(max(dif_val))




#Validation at each point selected (we selected 6 points)

mean_dif_val = []
max_dif_val = []
y_pred_val = ensemble_predictions(models, weights, X_val)
for j in range(6):
    dif_val = abs(scaler2.inverse_transform(y_pred_val) - scaler2.inverse_transform(y_val))
    dif_val = [dif_val[i][j] for i in range(len(dif_val))]
    mean_dif_val.append(np.mean(dif_val))
    max_dif_val.append(max(dif_val))
print(mean_dif_val)
print(max_dif_val)




############################################### TEST ###############################################################

# This show us the mean absolute error and the maximum absolute error made in the test set.


y_pred_test = ensemble_predictions(models, weights, X_test)
dif_test = abs(scaler2.inverse_transform(y_pred_test) - scaler2.inverse_transform(y_test))
dif_test = [dif_test[i][j] for i in range(len(dif_test)) for j in range(6)]
print(np.mean(dif_test))
print(max(dif_test)) 


#Test at each point


mean_dif_test = []
max_dif_test = []
y_pred_test = ensemble_predictions(models, weights, X_test)
for j in range(6):
    dif_test = abs(scaler2.inverse_transform(y_pred_test) - scaler2.inverse_transform(y_test))
    dif_test = [dif_test[i][j] for i in range(len(dif_test))]
    mean_dif_test.append(np.mean(dif_test))
    max_dif_test.append(max(dif_test))
print(mean_dif_test)
print(max_dif_test)




############################################### TRAINING ###############################################################

# This show us the mean absolute error and the maximum absolute error made in the training set.


y_pred_training = ensemble_predictions(models, weights, X_train)
dif_training = abs(scaler2.inverse_transform(y_pred_training) - scaler2.inverse_transform(y_train))
dif_training = [dif_training[i][j] for i in range(len(dif_training)) for j in range(6)]
print(np.mean(dif_training))
print(max(dif_training))



#Training at each point

mean_dif_train = []
max_dif_train = []
y_pred_train = ensemble_predictions(models, weights, X_train)
for j in range(6):
    dif_train = abs(scaler2.inverse_transform(y_pred_train) - scaler2.inverse_transform(y_train))
    dif_train = [dif_train[i][j] for i in range(len(dif_train))]
    mean_dif_train.append(np.mean(dif_train))
    max_dif_train.append(max(dif_train))
print(mean_dif_train)
print(max_dif_train)
