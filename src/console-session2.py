# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 22:41:10 2019

@author: REROTA_LNV
"""


#Aquí se hacen los import de las bibliotecas necesarias
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Mas imports
from sklearn import linear_model as model
from sklearn import metrics 
from IPython.display import display
from sklearn.metrics import r2_score



#Creas un Pandas DataFrame para cargar el CSV con los datos
dataset_frame = pd.read_csv('../data/bike_' + \
                      'sharing_hourly.csv')


#%%
#veamos cuantas dimensiones y registros contiene
display(dataset_frame.shape)

for i, e in enumerate(dataset_frame):
    print(str(i) + ':' + ' ' + str(e))
    
#veamos esas filas
display(dataset_frame.head())

#Borras una fila
dataset_array = dataset_frame.drop('dteday', 1).values
#Pasas todos los valores a float
dataset_array = dataset_array.astype(np.float64)


#%%
# Miro que no haya valores Nan, que podrían dar 
# cabida a datos erroneos.
i, j = np.where(dataset_array == np.nan)
print(i, '-', j);

#%%
better_frame = dataset_frame

#Estos tres valores están entre 0 y 1, por lo que tiene sentido
#multiplicar por 100 para obtener los valores reales.
#Si no recuerdo mal, las bibliotecas de Python trabajaban con 
#valores normalizados, asi que igual esto no hacía falta
better_frame['temp'] = better_frame['temp'] * 41.0
better_frame['atemp'] = better_frame['atemp'] * 50.0
better_frame['hum'] = better_frame['hum'] * 100.0
better_frame['windspeed'] = better_frame['windspeed'] * 67.0

#descripcion datos
print('descripcion datos')
display(better_frame.describe())

#Se muestra un histograma con los datos, pero eliminando dos cosas
better_frame.drop(['instant','cnt'],1).hist()
plt.show()

#%%

#Se calcula una matriz de correlación de todas las columnas con todas las columnas
correlacio = dataset_frame.corr()
plt.figure(figsize=(23, 23)) # Hago la figura más grande para que se displaye
                             # correctamente
ax = sb.heatmap(correlacio, annot=True, linewidths=.5)

#%%
#Funcion axuliar para estandarizar valores
def standarize(M):
    mean = M.mean(axis=0)
    std = M.std(axis=0)
    M = M - mean[None, :]
    M = M / std[None, :]
    return M

dataset_array = standarize(dataset_array)

#%%
#Construyes un objeto que puede calcular la regresión
regression = model.LinearRegression()


#Esta función "trocea" los datos
def split_data(x, y, train_ratio): 
    indices = np.arange(x.shape[0]) 
    np.random.shuffle(indices) 
    n_train = int(np.floor(x.shape[0]*train_ratio)) 
    indices_train = indices[:n_train] 
    indices_val = indices[n_train:] 
    x_train = x[indices_train, :] 
    y_train = y[indices_train] 
    x_val = x[indices_val, :] 
    y_val = y[indices_val] 
    return x_train, y_train, x_val, y_val


#Recorremos todas las columnas
for i in range(14):
        #Creamos un objeto que pueda manejar la regresión
        regression = model.LinearRegression()
        # .reshape(-1, 1) transpose

        
        n = 60
        x = dataset_array[:, i].reshape(-1, 1)
        y = dataset_array[:, 15]
        x_train = x[:-n, :]
        x_test =  x[-n:, :]
        y_train = y[:-n]
        y_test = y[-n:]
        regression.fit(x_train, y_train)
        y_predicted = regression.predict(x_test)
        plt.figure()
        ax = plt.scatter(x_test, y_test)
        plt.plot(x_test, y_predicted, '-ro', linewidth=3)  
        
      

