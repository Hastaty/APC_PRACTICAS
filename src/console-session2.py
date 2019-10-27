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
import sklearn.preprocessing as prepro
from IPython.display import display



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
plt.show()

#%%

#Se calcula una matriz de correlación de todas las columnas con todas las columnas
correlacio = dataset_frame.corr()
plt.figure(figsize=(23, 23)) # Hago la figura más grande para que se displaye
                             # correctamente
display(ax = sb.heatmap(correlacio, annot=True, linewidths=.5))
plt.show()
#%%
# Como no podemos binarizar sacamos los 
# siguientes atributos. 
# Fechas y el instante
dataset_frame=dataset_frame.\
                drop('instant', 1).\
                drop('dteday', 1).\
                drop('yr', 1).\
                drop('mnth', 1).\
                drop('hr', 1)
for i, e in enumerate(dataset_frame):
    print(str(i) + ':' + ' ' + str(e))
    
# Binarizamos atributos categoricos
# 4 - 1 - 7 - 1 - 4 - (1 -)* 
array1 = dataset_frame.values
dataset_array=np.empty((24, dataset_frame.shape[0]), dtype=np.float64)

one_hot_encoder = prepro.OneHotEncoder(sparse=False, categories='auto')
dataset_array[:, 0:3] = one_hot_encoder.fit_transform(array1[:, 0]).reshape(-1, 1)

dataset_array[:, 4] = dataset_frame['holiday'].values.reshape(-1, 1)
dataset_array[:, 5] = dataset_frame

