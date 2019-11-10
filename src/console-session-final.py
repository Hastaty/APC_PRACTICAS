# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 22:41:10 2019

@author: REROTA_LNV
"""


#Aquí se hacen los import de las bibliotecas necesarias
import numpy as np
import pandas as pd
import seaborn as sns
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
dataset_array = dataset_frame.drop(['instant','dteday'], 1).values
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
#Se muestra un histograma con los datos, pero eliminando dos cosas
dataset_frame.drop(['instant','cnt'],1).hist()
#%%
#lista = ['season','yr','mnth','hr','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed','casual','registered']
#for i in range(14):  
 #   plt.show()
  #  fg = plt.figure(1, figsize=(9,6))
   # ac = fg.add_subplot(111)
    #bp = ac.boxplot(dataset_array[i])
for i in range(14):    
    plt.show()
    plt.figure(1, figsize=(9,6))
    sns.boxplot(dataset_array[i])

#%%

#Se calcula una matriz de correlación de todas las columnas con todas las columnas
correlacio = dataset_frame.corr()
plt.figure(figsize=(23, 23)) # Hago la figura más grande para que se displaye
                             # correctamente
display(ax = sns.heatmap(correlacio, annot=True, linewidths=.5))
plt.show()

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
def split_data(x, y, train_ratio = 0.8):
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
#%%

#Y nos quedamos solo con los valores, sin las cabeceras
data=dataset_array
data_global = dataset_frame.drop(['instant','dteday'], 1)

#Averiguamos cual es el número de columna más alto
columnas=data.shape[1]-1

#Y en la matriz x metemos los atributos
x = data[:, :columnas]
#Y en la matriz (mas bien vector) metemos la cantidad de bicis
#alquiladas, es decir el valor que se pretende predecir
y = data[:, columnas]

#Se "parte" la matriz y nos quedamos con unas cuantas filas
#que se usaran para entrenar al modelo (x_train, y_train)
#y con unos cuantos valores para usar como prueba (x_test, y_test)
x_train, y_train, x_test, y_test=split_data(x, y)

#print("Longitudes:")
#print("Len del x_train:"+str(len(x_train)))
#print("Len del y_train:"+str(len(y_train)))
#print("Len del x_test:"+str(len(x_test)))
#print("Len del y_test:"+str(len(y_test)))
#print("dimensiones x_test"+str(x_test.shape))

for columna in range(0, columnas):
    #Sacamos una de las columnas (temp, atemp, etc...)
    columna_x=x_train[:, columna]
    #Transponemos la columna y se vuelve una fila (LinearRegression lo necesita así)
    fila_x=columna_x.reshape(-1, 1)
    modelo=model.LinearRegression()
    #Entrenamos el modelo pasandole
    modelo=modelo.fit(fila_x, y_train)

    #En este punto el modelo está "entrenado"
    #print("La curva para el atributo "+str(columna)+" tiene estos coeficiente")
    #print(modelo.coef_)
    #Ahora pedimos al modelo entrenado que haga una prediccion con los valores de prueba
    #que se recortaron con la función split_data. De nuevo el modelo
    #necesitará tomar los datos de x "traspuestos"
    x_test_filas=x_test[:, columna].reshape(-1,1)
    #print("x_test_filas")
    #print(x_test_filas.shape)
    predicciones_y=modelo.predict(x_test_filas)
    #print("Predicciones para y")
    #print(predicciones_y)
    
    print(str(data_global.columns[columna]))
    plt.figure() 
    ax = plt.scatter(x_test_filas, y_test) 
    plt.plot(x_test_filas, predicciones_y, 'r', linewidth=3)
    plt.show()

    

    #print("Len y test:"+str(len(y_test)))
    #print("Len y pred:"+str(len(predicciones_y)))
    #print(len(y_test))
    error_cuadratico_medio= metrics.mean_squared_error(y_test, predicciones_y)
    print("El error cuadratico")
    print(str(data_global.columns[columna]) +" ----> "+str(error_cuadratico_medio))
    r2 = metrics.r2_score(y_test, predicciones_y)
    print("El error r2  ")
    print(str(data_global.columns[columna]) +" ------> "+str(r2))
    #Ahora calculamos el error cuadratico medio que sale al examinar
    #las predicciones de y con los valores reales que ha tomado y
    #plt.figure()
    #ax = plt.scatter(fila_x[:,0], y_train)
    #plt.show()

#%%
# Hacemos nuestro regressor de varios atributos
#Creamos un objeto que pueda manejar la regresión
our_regressor = model.LinearRegression()

n = int(dataset_array.shape[0] * 0.20)
x = dataset_array[:, [13]].reshape(-1, 1)
y = dataset_array[:, 14]
x_train = x[:-n, :]
x_test =  x[-n:, :]
y_train = y[:-n]
y_test = y[-n:]
our_regressor.fit(x_train, y_train)
y_predicted = our_regressor.predict(x_test)
plt.figure()
ax = plt.scatter(x_test, y_test)
plt.plot(x_test, y_predicted, '-r', linewidth=3) 
plt.show()
mse = metrics.mean_squared_error(y_test, y_predicted)
print('mse:' + ' ' + str(mse))
r2 = metrics.r2_score(y_test, y_predicted)
print('r2 :' + ' '+ str(r2))
# lo mejor es un solo atributo 

#%%

# los m y b parametros son iniciados de forma aleatoria
# por que vamos a minimizar ese error con el descenso del gradienteprint(data[:,:columnas])


x = np.transpose(x)
x = x[0]

m = 0
b = 0

L = 0.01 # tasa de aprendizaje
iteraciones = 1000 # numero de iteraciones a su gusto (numero de pasos en nuestra analogía)
coste = []
mejor = False

n = float(len(x))
i = 0
print('Esto es n')
print(n)
for i in range(iteraciones):
    y_pred = m * x + b # nuestro modelo
    z = y_pred - y
    D_b = (1/n) * sum(z)  # Derivada parcial con respecto a b 
    z = x* z
    D_m = (1/n) * sum(z)  # Derivada parcial con respecto m 

    # actualizamos los nuevos valores 
    # (damos un paso en la zona baja para reducir el error)
    m = m - L * D_m  #theta[1]
    b = b - L * D_b  #theta[0]
    
    #funcion de coste   
    z = (m * x + b) - y 
    z = z**2
    coste.append((sum(z))/(2*n))
    if(coste[i] < 10**(-3)):
        mejor = True
        break;

#muestro grafica de como cotes varian
plt.plot(range(i+1), coste,'-r', linewidth=3)
plt.show()

#muestro grafica de la regresion lineañ
print('m y b')    
print(m,b) 
print('coste mse')
print(coste[i])
print(i)
plt.scatter(x, y)
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = b + m * x_vals
plt.plot(x_vals, y_vals, color="red")
plt.show()
print(coste[len(coste)-1])

#vale por fin ha salido