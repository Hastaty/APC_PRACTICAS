#!/usr/bin/python3
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from IPython.display import display
import seaborn as sb


#Para comprobar el error cuadratico medio
def mean_squeared_error(y1, y2):
    # comprovem que y1 i y2 tenen la mateixa mida
    assert(len(y1) == len(y2))
    mse = 0
    for i in range(len(y1)):
        mse += (y1[i] - y2[i])**2
    return mse / len(y1)

def split_data(x, y, train_ratio=0.8):
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

#Leemos el fichero
dataframe=pd.read_csv('../data/bike_' + \
                      'sharing_hourly.csv', header=0, delimiter=',')





#veamos cuantas dimensiones y registros contiene
display(dataframe.shape)

#veamos esas filas
display(dataframe.head())

#descripcion datos
display(dataframe.describe())


#Se muestra un histograma con los datos, pero eliminando dos cosas
dataframe.drop(['instant','cnt'],1).hist()

plt.show()


#Se calcula una matriz de correlación de todas las columnas con todas las columnas
correlacio = dataframe.corr()
plt.figure()

ax = sb.heatmap(correlacio, annot=True, linewidths=.5)

#Quitamos algunas columnas inútiles
data_global=dataframe.drop(["instant", "dteday"], axis=1)




better_frame = data_global
better_frame['temp'] = better_frame['temp'] * 41.0
better_frame['atemp'] = better_frame['atemp'] * 50.0
better_frame['hum'] = better_frame['hum'] * 100.0
better_frame['windspeed'] = better_frame['windspeed'] * 67.0






#Y nos quedamos solo con los valores, sin las cabeceras
data=better_frame.values

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
    modelo=LinearRegression()
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


    

    #print("Len y test:"+str(len(y_test)))
    #print("Len y pred:"+str(len(predicciones_y)))
    #print(len(y_test))
    error_cuadratico_medio=mean_squeared_error(predicciones_y, y_test)
    print("El error cuadratico para el atributo "+str(data_global.columns[columna]) +" es "+str(error_cuadratico_medio))

    #Ahora calculamos el error cuadratico medio que sale al examinar
    #las predicciones de y con los valores reales que ha tomado y
    #plt.figure()
    #ax = plt.scatter(fila_x[:,0], y_train)
    #plt.show()
    
    