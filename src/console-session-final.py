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
from mpl_toolkits.mplot3d import axes3d, Axes3D 



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
'''
x_b = []
x_b = np.array(x_b)
x_b = np.transpose(x)
x_b = x_b[0] '''

x_b = dataset_array[:,13]
x_b = dataset_array[:-n,13]
x_test = dataset_array[-n:,13]
m = 0
b = 0
print(x_b)
L = 0.1 # tasa de aprendizaje
iteraciones = 200 # numero de iteraciones a su gusto (numero de pasos en nuestra analogía)
coste = []


num = float(len(x_b))
i = 0

#esto es para una sola variable
for i in range(iteraciones):
    y_pred = m * x_b + b # nuestro modelo
    z =  y_pred - y_train
    D_b = (1/num) * sum(z)  # Derivada parcial con respecto a b 
    z = x_b* z
    D_m = (1/num) * sum(z)  # Derivada parcial con respecto m 

    # actualizamos los nuevos valores 
    # (damos un paso en la zona baja para reducir el error)
    m = m - L * D_m  #theta[1]
    b = b - L * D_b  #theta[0]
    
    #funcion de coste   
    z = (m * x_b + b) - y_train 
    z = z**2
    coste.append((sum(z))/(2*num))
    if(i>0):
        if((abs(coste[i] - coste[i-1])) < 0.001):
            print(coste[i] - coste[i-1])
            break;

#muestro grafica de como cotes varian
plt.plot(range(i+1), coste,'-r', linewidth=3)
plt.show()

#muestro grafica de la regresion lineañ
print('m y b')    
print(m,b) 
print('coste mse')
print(i)
print(coste[i])
print(coste)
plt.scatter(x_test, y_test)
axes = plt.gca()
#x_vals = np.array(axes.get_xlim())
y_pred = b + m * x_test
plt.plot(x_test, y_pred, color="red")
plt.show()
print(coste[len(coste)-1])

#%%
#con todas las variables

def modelo(m,x,n, b):
    y_pred = m[0]*x[:,0]
    for i in range(1,14):
        y_pred = y_pred + ( x[:,i] * m[i])
    y_pred = y_pred + b
    return y_pred

def calculo_thetas(x,y_pred,y,n,L,m,b):
    z = y_pred - y
    D_b = (1/n)* sum(z)
    D_b = D_b*L
    D_m = np.zeros(14)
    for i in range(14):
        aux = x[:,i] * z
        D_m[i] = (1/n) * sum(aux)
        
    b = b - D_b *L
    m = m - D_m*L
         
    return b, m

#no se porque me da error
def coste(y_pred,y,n):
    aux = y_pred - y
    aux = aux ** 2
    coste = sum(aux)/(2*n)
    return coste
            
L = 0.1 # tasa de aprendizaje
iteraciones = 400 # numero de iteraciones a su gusto (numero de pasos en nuestra analogía)
coste = []


print(dataset_array)
print(len(dataset_array))




print(x_b)
print(len(x_b))
x_mb = dataset_array[:-n,:]
x_test = dataset_array[-n:,:]
#n = float(len(x_mb[:, 0]))
i = 0
m = np.zeros(14)
b = 0
print('Esto es n')
print(len(x_mb))
print(n)
 



for i in range(iteraciones):
    y_pred = modelo(m,x_mb,num, b)
    #coste = coste(y_pred,y,n)
    
    aux = y_pred - y_train
    aux = aux ** 2
    coste.append(sum(aux)/(2*num))
    if(i > 0):
        if(abs(coste[i] - coste[i-1]) < 0.000001):
            break;
    b , m = calculo_thetas(x_mb,y_pred,y_train,num,L,m,b)
    
print (m,b)
print('i,coste')

print(coste[i])
print(num)

plt.plot(range(i+1), coste,'-r', linewidth=3)
plt.show()

#una variable, la mejor

plt.scatter(x_test[:,13], y_test)
axes = plt.gca()
y_pred = modelo(m,x_test,num,b)
#x_vals = np.array(axes.get_xlim())
#y_vals = b + m[13] * x_vals
y_pred = x_test[:,13]*m[13] + b
plt.plot(x_test[:,13], y_pred, color="red")
plt.show()



#%%

# generem dades 3D d'exemple 

#regr = regression(x_val, y_val) 
#predX3D = modelo(m,x_test,num,b)
print(x_test[:,12:13])
print(x_test[:,13])
predX3D = x_test[:,13]*m[13] + x_test[:,12]*m[12] + b
#plt3d =plt.axes(projection='3d')
#plt3d.plot_surface(xplot,yplot,zplot, color='red') 

# Afegim els 1's 
A = np.hstack((x_test[:,12:14],np.ones([x_test[:,12:14].shape[0],1]))) 
w = np.linalg.lstsq(A,predX3D)[0]
#Dibuixem

#1r creem una malla acoplada a la zona de punts per tal de representar el pla 
malla = (range(20) + 0 * np.ones(20)) / 10
malla_x1 = malla * (max(x_test[:,12]) - min(x_test[:,12]))/2 + min(x_test[:,12]) 
malla_x2 = malla * (max(x_test[:,13]) - min(x_test[:,13]))/2 + min(x_test[:,13])
#la fucnio meshgrid ens aparella un de malla_x1 amb un de malla_x2, per atot #element de mallax_1 i per a tot element de malla_x2. 
xplot, yplot = np.meshgrid(malla_x1 ,malla_x2)
# Cal desnormalitzar les dades 
def desnormalitzar(x, mean, std): return x * std + mean
#ara creem la superficies que es un pla 
zplot = w[0] * xplot + w[1] * yplot + w[2]
#Dibuixem punts i superficie 
#plt3d = plt.figure('Coeficiente prismatico -- Relacio longitud desplacament 3D', dpi=100.0) 
plt3d =plt.axes(projection='3d')
plt3d.plot_surface(xplot,yplot,zplot, color='red') 
plt3d.scatter(x_test[:,12],x_test[:,13],y_test)





#plt3d.scatter(x_test[:,12],x_test[:,13],y_test)
#plt3d.plot3D(x_test[:,12],x_test[:,13],predX3D, color='red')

print(y_predicted)
