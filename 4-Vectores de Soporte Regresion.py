# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 07:21:18 2020

@author: Adrian Duardo Yanes
"""
######## LIBRERIAS A UTILIZAR ############

#Se importan las librerias a utilizar
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

######## PREPARAR LA DATA ############

#Importamos los datos de ejemplo de la libreria de scikit-learn
boston= datasets.load_boston()
print(boston)
print()

######## ENTENDIMIENTO DE LA DATA ############

#Verifico la informacion contenida en el dataset
print('Informacion en el dataset: ')
print(boston.keys())
print()

#Verifico las caracteristicas del dataset
print('Caracteristicas del dataset: ')
print(boston.DESCR)
print()

#Verifico las cantidad de datos del dataset
print('Cantidad de datos: ')
print(boston.data.shape)
print()

#Verifico la Informacion de las columnas
print('Nombre de las Columnas: ')
print(boston.feature_names)
print()

######## PREPARAR LA DATA VESTORES DE SOPORTE DE REGRESION ############

#Seleccionamos la columna 6 de dataset como variable independiente (numero habitaciones)
#los datos se encuentran almacenados en numpy(np)
x_svr= boston.data[:, np.newaxis, 5]

#defino los datos correspondientes a las etiquetas
y_svr= boston.target

#Graficamos los datos correspondientes
plt.scatter(x_svr,y_svr)
plt.xlabel('Numero de Habitaciones')
plt.ylabel('Valor Medio')
plt.show()

######## IMPLEMENTACION VESTORES DE SOPORTE DE REGRESION ############

from sklearn.model_selection import train_test_split

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
x_train, x_test, y_train, y_test= train_test_split(x_svr, y_svr, test_size=0.2)

from sklearn.svm import SVR

#Defino el Algoritmo a Utilizar
svr= SVR(kernel='linear', C=1.0, epsilon=0.2)
# o sin especificar parametros
# svr= SVR()

#Entreno el modelo
svr.fit(x_train, y_train)

#Realizo una Prediccion
Y_pred= svr.predict(x_test)

#Graficamos los Datos junto con el Modelo
plt.scatter(x_test, y_test)
plt.plot(x_test, Y_pred, color='red', linewidth=3)
plt.title('VESTORES DE SOPORTE DE REGRESION')
plt.show()
print()

print('La Presicion del Modelo: (R cuadrado - cerca de 1 mejor)')
print(svr.score(x_train, y_train))

