# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 11:34:32 2020

@author: Adrian Duardo Yanes
"""
######## LIBRERIAS A UTILIZAR ############

#Se importan las librerias a utilizar
import numpy as np
from sklearn import datasets
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

######## PREPARAR LA DATA ARBOLES DE DECISION REGRESION ############

#Seleccionamos la columna 6 de dataset como variable independiente (numero habitaciones)
#los datos se encuentran almacenados en numpy(np)
x_adr= boston.data[:, np.newaxis, 5]

#defino los datos correspondientes a las etiquetas
y_adr= boston.target

#Graficamos los datos correspondientes
plt.scatter(x_adr,y_adr)
plt.xlabel('Numero de Habitaciones')
plt.ylabel('Valor Medio')
plt.show()

######## IMPLEMENTACION ARBOLES DE DECISION REGRESION ############

from sklearn.model_selection import train_test_split

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
x_train, x_test, y_train, y_test= train_test_split(x_adr, y_adr, test_size=0.2)

from sklearn.tree import DecisionTreeRegressor

#Defino el Algoritmo a Utilizar
adr= DecisionTreeRegressor(max_depth=5)

#Entreno el modelo
adr.fit(x_train, y_train)

#Realizo una Prediccion
Y_pred= adr.predict(x_test)

#Graficamos los Datos junto con la Prediccion
X_grid= np.arange(min(x_test), max(x_test), 0.1)
X_grid= X_grid.reshape((len(X_grid),1))

plt.scatter(x_test, y_test)
plt.plot(X_grid, adr.predict(X_grid), color='red', linewidth=3)
plt.title('ARBOLES DE DECISION REGRESION')
plt.show()
print()

print('La Presicion del Modelo: (R cuadrado - cerca de 1 mejor)')
print(adr.score(x_train, y_train))