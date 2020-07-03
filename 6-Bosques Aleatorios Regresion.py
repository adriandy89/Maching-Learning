# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 04:48:46 2020

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

######## PREPARAR LA DATA BOSQUES ALEATORIOS REGRESION ############

#Seleccionamos la columna 6 de dataset como variable independiente (numero habitaciones)
#los datos se encuentran almacenados en numpy(np)
x_bar= boston.data[:, np.newaxis, 5]

#defino los datos correspondientes a las etiquetas
y_bar= boston.target

#Graficamos los datos correspondientes
plt.scatter(x_adr,y_adr)
plt.show()

######## IMPLEMENTACION ARBOLES DE DECISION REGRESION ############

from sklearn.model_selection import train_test_split

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
x_train, x_test, y_train, y_test= train_test_split(x_bar, y_bar, test_size=0.2)

from sklearn.ensemble import RandomForestRegressor

#Defino el Algoritmo a Utilizar - n_estimators=Numero de Arboles, max_depth=Profundidad de 1 Arbol
bar= RandomForestRegressor(n_estimators=300, max_depth=8)
