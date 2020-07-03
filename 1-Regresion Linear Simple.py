# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 00:47:27 2020

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

######## PREPARAR LA DATA PARA LA REGRESION LINEAL SIMPLE ############

#Seleccionamos la columna 5 de dataset como variable independiente (numero habitaciones)
#los datos se encuentran almacenados en numpy(np)
x= boston.data[:, np.newaxis, 5]

#defino los datos correspondientes a las etiquetas
y= boston.target

#Graficamos los datos correspondientes
plt.scatter(x,y)
plt.xlabel('Numero de Habitaciones')
plt.ylabel('Valor Medio')
plt.show()
 
######## IMPLEMENTACION DE REGRESION LINEAL SIMPLE ############

from sklearn.model_selection import train_test_split

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2)

#Defino el Algoritmo a Utilizar
lr= linear_model.LinearRegression()

#Entreno el modelo
lr.fit(x_train, y_train)

#Realizo una Prediccion
Y_pred= lr.predict(x_test)

#Graficamos los Datos junto con el Modelo
plt.scatter(x_test, y_test)
plt.plot(x_test, Y_pred, color='red', linewidth=3)
plt.title('Regresion Lineal Simple')
plt.xlabel('Numero de Habitaciones')
plt.ylabel('Valor Medio')
plt.show()
print()

print('DATOS DEL MODELO DE REGRESION LINEAL MULTIPLE')
print()
print('Valos de la pendiente o coeficiente "a":')
print(lr.coef_)
print('Valor de la interseccion o coeficiente "b":')
print(lr.intercept_)
print()
print('La Ecuacion del Modelo en igual a: ')
print('y = ', lr.coef_, 'x', lr.intercept_)
print()
print('La Presicion del Modelo: (R cuadrado - cerca de 1 mejor)')
print(lr.score(x_train, y_train))
