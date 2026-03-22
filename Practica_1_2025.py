#!/usr/bin/env python
# coding: utf-8

# <br><br><br>
# <h2><font size=6>Práctica 1</font></h2>
# 
# 
# 
# <h1><font size=7>Árboles de decisión</font></h1>
# 
# <br>
# <div style="text-align: right">
# <font size=4>Pablo Torrijos Arenas (Pablo.Torrijos@uclm.es)</font><br>
# <font size=4>José Miguel Puerta Callejón (Jose.Puerta@uclm.es)</font><br>
# </div>

# In[ ]:





# In[ ]:





# In[ ]:





# **<font color="#B30033" size=5>Estudiantes: </font>** 
# 
# * Guillermo Morcillo Conchán
# * David Gómez Aniorte

# ---
# 
# # 1. Introducción
# 
# El objetivo de esta práctica es estudiar el uso de árboles de decisión para la predicción del ingreso de distintas personas en función de sus datos censales. Para ello usaremos la base de datos [`adult`](https://archive.ics.uci.edu/dataset/2/adult), también conocida como [`census-income`](https://archive.ics.uci.edu/dataset/20/census+income). 
# 
# En esta práctica comenzaremos explorando `scikit-learn` y su implementación de los árboles de decisión, realizando un estudio comparativo de los distintos hiperparámetros que ofrece. 
# 
# Posteriormente, se proporciona el esqueleto para la implementación del algoritmo C4.5 que usaremos como base para el resto de la práctica. A partir de ella, se pide:
# - Capacidad de tratar con variables y discretas continuas.
# - Implementar el error de clasificación, el índice GINI y la entropía condicional para el cálculo del error.
# - Poda del árbol.
# - Estudio del algoritmo implementado.
# 
# Baremo de puntuaciones:
# 
# | Tarea                     | Peso | 
# |----------|----------|
# | Estudio comparativo con `scikit-learn`      | 10%   |
# | Variables discretas       | 15%   |
# | Variables continuas       | 25%   |
# | Implementación de las métricas           | 10%   |
# | Poda del árbol            | 25%   |
# | Estudio final del algoritmo implementado           | 15%   |
# 
# 

# ---
# 
# # 2. Carga del dataset
# 
# El dataset que usaremos trata de predecir si los ingresos son superiores o inferiores a 50K en base a una serie de variables. Para cargar los datos usaremos `pandas`, mientras que `numpy` será necesario para realizar diversas funciones a lo largo de la práctica.

# In[2]:


import numpy as np
import pandas as pd



# In[3]:


df = pd.read_csv('adult.csv')
df


# ## 2.1. Análisis exploratorio
# 
# Podemos ver información de las distintas variables con `df.info()`:

# In[4]:


df.info()


# Así, vemos como efectivamente tenemos tanto variables categóricas como numéricas. En principio parece estar todo correcto, sin valores perdidos, pero si observamos los valores únicos de cada variable:

# In[5]:


df.nunique()


# In[6]:


df.apply(lambda col: col.sort_values().unique())


# Podemos ver cómo en `workclass`, `occupation` y `native-country` hay valores desconocidos representados por `?`. Vamos a ver cómo quedaría nuestro DataFrame si los reemplazamos por `NaN` para que `pandas` los reconozca como valores perdidos:

# In[7]:


df.replace('?', np.nan).info()


# Así, ahora podemos ver cómo la cuenta de valores no nulos ha cambiado. Por defecto, como las variables eran categóricas, estaba contando las `?` como una categoría más. 
# 
# Cuando conocemos la causa de los valores perdidos puede tener sentido dejarlos como una categoría más. Por ejemplo, suponed que estamos recogiendo datos de un radar en el que la velocidad máxima que puede medir son 200 km/h. Si un coche pasa a 215 km/h el radar nos daría un `?` en ese dato, pero si sabemos el motivo de estos valores perdidos, podríamos cambiar el nombre de esa categoría a `>200km/h`.
# 
# Ya que en esta práctica no vamos a introducir el manejo de los valores perdidos en nuestros árboles de decisión, y puesto que dichos valores solo aparecen en variables categóricas, por simplicidad vamos a dejar la base de datos tal cual está, contando a `?` como un valor categórico más. 

# ## 2.2. De `pandas` a `numpy`
# 
# A continuación vamos a transformar nuestros datos en arrays de `numpy` ya que los necesitaremos para trabajar con ellos posteriormente. `pandas` tiene muchas características muy útiles para hacer el análisis exploratorio y el preprocesamiento de los datos gracias a sus funciones de selección, agregación, agrupación... pero posteriormente todos los algoritmos de aprendizaje automático suelen trabajar con arrays de `numpy` dada su velocidad.
# 
# Vamos a empezar con los nombres de las variables. Por un lado vamos a guardar cuáles son nuestras variables predictoras y cuál nuestra variable objetivo.

# In[8]:


features = df.columns
features


# In[9]:


attributes = features[:-1]
target = features[-1]

print('Predictoras:',attributes)
print('Objetivo:',target)


# Además, vamos a distinguir entre variables continuas (las que son de tipo `int64`) y discretas (de tipo `object`), ya que a la hora de hacer nuestros árboles de decisión habrá que tratarlas de forma distinta.

# In[10]:


cont_atts = df.columns[df.dtypes == 'int64']
cont_atts = cont_atts.drop('fnlwgt')

disc_atts = df.columns[df.dtypes == 'str'] 
disc_atts = disc_atts.drop(target)

print('Continuas:',cont_atts)
print('Discretas:',disc_atts)


# Finalmente, separamos el dataset en predictor y objetivo. Es convención en ciencia de datos usar $X$ para las variables predictoras e $y$ para la variable objetivo. La mayoría de los modelos se entrenan usando esas dos variables por separado.

# In[11]:


X, y = df[attributes].to_numpy(), df[target].to_numpy()
X


# In[12]:


y


# ## 2.3. Datos de test
# 
# Cuando nos enfrentamos a un problema de aprendizaje automático, es imprescindible que los datos de test que usemos para medir el rendimiento del modelo sean distintos a los datos con los que se entrena. En este caso, como los autores del conjunto de datos nos proporcionan un conjunto de datos separado para test, lo usaremos directamente. Si no, tendríamos que dividir el conjunto de datos original en dos partes, una para entrenar y otra para test.
# 
# **Nota:** En la práctica, es común dividir el conjunto de datos en tres partes: entrenamiento, validación y test. La validación se usa para ajustar los hiperparámetros del modelo, y el conjunto de test se usa para medir el rendimiento final del modelo. En este caso, como no vamos a ajustar hiperparámetros y por simplicidad, no usaremos conjunto de validación.

# In[13]:


df_test = pd.read_csv('adult_test.csv')
df_test


# In[14]:


X_test, y_test = df_test[attributes].to_numpy(), df_test[target].to_numpy()
X_test


# In[15]:


y_test


# ---
# 
# # 3. Árboles de decisión en `scikit-learn`
# 
# 

# Para esta práctica vamos a utilizar el módulo [tree](http://scikit-learn.org/stable/modules/tree.html) de `scikit-learn`. Esta librería permite utilizar diversos algoritmos de _machine learning_ en Python, siendo los árboles de decisión uno de ellos. En particular, utilizaremos [`DecisionTreeClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html), la implementación de un árbol de decisión para problemas de clasificación de `scikit-learn`. 
# 
# La implementación que tiene `scikit-learn` de los árboles de decisión no es exactamente la del C4.5 si no que el algoritmo se llama CART. Existen algunas diferencias, pero la que más nos afectará es que los árboles generados son **binarios** y que no puede tratar variables discretas sin un procesado previo. 
# 
# Por tanto, para poder usar este algoritmo con nuestros datos tendremos que convertir las variables categóricas a numéricas. Para ello, podríamos usar el método [`OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) de `scikit-learn`, pero usaremos el método `df.get_dummies()` que nos proporciona directamente `pandas` ya que es similar y más simple de aplicar. Este método básicamente crea una nueva columna por cada valor posible de cada variable categórica y pone un 1 en la columna correspondiente al valor de la fila y un 0 en las demás, pasando así de variables categóricas a numéricas:
# 
# ![OneHotEnconding](./imagenes/get_dummies.png)

# ## 3.1. Transformación de los datos para usarlos con `DecisionTreeClassifier`
# 
# Vamos a transformar nuestros datos:

# In[16]:


df_ohe = pd.get_dummies(df.drop(target, axis=1))
df_ohe


# In[17]:


df_test_ohe = pd.get_dummies(df_test.drop(target, axis=1))
df_test_ohe


# In[18]:


set(df_ohe.columns) - set(df_test_ohe.columns)


# Como en el test nos falta una columna (ya que `Holand-Netherlands` no aparece en la variable `native-country`), tenemos que añadirla para no tener problemas posteriores.

# In[19]:


df_ohe, df_test_ohe = df_ohe.align(df_test_ohe, join='outer', axis=1, fill_value=0)
df_test_ohe


# In[20]:


X_ohe, X_test_ohe = df_ohe.to_numpy(), df_test_ohe.to_numpy()
X_ohe


# ## 3.2. Estudio de `DecisionTreeClassifier`

# El `DecisionTreeClassifier` cuenta con una serie de hiperparámetros con los que podemos ajustar su funcionamiento. Algunos de los que nos pueden ser más útiles son:
# * `criterion`: Especifica la función para medir la calidad de una partición. Puede ser `gini` o `entropy`.
# * `max_depth`: Profundidad máxima del árbol. 
# * `min_samples_leaf`: Mínimo número de ejemplos que debe haber en una hoja.
# 
# En la documentación de `scikit-learn` está toda la información sobre los hiperparámetros del algoritmo. Por lo que si se desea se puede extender el estudio probando más configuraciones. Para ello se debe especificar que hiperparámetros extra se han seleccionado, para que sirven y como afectan al árbol y justificar dicho comportamiento con los resultados obtenidos.

# In[21]:


from sklearn import tree 
from sklearn.tree import DecisionTreeClassifier


# In[22]:


criterion = 'entropy'
max_depth = 2 #Bajo 2-4, medio 8-12 y alto 20-30
min_samples_leaf = 100  #Bajo 1-5, medio 50-20 alto 1000-3000



# La siguiente crea un objeto `DecisionTreeClassifier` especificando los parámetros anteriores, y genera el arbol a partir de los datos con el método `fit(X,y)`.

# In[23]:


arbol = DecisionTreeClassifier(criterion = criterion,
                               max_depth = max_depth,
                               min_samples_leaf = min_samples_leaf)
arbol.fit(X_ohe,y)


# El árbol se puede visualizar mediante la función `plot_tree()`:

# In[24]:


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,5)

_ = tree.plot_tree(arbol, filled=True, rounded=True, fontsize=10, feature_names=df_ohe.columns, class_names=arbol.classes_)


# También podemos ver el porcentaje de aciertos o *accuracy* obtenido con el árbol tanto en el conjunto de entrenamiento como en el de test mediante el método `score()`:

# In[25]:


print('Accuracy train:\t', arbol.score(X_ohe,y))
print('Accuracy test: \t', arbol.score(X_test_ohe,y_test))


# Como el árbol que hemos creado está muy limitado a solo dos niveles de profundidad, el rendimiento del algoritmo en los datos de test es igual o incluso superior al que obtiene al intentar predecir directamente los mismos datos de entrenamiento con los que ha sido entrenado. Para comparar, vamos a crear un árbol por defecto (sin limitar):

# In[26]:


arbol2 = DecisionTreeClassifier()
arbol2.fit(X_ohe,y)


# Si vemos la puntuación que obtiene, al no estar limitado, sobreajusta al máximo a los datos de entrenamiento (se los está aprendiendo de memoria). Esto hace que luego en el conjunto de test obtenga un resultado mucho peor, incluso peor al del árbol básico de 2 niveles.

# In[27]:


print('Accuracy train:\t', arbol2.score(X_ohe,y))
print('Accuracy test: \t', arbol2.score(X_test_ohe,y_test))


# **Nota:** Este árbol no lo dibujamos porque al ser tan grande, tarda una eternidad y no se ve nada. El árbol limitado tenía 7 nodos, este tiene 9343:

# In[28]:


print('Número de nodos limitado:', arbol.tree_.node_count)
print('Número de nodos hoja limitado:', arbol.tree_.n_leaves)

print('\nNúmero de nodos:', arbol2.tree_.node_count)
print('Número de nodos hoja:', arbol2.tree_.n_leaves)


# ## **<font color="#B30033" size=6>TAREA: </font>** Estudio de diferentes configuraciones
# 
# Debes llevar a cabo un estudio donde debes variar los hiperparámetros del árbol para obtener un buen clasificador. Además, como mínimo se debe mostrar información sobre el `score` obtenido tanto con los datos de entrenamiento como de test, el número de nodos del árbol y la cantidad de nodos hoja del mismo. 
# 
# Después responde a las siguientes preguntas:
# * ¿Qué efecto observas con la variación de cada uno de los parámetros?
# * ¿Qué configuración escogerías para obtener un buen clasificador? Justifica tu respuesta.
# 
# Consejos:
# * Fíjate en los parámetros por defecto del algoritmo y en la explicación proporcionada para ajustar correctamente los valores. 
# * Los datos están desbalanceados (hay muchos más casos para el valor de la clase `<=50K` que para `>50K`, como se puede ver en la primera de las gráficas de abajo). En este caso, si predecimos siempre `<=50K` obtenemos un 0.7592 de accuracy, cuando está claro que es una predicción muy mala. Por ello, puede ser interesante utilizar además otras métricas como por ejemplo F-score ([`f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) en `scikit-learn`) que tengan en cuenta los valores de *precision* and *recall*. 
# * Se recomienda también mostrar información en forma de gráficas, ya pueden ser en el propio Python o incluso con Excel. En Python, una de las opciones más sencillas es usar la librería `seaborn`. A continuación se dejan una serie de ejemplos de gráficas usando `seaborn`.

# In[29]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# Métricas accuracy y F1 cuando se predice siempre la clase mayoritaria
print('Zero-R accuracy: ', accuracy_score(y == '>50K', np.repeat(0, len(y))))
print('Zero-R F-score: ', f1_score(y == '>50K', np.repeat(0, len(y))))


# In[30]:


import seaborn as sns
sns.set(style="darkgrid")

# Número de valores para cada clase (categórica)
g = sns.countplot(df, x=target)


# In[31]:


# Relación entre la edad (numérica) y las horas de trabajo por semana (numérica), diferenciando por clase
g = sns.lineplot(df, x='age', y='hours-per-week', hue='income')


# In[32]:


# Relación entre el nivel de educación (numérica) y las horas de trabajo por semana (numérica), diferenciando por clase y eliminando los intervalos de confianza
g = sns.lineplot(df, x='education-num', y='hours-per-week', hue='income', errorbar=None)


# In[33]:


# Relación entre el tipo de trabajo (categórica) y las horas de trabajo por semana (numérica), diferenciando por clase
g = sns.barplot(df, x='workclass', y='hours-per-week', hue='income')


# In[ ]:


#En esta celda se va a realizar una funcion para que sea la que realice los experimentos
import time as t

def experimento(df,df_test,criterion,prof,min_muestras):
    df_ohe = pd.get_dummies(df.drop(target, axis=1)) #Nuevo dataframe con variables predictoras. Elimina última variable (si o no <50K). Divide en columnas binarias
    df_test_ohe = pd.get_dummies(df_test.drop(target, axis=1))
    df_ohe, df_test_ohe = df_ohe.align(df_test_ohe, join='outer', axis=1, fill_value=0) #Mismas columnas mismo orden. Axis->Columna. Rellenar con 0

    y,y_test= df[target].to_numpy(),df_test[target].to_numpy() #Crea los 2 vectores. Pasa columna pandas a array de numpy con income
    X_ohe, X_test_ohe = df_ohe.to_numpy(), df_test_ohe.to_numpy()

    arbol = DecisionTreeClassifier(criterion=criterion,max_depth=prof,min_samples_leaf=min_muestras)
    arbol.fit(X_ohe,y) #Fit entrena modelo

    puntuacion,puntuacion_test = arbol.score(X_ohe,y),arbol.score(X_test_ohe,y_test) #Saca % aciertos(Accuracy)
    print('Accuracy train:\t', puntuacion)
    print('Accuracy test: \t', puntuacion_test)

    return (puntuacion,puntuacion_test)


# In[35]:


#En estos experimentos vamos a graficar tanto la precisión como el tiempo con respecto a las distintas variables que se nos presentan
#Primero establecemos una lista con los distintos criterios de evaluacion
criterios = ['entropy','gini']


#Aqui realizaremos los experimentos para distinto numero de profundidades, manteniendo el min_samples_leaf a 2000
for c in criterios:
    tiempos = []
    precision = []
    precision_test = []
    for i in range (2,30): #Variamos max depth
      print("Profundidad", i )
      inicio = t.time()
      p,p_t = experimento(df,df_test,c,i,1000) #Hacer el experimento con 1000 ejemplos por hoja
      tiempos.append(t.time() - inicio) 
      precision.append(p)
      precision_test.append(p_t)
       

    #Hago las gráficas
    profundidades = list(range(2,30))

    plt.figure()
    plt.plot(profundidades, precision, label="Train")
    plt.plot(profundidades, precision_test, label="Test")
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Profundidad ({c})")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(profundidades, tiempos, label="Tiempos")
    plt.xlabel("max_depth")
    plt.ylabel("Times")
    plt.title(f"Times vs Profundidad ({c})")
    plt.legend()
    plt.show()
    



# In[36]:


#Aqui realizaremos los experimentos para distinto numero minimo de ejemplos por hoja, manteniendo max_depth = 15
for c in criterios:
    tiempos = []
    precision = []
    precision_test = []
    for i in range (100,3000,100): #Variamos min samples leaf
      print("min_samples_leaf", i )
      inicio = t.time()
      p,p_t = experimento(df,df_test,c,15,i) #Hacer el experimento con profundidad 15
      tiempos.append(t.time() - inicio) 
      precision.append(p)
      precision_test.append(p_t)
       

    #Hago las gráficas
    min_samples = list(range(100,3000,100))

    plt.figure()
    plt.plot(min_samples, precision, label="Train")
    plt.plot(min_samples, precision_test, label="Test")
    plt.xlabel("mins_samples_leaf")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs min_samples ({c})")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(min_samples, tiempos, label="Tiempos")
    plt.xlabel("mins_samples_leaf")
    plt.ylabel("Times")
    plt.title(f"Times vs min_samples ({c})")
    plt.legend()
    plt.show()


# --- 
# 
# # 4. Implementación de un árbol de clasificación
# 
# En este apartado, vamos a implementar un árbol de clasificación C4.5. Se proporciona un modelo básico capaz de tratar variables categóricas, realizando las divisiones por error simple, y sin poda. Hay que ampliar el modelo para que cuente con las siguientes características:
# 1. Utilizar el índice GINI para el cálculo del error.
# 2. Utilizar la entropía condicional para el cálculo del error.
# 3. Utilizar variables continuas en el entrenamiento y predicción.
# 4. Poda del árbol.
# 
# Para la estructura del código vamos a seguir la de los algoritmos de `scikit-learn`. Por tanto, nuestro modelo `C45Classifier` heredará de [`BaseEstimator`](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html), la clase base para todos los estimadores de `scikit-learn`, y de [`ClassifierMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html), la clase base de los clasificadores. Además, debemos implementar los siguientes métodos principales: 
# 
# - `__init__()`: Constructor del modelo, recibirá los hiperparámetros necesarios.
# - `fit(X,y)`: Método de entrenamiento del modelo. Recibe $X$ e $y$ y devuelve el modelo ya entrenado.
# - `predict(X)`: Método de predicción del modelo. Recibe $X$ como un conjunto de instancias a predecir y devuelve $y_{pred}$, un vector de predicciones asociadas a $X$.
# 
# Por otro lado, está el siguiente método que, si bien es importante, al heredar de `ClassifierMixin` ya viene establecido por defecto a `accuracy_score`:
# - `score(X,y)`: Método de evaluación del modelo. Recibe $X$ e $y$, predice $y_{pred}$ a partir de $X$, y devuelve el porcentaje de acierto de $y_{pred}$ respecto a $y$.

# ## 4.1 Clase `Node`
# Antes de implementar la clase principal `C45Classifier`, vamos a crear una clase `Node` que codifique la información necesaria para cada uno de los nodos del árbol. Cuenta con las siguientes funciones:
# - `__init__(self):` Constructor. En él inicializamos las variables necesarias, explicadas en los comentarios del código.
# - `__str__(self):` Método que nos permite imprimir nuestros árboles.
# - `predict(self,x):` Método que nos permitirá hacer predicciones recursivamente hasta llegar a un nodo hoja. Cuando el `Node` es hoja devuelve el valor de su clase, y si no, tendrá que llamar a la función `predict(x)` del hijo que corresponda. 
# 
# ## **<font color="#B30033" size=6>TAREA: </font>** Método predict para variables continuas
# El método `predict` actualmente se proporciona adaptado a la predicción de variables discretas. Deberéis ampliarlo para que funcione cuando la variable del nodo es continua.

# In[37]:


import random

class Node:
    def __init__(self):
        # Indica si el nodo es una hoja, o no
        self.is_leaf = False

        # Atributos relacionados con la variable que representa el nodo
        self.is_num = True      # Indica si la variable es numérica (True) o categórica (False)
        self.cat_dict = None    # Diccionario para variables categóricas con formato {valor: indice}
        
        # Atributos cuando el objeto es una raíz
        self.var = None         # Nombre de la variable de corte
        self.var_index = -1     # Índice de la variable de corte
        self.cut_value = 0      # Valor de la variable de corte, en caso de ser numérica
        self.children = []      # Lista de hijos

        # Atributos cuando el objeto es una hoja
        self.class_value = -1       # Valor de la clase si el nodo es hoja
        self.class_count = (0,0)    # Tupla con el formato (casos con valor class_value, casos totales en la hoja)(numero de casos acertados, número de casos totales)

        # Profundidad del nodo
        self.depth = -1

    def __str__(self):
        output = ''
        if(self.is_leaf):
            output += 'Class value: ' + str(self.class_value) + '\tCounts: ' + str(self.class_count)
        else:
            output += 'Feature '+ str(self.var)
            for i in range(len(self.children)):
                output += '\n'+'\t'*(self.depth+1)+str(self.cut_value)+': '+str(self.children[i]) 
            
        return output
    
    # Esta función nos servirá para hacer predicciones recursivamente hasta llegar a un nodo hoja. Debe ser completada
    # Simplemente tiene que hacer las predicciones y devolverlas, los calculos de los distintos valores que hay que comprobar y devolver (cut_value, class_value,etc...) se realizan dentro del desarrollo del árbol
    def predict(self,x):
        if self.is_leaf:
            return self.class_value
        else:
            value = x[self.var_index] # Buscamos la mejor categoría
            if self.is_num:
                if not isinstance(value, (int, float)): #para los numeros comprobamos si es entero o float, si no lo es devolvemos error
                    raise ValueError(f"Expected numeric for {self.var}, got {type(value)}: {value}")
                if value <= self.cut_value: # devolvemos la mejor predicción dependiendo si es menos (nos vamos por el primero) o si es mayor (nos vamos por el segundo)
                    return self.children[0].predict(x)
                else:
                    return self.children[1].predict(x)
            else:
                if not isinstance(value, str): #para los discretos, comprobamos si es un string, si no lo es devolvemos errror
                    raise ValueError(f"Expected string for {self.var}, got {type(value)}: {value}")
                if value not in self.cat_dict: # Si el valor a buscar no aparece en el diccionario asignamos el primer hijo, si no cogemos el índice
                    child = 0 
                else:
                    child = self.cat_dict[value]
                return self.children[child].predict(x) # devolvemos la predicción del hujo correcpondiente


# ## 4.2 Clase `C45Classifier`
# Esta será la clase principal, que representará nuestro clasificador C4.5. Los argumentos que recibirá serán los siguientes:
# * `vars`, `disc`, `cont`: 3 listas. Nombres de las variables con el mismo orden con el que aparecen en $X$, y de ellas, cuáles son discretas y cuales son continuas. No sería estrictamente necesario, pero simplificará bastante el desarrollo de la práctica.
# * `max_depth`: Profundidad máxima del árbol. Si no se especifica, será 2.
# * `criterion`: Criterio de partición. Puede tomar los valores `classification_error`, `entropy` y `gini`. Si no se especifica, será 'entropy'.
# * `prune`: Booleano. Si es `True`, se podará el árbol. Si no se especifica, será `False`.
# 
# 
# ## **<font color="#B30033" size=6>TAREA: </font>** Implementación del índice GINI y la entropía condicional
# 
# 
# ## **<font color="#B30033" size=6>TAREA: </font>** Uso de variables continuas
# 
# 
# ## **<font color="#B30033" size=6>TAREA: </font>** Poda del árbol
# 

# In[38]:


from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from collections import Counter

class C45Classifier(BaseEstimator, ClassifierMixin):

    # Constructor de la clase, aquí se definen e inicializan las variables de la clase.
    def __init__(self, vars, disc, cont, max_depth=2, criterion='classification_error', prune=False):
        self.max_depth = max_depth
        self.criterion = criterion
        self.prune = prune

        self.vars = vars
        self.disc = disc
        self.cont = cont

        # Diccionario que nos permitirá convertir el nombre de la variable en su índice.
        self.features_dict = {feat: i for i, feat in enumerate(self.vars)}

        # Raíz del árbol
        self.tree = Node()   


    # Función para entrenar el modelo.
    def fit(self, X, y):#Contruye árbol de decisión
        # Llamada a la función recursiva que aprende el árbol.
        self._partial_fit(X, y, self.tree, 0, set([]))

        if self.prune:
            self._prune_tree()
        
        return self
    

    # Función para hacer predicciones.
    def predict(self, X):#Usar árbol ya clasificado para clasificar nuevos datos
        return np.array([self.tree.predict(x) for x in X])
    

    # Función recursiva que busca la variable y corte que maximiza la ganancia de información.
    # - Las variables continuas se tratan con un corte binario, lo que quiere decir que pueden ser usadas multiples veces. 
    # - Las variables discretas ramifican tantas veces como valores tengan, asi que solo pueden ser usadas una vez por camino, 
    #   debiendo almacenarlas en el conjunto `borradas`. 
    def _partial_fit(self, X, y, current_tree, current_depth, borradas):
        def _make_leaf(): #Convierte nodo en hoja
            current_tree.is_leaf = True
            counts = Counter(y)
            max_value = counts.most_common(1) # most_common(1) devuelve una lista con el elemento más común y su frecuencia.
            current_tree.class_value = max_value[0][0]#
            current_tree.class_count = (max_value[0][1], len(y)) #Casos de esa clase en esa hoja
            return
        
        # Antes de nada, si hemos alcanzado la profundidad máxima, el nodo se convierte en hoja.
        if current_depth >= self.max_depth:
            _make_leaf()
            return

        # Primero obtenemos el mejor punto de corte para el nodo actual dependiendo del criterio.
        best_var, cut_value, is_num = self._split(X, y, borradas, self.criterion) #Criterios son classification_error, entropy, gini
        #best_var es nombre variable con mejor corte
        #cut_value es valor partición

        # Si no hay ninguna partición que mejore la actual, el nodo se convierte en hoja.
        if best_var is None:
            _make_leaf()
            return
    
        # Antes de llamar a la función recursiva, hay que actualizar los valores del árbol.
        borradas_copy = borradas.copy()
        if not is_num:    # Solo borramos las variables categóricas ya que estarán totalmente particionadas.
            borradas_copy.add(best_var)
            current_tree.is_num = False

        current_tree.is_leaf = False
        current_tree.depth = current_depth
        current_tree.var = best_var
        current_tree.var_index = self.features_dict[best_var]

        # Finalmente, se hace la llamada recursiva en función de si es numérica o categórica.
        if is_num:
            nodoMayor=Node()  #Se crean los dos nodos y las particiones
            nodoMenor=Node()
            izquierda,derecha = X[:,current_tree.var_index] <= cut_value, X[:,current_tree.var_index] > cut_value

            X_izq , y_izq = X[izquierda], y[izquierda] # Separamos las partes
            X_der , y_der = X[derecha], y[derecha]
            
            self._partial_fit(X_izq, y_izq, nodoMenor, current_depth + 1, borradas_copy) #Hacemos las llamadas para calcular los dos hijos
            self._partial_fit(X_der, y_der, nodoMayor, current_depth + 1, borradas_copy)

            current_tree.children = [nodoMayor, nodoMenor] #Guardamos los hijos
            current_tree.cut_value = cut_value
            current_tree.is_num = True
        else:
            ramificaciones = np.unique(X[:,current_tree.var_index])
            i = 0
            cat_dict = {}
            for rama in ramificaciones:
                nodo = Node()
                dato = X[:, current_tree.var_index] == rama
                X_sub,y_sub = X[dato],y[dato]

                self._partial_fit(X_sub,y_sub,nodo,current_depth + 1, borradas_copy)

                current_tree.children.append(nodo)
                cat_dict[rama] = i
                i+=1
            current_tree.cat_dict = cat_dict
            current_tree.is_num = False
            current_tree.cut_value = None
        return


    # Cálculo del mejor punto de corte en función de: Error de clasificación.
    #classification_error es el % datos clasifican incorrectamente
    def _split(self, X, y, borradas, criterion='classification_error'):
        # Error actual (sin partición)
        error_best = self._compute_split_criterion(y, criterion) #_compute_split_criterion mejor punto corte

        best_var = None
        is_num = True
        cut_value = None    # Para variables categóricas no hay valor de corte (devolvemos None).
        
        for var in self.vars:
            # Saltamos variables que ya fueron usadas (en discretas)
            if var in borradas:
                continue
                
            index = self.features_dict[var]
            
            if var in self.disc: #Hacemos una ramificación por cada opción.
                # Obtener valores únicos de la variable discreta
                ramificaciones = np.unique(X[:, index]) #Con esto saco todas ramificaciones
                
                #Calcular media todas ramas
                error = 0
                for val in ramificaciones: #X[:, index] extrae una columna
                    esta = X[:, index] == val #array de si está dicho dato
                    y_subset = y[esta] #Todos los verdaderos por rama
                    #Error para este subset
                    error_subset = self._compute_split_criterion(y_subset, criterion)
                    #Peso es la proporción de ejemplos en este subset
                    peso = len(y_subset) / len(y) #Divido el la rama entre todos los q hay
                    error += peso * error_subset
                
                # Comparar con el mejor error encontrado hasta ahora
                if error < error_best: #Es mejor
                    error_best = error
                    best_var = var
                    is_num = False
                    cut_value = None
                    #borradas.add(var)

            elif var in self.cont:
                valores = np.unique(X[:, index]) #como arriba

                if len(valores) > 1: # Si hay suficientes valores, probamos puntos de corte entre ellos
                    for i in range(len(valores) - 1):
                        umbral = (valores[i] + valores[i+1]) / 2 #media presente y siguiente
                        
                        #Partición izquierda. los q tienen <= umbral
                        left_mask = X[:, index] <= umbral
                        y_left = y[left_mask]
                        
                        #Partición derecha
                        right_mask = X[:, index] > umbral
                        y_right = y[right_mask]
                        
                        if len(y_left) > 0 and len(y_right) > 0: #No casos vacios
                            error_left = self._compute_split_criterion(y_left, criterion)
                            error_right = self._compute_split_criterion(y_right, criterion)
                            
                            weight_left = len(y_left) / len(y) #Proporcionalmente como arriba
                            weight_right = len(y_right) / len(y)
                            
                            error = weight_left * error_left + weight_right * error_right
                            
                            if error < error_best: #Guardar si si
                                error_best = error
                                best_var = var
                                is_num = True
                                cut_value = umbral
                                best_ramificaciones = None

            # Si conseguimos un error de 0 (óptimo), terminamos
            if error_best == 0:
                break

        if best_var in self.disc:
            borradas.add(best_var)
        return best_var, cut_value, is_num

    # Cálculo del mejor punto de corte en función de: Error de clasificación; Entropía; Índice Gini.
    def _compute_split_criterion(self, y, criterion='classification_error'):
        # Completar aquí si tenéis código común a los tres criterios.
        ramas,contaRamas = np.unique(y, return_counts=True) #Devuelve [A,B,c] y [1,5,2] por ejemplo
  
        if criterion == 'classification_error': #Apuesto a la clase q más se repite
            # indice = contaRamas.index(max(contaRamas)) #Saca el indice del numero max
            return 1 - (max(contaRamas)/sum(contaRamas)) #igue la fórmula
        elif criterion == 'entropy': #H(S) = −SUM(ci=1pilog2(pi))
            return - sum( contaRamas/sum(contaRamas) * np.log2(contaRamas/sum(contaRamas) ) ) #Como en clase sacábamos
                                                                            #cuanto % era
        elif criterion == 'gini':
            return 1 - sum((contaRamas/sum(contaRamas))**2) #Creo q es así x los paréntesis t tal
        else:
            raise ValueError('Criterio no válido.')

    
    # TODO: Completar esta función para realizar la poda del modelo.
    def _prune_tree(self): #por terminar
        self.k=2
        if self.tree:
            self.prune_recursive(self.tree)
    
    def laplace(self,N,n):
        if N == 0:
            return 0
        else:
            return (N-n + self.k-1)/(N+self.k)
    
    def prune_recursive(self,nodo):
        #Si es hoja devolvemos su error y su valor
        if nodo.is_leaf:
            n,N = nodo.class_count
            return nodo.is_leaf,self.laplace(N,n), nodo.class_count, nodo.class_value
        
        else: #Si tiene hijo hacemos llamada recursiva
            print("No soy nodo hoja")
            children = nodo.children if nodo.is_num else list(nodo.cat_dict.values())
            poda = True
            child_errors=[]
            N_children=[]
            dict_clases = {"<=50K":0, ">50K":0}
            if nodo.is_num: #Si es numérica podemos acceder a los hijos directamente
                for child in children:
                    is_leaf,child_error,(n_child,N_child),value=self.prune_recursive(child)
                    if not is_leaf: 
                        poda = False
                    child_errors.append(child_error)
                    N_children.append(N_child)
                    if value != -1:
                        dict_clases[value]+=n_child
            else: # Si es categórica accedemos a una lista de los índices de una lista (muy parecido a lo implementado en las numéricas)
                for index in children:
                    child = nodo.children[index]
                    is_leaf,child_error,(n_child,N_child),value=self.prune_recursive(child)
                    if not is_leaf: 
                        poda = False
                    child_errors.append(child_error)
                    N_children.append(N_child)
                    if value != -1:
                        dict_clases[value]+=n_child
            
            N_padre = sum(N_children)
            n_padre = max(dict_clases.values())
            child_errorssum=0
            if N_padre > 0:
                for i in range(child_errors.__len__()):
                    child_errorssum += (N_children[i]/N_padre) * child_errors[i]

            error_padre = self.laplace(N_padre,n_padre)
            
            if error_padre <= child_errorssum and poda:
                nodo.is_leaf = True
                nodo.class_value = max(dict_clases, key = dict_clases.get)
                nodo.class_count = (n_padre,N_padre)
                return True,error_padre,nodo.class_count,nodo.class_value
            else:
                return False,error_padre, nodo.class_count, nodo.class_value

    # Función para imprimir el modelo.
    def __str__(self):
        return str(self.tree)


# --- 
# 
# # 5. Pruebas y estudio del algoritmo implementado
# 
# Finalmente, se deberán realizar pruebas con el clasificador para verificar su funcionamiento. A continuación, se incluyen algunos ejemplos de ejecución. Podéis incluir estos ejemplos en vuestra entrega, pero deberéis añadir más para demostrar que todas las partes de la práctica funcionan correctamente (variables continuas/discretas; error de clasificación/entropía/gini; con poda/sin poda, etc.). Además, se deberá razonar por qué los resultados son distintos de un caso a otro. 
# 
# Este apartado es más "libre", por lo que podéis hacer todas las pruebas y comparaciones que consideréis relevantes. Por ejemplo, podéis comparar vuestro algoritmo con los valores obtenidos por los árboles de `scikit-learn`, medir tiempos de ejecución... Además, si habéis incluido alguna característica opcional o distintiva de vuestro algoritmo, también debéis explicarla en este apartado.
# 
# 
# ### IMPORTANTE
# 
# **Se deberá mantener la eficiencia del clasificador. Esto significa que el tiempo de entrenamiento del árbol utilizando variables discretas o ambos tipos de variables debe ser similar. Obviamente, será mayor al incluir variables continuas en comparación con entrenar solo con las discretas (ya que solo se pueden particionar una vez), pero debe mantenerse dentro de un orden de magnitud similar.**

# ### Variables discretas, profundidad máxima 3, criterion='classification_error', sin poda

# In[39]:


arbol = C45Classifier(attributes, disc_atts, [], max_depth=3, criterion='classification_error', prune=True)
arbol.fit(X,y)


# In[40]:


print("Error en train: ", arbol.score(X,y))
print("Error en test: ", arbol.score(X_test,y_test))


# ### Variables discretas, profundidad máxima 10, criterion='classification_error', sin poda

# In[41]:


# Al no estar implementado todavía el tratamiento de variables continuas, da igual que se especifiquen o no.
arbol = C45Classifier(attributes, disc_atts, [], max_depth=10, criterion='classification_error', prune=False)
arbol.fit(X,y)


# In[42]:


print("Error en train: ", arbol.score(X,y))
print("Error en test:  ", arbol.score(X_test,y_test))


# ### Variables discretas y continuas, profundidad máxima 2, criterion='classification_error', sin poda

# In[43]:


# Al no estar implementado todavía el tratamiento de variables continuas, da igual que se especifiquen o no.
arbol = C45Classifier(attributes, disc_atts, cont_atts, max_depth=2, criterion='classification_error', prune=True)
arbol.fit(X,y) 


# In[44]:


print("Error en train: ", arbol.score(X,y))
print("Error en test:  ", arbol.score(X_test,y_test))

