# -*- coding: utf-8 -*-
"""P1-Clasificacion.ipynb


## Descripción del problema
La hipótesis de partida es que es posible determinar (clasificar) el género de un usuario de Twitter en base a sus características. Por tanto, el problema queda definido como:

> El problema de clasificación a resolver consiste en determinar el género de un usuario de Twitter según las características de su perfil en dicha red social, así como sus mensajes y manera de expresarse.

### El conjunto de datos
Para esta práctica usaremos un conjunto de datos recopilado de [Data For Everyone Library](https://www.crowdflower.com/data-for-everyone/).

El conjunto de datos está formado por un único fichero CSV. Cada fila del archivo corresponde a un usuario de la red social Twitter.

Las características (o _features_) de este conjunto de datos son las siguientes:

* `_unit_id`: id único para el usuario
* `_golden`: metadatos de Crowdflower
* `_unit_state`: metadatos de Crowdflower
* `_trusted_judgments`: metadatos de Crowdflower
* `_last_judgment_at`: metadatos de Crowdflower
* `gender`: **esta es la clase a predecir (TARGET)**
* `gender:confidence`: margen de confianza del género
* `profile_yn`: indica si existe información del usuario del perfil
* `profile_yn:confidence`: valor de confianza de la característica anterior
* `created`: fecha y hora de creación de la cuenta
* `description`: descripción del usuario en su perfil
* `fav_number`: número de tweets 'gustados' por el usuario
* `gender_gold`: metadatos de Crowdflower
* `link_color`: valor hexadecimal del color configurado por el usuario para los enlaces
* `name`: nombre del usuario
* `profile_yn_gold`: metadatos de Crowdflower
* `profileimage`: enlace a la imagen de perfil del usuario
* `retweet_count`: número de retweets del usuario
* `sidebar_color`: color configurado por el usuario para la barra lateral
* `text`: texto de un tweet aleatorio del usuario
* `tweet_coord`: si el usuario tiene habilitada la geo-localización de sus tweets, las coordenadas en formato "\[latitud, longitud\]"
* `tweet_count`: número de tweets publicados por el usuario
* `tweet_created`: fecha y hora de creación del tweet seleccionado aleatoriamente
* `tweet_id`: id del tweet aleatorio
* `tweet_location`: localización del tweet aleatorio
* `user_timezone`: zona horaria declarada por el usuario



---

## Desarrollo de la práctica

Esta práctica ha sido desarrollada por:

* Miguel Hernández Roca
* Mohammed Makhfi Boulaich
"""

# Incluir aquí todas las librerías que se vayan a utilizar en el resto de
# la libreta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
import statistics as stat
from nltk.corpus import stopwords
# for quick and dirty counting
from collections import defaultdict

# Modelo Naive Bayes
from sklearn.naive_bayes import MultinomialNB
# Separar los datos para aplicar crosvalidación
from sklearn.model_selection import train_test_split
# Crea un documento de frecuencias
from sklearn.feature_extraction.text import CountVectorizer
# Codifica los datos en categorías
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import ComplementNB

"""### Carga del conjunto de datos
La carga del conjunto de datos se realiza mediante la URL del mismo:
"""

datos = pd.read_csv('https://drive.upm.es/index.php/s/OJEqz7CpfofJXcA/download', encoding='latin1')

"""### Análisis exploratorio de datos"""

datos.head(5)

datos.columns

datos.dtypes

# Eliminar las columnas que corresponden a los metadatos del CrowdFlower
cols = [col for col in datos.columns if (col.lower()[:1] != '_' and col.lower()[-4:] != 'gold')]
datos = datos[cols]

datos.shape

# Buscar filas duplicadas
filas_duplicadas_datos = datos[datos.duplicated()]
print('number of duplicate rows: ', filas_duplicadas_datos.shape)

# Eliminar filas duplicadas
datos = datos.drop_duplicates()
datos.head(5)

# Buscar valores nulos
print(datos.isnull().sum())

# Eliminar filas con valores nulos
datos = datos.dropna()

# Limpiar las columnas de color para que contenga valores válidos en HEX
def validar_color_hex(colName):
  values = []
  dropIndex = []
  for index, elem in enumerate(datos[colName]):
    x = re.findall("^([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$", str(elem))
    if x:
      values.append(x[0])
    else:
      dropIndex.append(index)
  datos = datos.drop(datos.index[dropIndex])
  datos[colName] = values

  validar_color_hex('link_color')
  validar_color_hex('sidebar_color')

datos['sidebar_color'].unique()

datos.sidebar_color.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title('Número de usuarios por color')
plt.ylabel('Número de usuarios')
plt.xlabel('Color')

"""### Transformaciones aplicadas"""

def normalizar_texto(txt):
  txt = str(txt).lower()

  # E
  txt = re.sub('\s\W+', ' ', txt)
  txt = re.sub('\W+\s', ' ', txt)
  txt = re.sub('\s+', ' ', txt)

  return txt

datos['text_norm'] = [normalizar_texto(s) for s in datos['text']]
datos['description_norm'] = [normalizar_texto(s) for s in datos['description']]

# Al parecer, filtrando los tweets por s
# es posible que las stopwords sean útiles para determinar el sexo porque, por ejemplo, un género puede utulizar mas artizulos
nltk.download("popular")
english_stopwords = stopwords.words('english')
count_vectorizer = CountVectorizer(stop_words=english_stopwords)

"""### Aplicando el clasificador de Naive Bayes"""

# Matriz de frecuencias
x = CountVectorizer().fit_transform(datos['text_norm'])

# Codificar la columna género
encoder = LabelEncoder()
y = encoder.fit_transform(datos['gender'])

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Naive Bayes
naive_bayes = MultinomialNB()
#naive_bayes.fit(x_train, y_train)

#print(naive_bayes.score(x_test, y_test))

kf = KFold(n_splits=5, random_state=0)
mnb_results = cross_validate(estimator=naive_bayes, X=x, y=y, scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'], cv=kf, return_train_score=True)



pd.DataFrame(mnb_results)

stat.mean(mnb_results['test_accuracy'])

cnb = ComplementNB()

cnb_results = cross_validate(estimator=cnb, X=x, y=y, scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'], cv=kf, return_train_score=True)

pd.DataFrame(mnb_results)

stat.mean(cnb_results['test_accuracy'])
