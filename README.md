# Kaggle Titanic Challenge
#### [Link a colab](https://colab.research.google.com/drive/1wemBpaP3h8UycMLZgsVMcgkOjS_vItOG?usp=sharing): https://colab.research.google.com/drive/1wemBpaP3h8UycMLZgsVMcgkOjS_vItOG?usp=sharing

## **Equipo 4**:
| <h4>Alumno</h4> | <h4>Matricula</h4> |
| ---|---|
| <h5>Mauricio Juárez Sánchez</h5> | <h5>A01660336</h5> |
| <h5>Alfredo Jeong Hyun Park</h5> | <h5>A01658259</h5> |
| <h5>Fernando Alfonso Arana Salas</h5> | <h5>A01272933</h5> |
| <h5>Miguel Ángel Bustamante Pérez</h5> | <h5>A01781583</h5> |

# Reto Kaggle – Titanic classification

El objetivo de este proyecto es resolver el problema Titanic - Machine Learning from Disaster de Kaggle Competition utilizando algoritmos de clasificación. Para ello, se seguirán las siguientes etapas:

## 1. Exploración y preprocesamiento de los datos:

### a. Distribuciones:

- **Verificar si los datos están balanceados en las clases en las cuales se deben separar los datos.**

Para el conjunto de datos del reto **Titanic** nuestra tarea es predecir si un pasajero sobrevivió o no al naufragio. Es decir, las clases objetivo son "sobrevivió" y "no sobrevivió".<br>
Hemos considerado que los datos **NO** están equilibrados ya que el número de muestras de pasajeros que sobrevivieron es mayor al número de muestras de pasajeros que no sobrevivieron. El **porcentage qué 'sobrevivió' [549 personas] es**:  **61.62 %**. El **porcentage of 'no sobrevivió' [341 personas]** es  **38.38%**. <br>

Está verificación del equilibrio de clases es importante para nuestro análisis de datos y la modelización, ya que un desequilibrio significativo puede llevar a problemas de sesgo en los modelos de machine learning. Si una clase está sobrerepresentada en comparación con la otra, el modelo puede tener dificultades para aprender y predecir la clase minoritaria de manera efectiva. Por lo tanto, fue fundamental **evaluar y abordar el desequilibrio de clases en el preprocesamiento** de datos antes de entrenar cualquier modelo.

- **Analizar la distribución de los datos categóricos y su relación con la clase "survived". Comienza a identificar características relevantes para una clasificación precisa.**

Los atributos categóricos son:

| <h4>Atributo</h4> | <h4>Descripción</h4> |
| ---|---|
| <h6>Name</h6> | <h6>Name of passenger</h6> |
| <h6>Sex</h6> | <h6>Sex</h6> |
| <h6>Ticket</h6> | <h6>Ticket number</h6> |
| <h6>Cabin</h6> | <h6>Cabin number</h6> |
| <h6>Embarked</h6> | <h6>Port of Embarkation => C = Cherbourg, Q = Queenstown, S = Southampton</h6> |

<h4>Atributo 'Sex'</h4><br>
Porcentage of male ( 577 ):  64.76 %<br>
Porcentage of female ( 314 ):  35.24 %<br>
Porcentage of male surviving:  18.89 %<br>
Porcentage of male not surviving:  81.11 %<br>
Porcentage of female surviving:  74.2 %<br>
Porcentage of female not surviving:  25.8 %<br>

![SexRatio](./assets/sex_ratio.png)

<h4>Atributo 'Embarked'</h4><br>
Porcentage of embark in C = Cherbourg ( 168 ): 18.86 %		Not-Survived:  44.64 %		Survived: 55.36 %<br>
Porcentage of embark in Q = Queenstown ( 77 ):  8.64 %		Not-Survived:  61.04 %		Survived: 38.96 %<br>
Porcentage of embark in S = Southampton ( 644 ):  72.28 %	Not-Survived:  66.3 %		Survived: 33.7 %<br>

![EmbarkedRatio](./assets/embarked_ratio.png)

Hemos decidido no utilizar los atributos **'Name'**, **'Ticket'** y **'Cabin'**. Creemos que no aportan una mejora significativa en nuestras predicciones. De hecho, probamos extraer los prefijos *Mr*, *Miss*, *Don*, *Lady*, para analizar si había alguna relación con *'sobrevivir'* o *'no sobrevivir'* pero nos dimos cuenta que dichos prefijos están estrechamente ligados al *'sexo'* de una persona. Es decir, aquellos que tienen prefijos masculinos son hombres y viceversa.<br>


- **Comprender la distribución de los datos numéricos y determinar si es necesario aplicar procesos de normalización o estandarización.**


Los atributos númericos son:

| <h5>Atributo</h5> | <h5>Descripción</h5> |
| ---|---|
| <h6>PassengerId</h6> | <h6>Passenger Id</h6> |
| <h6>Survived</h6> | <h6>Survival	0 = No, 1 = Yes</h6> |
| <h6>Pclass</h6> | <h6>Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd</h6> |
| <h6>Age</h6> | <h6>Age in years</h6> |
| <h6>SibSp</h6> | <h6># of siblings / spouses aboard the Titanic</h6> |
| <h6>Parch</h6> | <h6># of parents / children aboard the Titanic</h6> |
| <h6>Fare</h6> | <h6>Passenger fare</h6> |

<h4>Atributo 'Age'</h4>
Under age ( 139 ):	     0 - 18 years old	Not-Survived:  49.64 %		Survived: 50.36 %<br>
Young adults ( 270 ):	19 - 30 years old	Not-Survived:  64.44 %		Survived: 35.56 %<br>
Adults ( 241 ):		30 - 50 years old	Not-Survived:  57.68 %		Survived: 42.32 %<br>
Elderly ( 64 ):		50+ years old		Not-Survived:  65.62 %		Survived: 34.38 %<br>

![AgeRatio](./assets/age_ratio.png)

### b. Datos faltantes:

- **Identificar y visualizar los datos faltantes.**

Pclass has 0 missing values<br>
Name has 0 missing values<br>
Sex has 0 missing values<br>
Age has 177 missing values<br>
SibSp has 0 missing values<br>
Parch has 0 missing values<br>
Ticket has 0 missing values<br>
Fare has 0 missing values<br>
Cabin has 687 missing values<br>
Embarked has 2 missing values<br>
Survived has 0 missing values<br>

![Faltantes](./assets/datos_faltantes.png)

- **Decidir qué características requieren imputaciones y cuáles no. Justificar la decisión y eliminar las columnas no seleccionadas.**

La decisión de qué características requieren imputaciones la tomamos bajo los siguientes criterios:<br>
1. **Age**: Tiene 177 valores faltantes. Dado que la edad podría ser una característica importante para predecir la supervivencia en el contexto del Titanic (por ejemplo, es probable que los niños tengan tasas de supervivencia diferentes a los adultos), creímos razonable imputar estos valores faltantes.

2. **Cabin**: Tiene 687 valores faltantes. Dado que la cantidad de valores faltantes es considerablemente alta, hemos decidido eliminar esta característica en su totalidad. Además, creemos que la información de la *cabina* puede no ser esencial para predecir la supervivencia.

3. **Embarked**: Tiene 2 valores faltantes. Dado que solo hay dos valores faltantes, decidimos imputarlos utilizando la moda (el valor más común) de la característica "Embarked".
<br>

- **Aplicar técnicas de imputación para los datos faltantes. Seleccionar la mejor técnica y justificar la elección.**

Usamos la técnica incluída en scikit learn para realizar imputaciones.
<br><br>
```from sklearn.impute import SimpleImputer```
<br><br>
Para los valores numéricos usamos:
<br><br>
```imputer_num = SimpleImputer(strategy="median")```
<br><br>
Para los valores categóricos usamos:
<br><br>
```"imputer_cat = SimpleImputer(strategy="most_frequent")```
<br><br>
Usamos la mediana porque es una medida de tendencia central robusta que no se ve afectada por valores extremos o atípicos. En el caso de "Age", podría haber valores atípicos que afecten la media, es por eso que la mediana fue nuestra elección más robusta para imputar los valores faltantes.

Observamos que la distribución de "Age" es asimétrica pero *NO* contiene valores atípicos significativos. Aun así, creímos preferible utilizar la mediana para evitar la influencia de estos valores atípicos en la imputación. Además, dicha asimetría no es muy significativa, dado que la desviación estándar es de 14. A continuación presentamos una gráfica que refleja la distribución de esta categoria numérica:

![AgeDistribution](./assets/age_dist.png)

Por otro lado, para la categoría 'Embarked' únicamente imputamos 2 valores. Decidimos usar la estrategia de reemplazo por el valor **más frecuente** también conocida como la **moda**, está decisión se tomó para seguir la tendencia central y dar una distribución concisa a la categoría.<br>

### c. Análisis de correlación:

- **Realizar un análisis de correlación para decidir qué características deben mantenerse y cuáles descartarse.**

El resultado del análisis de correlación dio la siguiente figura:

![Correlacion](./assets/corr.png)

Con base en esta figura vemos que el rango de valores va de -1 a 1. *-1* indica que hay una fuerte correlación **negativa**. *0* nos indica que no hay correlación. *1* nos indica que hay una fuerte correlación **positiva**.

1. **Survived**:
   - Su correlación con otros atributos es lógicamente de gran interes: 
     - Hay correlación positiva con Fare (0.247519).
     - Hay correlación negativa con Pclass (-0.268318) y Sex (-0.509660).

2. **Pclass**:
   - Su correlación con 'Fare' y 'Age' fue como sigue:
     - Correlación negativa con Fare (-0.594245).
     - Correlación negativa con Age (-0.304298).
   - El hecho de que esté correlacionada con otras 2 clases / atributos, nos pareció como un atributo robusto para las predicciones.

3. **Sex**:
   - Tuvo una fuerte correlación negativa con 'Sobrevivió' (-0.509660), asi que la mantuvimos pues creímos que es un predictor significativo.

4. **Age**:
   - Tiene una correlación negativa con Pclass (-0.304298). Decidimos agrupar Age en categorías para mejorar la distribución y asemejarla más a una campana gaussiana. Sin embargo, vemos que su correlación con otros atributos no es tan fuerte como lo hubieramos pensado en un inicio. No es un atributo que tenga gran significado en las predicciones.

5. **SibSp**:
   - Tiene una correlación positiva con Parch (0.392197). Este atributo decidimos combinarlo con Parch para la creación de un nuevo feature llamado 'Travel Alone'. El objetivo fue identificar a los pasajeros que viajaron junto a su familia y aquellos pasajeros que viajaron solitariamente. Creemos que este factor puede ser un buen predictor de supervivencia. 

6. **Parch**:
   - Mismo caso que SibSp.

7. **Fare**:
   - *Fare* es un atributo interesante porque tiene correlaciones negativas fuertes con 'Pclass' (-0.594245) y correlaciones positivas con 'Survived' (0.247519) y moderadamente correlacionada con otros atributos.

8. **Embarked**:
   - No tiene correlaciones fuertes pero decidimos quedarnoslo como una versión de prueba. Sin embargo, también optamos por eliminarlo para probar nuestros modelos en una segunda iteración.

### d. Transformación de datos:

- **Convertir los datos categóricos en numéricos. Explorar diferentes métodos y seleccionar el más adecuado. Justificar la elección.**

Para la transformación de los datos usamos el siguiente codificador:
<br><br>
```from sklearn.preprocessing import OneHotEncoder```
<br><br>
```onehot = OneHotEncoder(handle_unknown="ignore")```
<br><br>
La elección de codificación estuvo entre **OneHotEncoder** y **Codificación por enteros** (por ejemplo, [0, 1, 2]) para variables categóricas. Tomamos nuestra decisión basándonos en los modelos que ibamos a usar, concluímos que la mejor estrategia sería usar *OneHotEncoder*. Además, sabemos que cuando aplicamos una transformación, la regla que podemos usar cuando no hay ningún parámetro confiable en qué basarnos, es hacer una transformación de One-Hot en vez de *Codificación por enteros*.

**1. Regresión Logística:** 

   - **Elección de Codificación:** Usualmente, la codificación one-hot es una mejor opción para la regresión logística. Esto se debe a que la regresión logística no asume ninguna relación ordinal inherente entre las categorías, y la codificación one-hot garantiza que cada categoría se trate como una característica binaria separada.

**2. Clasificador de Bosque Aleatorio (Random Forest):**

   - **Elección de Codificación:** Los bosques aleatorios pueden funcionar bien tanto con codificación one-hot como con codificación entera. Los bosques aleatorios son robustos y pueden manejar la codificación ordinal si hay un orden significativo. Sin embargo, si deseas evitar posibles interpretaciones erróneas de la ordinalidad, la codificación one-hot es una elección más segura. A menudo es más fácil comenzar con la codificación one-hot y ver si funciona bien para tu conjunto de datos específico.

**3. Clasificador de Vecinos Más Cercanos (KNN):**

   - **Elección de Codificación:** KNN es un algoritmo basado en la distancia y puede ser sensible a la elección de la codificación. En la mayoría de los casos, se recomienda la codificación one-hot para KNN porque trata cada categoría como una dimensión separada. Sin embargo, si tienes datos ordinales y estás seguro de la significatividad del orden, podrías experimentar con la codificación entera.

**4. Clasificador de Máquinas de Soporte Vectorial (SVM):**

   - **Elección de Codificación:** Las SVM pueden trabajar con codificación one-hot y codificación entera. Depende del kernel que elijas y la naturaleza de tus datos. Los kernels lineales suelen funcionar bien con la codificación one-hot, mientras que las funciones de kernel como la Función de Base Radial (RBF) pueden manejar ambos tipos de codificación.

En resumen, nosotros usamos codificación one-hot por ser la opción más segura y, como resultado de nuestra investigación, vimos que es la más ampliamente utilizada para la mayoría de los algoritmos de aprendizaje automático, ya que garantiza que cada categoría se trate como una característica separada, dejando claro que no existe una relación ordinal inherente.

## 2. Clasificación:

### a. Selección de clasificadores:

- **Elegir tres algoritmos de clasificación que se utilizarán en el proyecto. Justificar la selección de cada algoritmo.**

Usamos los siguientes 4 modelos de clasificación:

| <h4>Modelo</h4> | <h4>Breve descripción</h4> |
| ---|---|
| <h6>Logistic Regression</h6> | <h6>Fue una elección sólida por el hecho de ser un problema de clasificación binaria. Nos proporcionó una interpretación sencilla de las relaciones entre las características y la variable objetivo. Fue útil para comprender mejor los factores que influyen en la supervivencia.</h6> |
| <h6>K Nearest Neighbors</h6> | <h6>Fue una sorpresa que los datos no presentaran correlaciones fuertes en relación a la supervivencia. Es por ello que KNN fue efectivo cuando no se asumimos ninguna distribución específica en los datos. Nos ayudó a capturar relaciones no lineales. Además fue útil para observar si existen agrupamientos de pasajeros que sobrevivieron o no en el Titanic.</h6> |
| <h6>Support Vector Machine</h6> | <h6>Fue adecuado para separar clases en espacios de alta dimensión y también resultó el más eficaz en la clasificación binaria. Demostró ser capaz de manejar de manera efectiva características no lineales. Además, como vimos, fue útil ya que existían fronteras de decisión complejas entre las clases.</h6> |
| <h6>Random Forest Classifier</h6> | <h6>Es robusto y versátil. Nos ayudó a manejar características irrelevantes, manejar conjuntos de datos desequilibrados y nos proporcionó una buena estimación de la importancia de las características. Además, tiende a reducir el sobreajuste.</h6> |

### b. Train-test-validate split:

- **Utilizar k-cross validation para realizar la clasificación. Seleccionar el valor de "k" y justificar la elección.**

La elección del valor de "k" en k-cross validation es un proceso importante ya que influyó en la evaluación del rendimiento de nuestros modelos. La justificación para seleccionar un valor particular de "k" dependió de varios factores, incluidos el tamaño del dataset y nuestros objetivos.<br>

   - Dado que tenemos un conjunto de datos pequeño, usamos un valor alto de "k" para aprovechar al máximo nuestros datos y obtener una estimación más estable del rendimiento.
        - Esto significa que estamos promediando el rendimiento del modelo sobre más particiones diferentes de nuestros datos. Como resultado, las estimaciones de rendimiento fueron más estables, es decir, menos sensibles a pequeñas variaciones en los datos de entrenamiento y prueba.
   - Dado lo anterior, estuvimos dispuestos a obtener la estimación más precisa del rendimiento de nuestros modelos. Es por eso que seleccionamos k=15. Aún cuando esto implico sacrificar un poco de velocidad de cálculo.
   - Dado el punto anterior, un valor más alto de "k" significa que realizamos más divisiones del conjunto de datos y, por lo tanto, requerimos más tiempo de cómputo.
   - Sabemos que **k** afecta nuestras estimaciones de rendimiento. Aunque el valor "k = 15 [considerado alto]" nos proporcionó estimaciones de rendimiento más estables, esto fue expensas de una mayor varianza.
        - Esto quiere decir que este aumento en la estabilidad tiene un costo potencial. Cuanto mayor era nuestro valor de "k," más pequeños fueron los conjuntos de entrenamiento en cada iteración de la validación cruzada. Con conjuntos de entrenamiento más pequeños, es posible que los modelo no capturaran todas las variaciones y patrones presentes en los datos. Esto llevó a una mayor variabilidad en las estimaciones de rendimiento **[como se mostrará a continuación]**: esto significa que las métricas varian más entre las diferentes divisiones de los datos.
<br>

En resumen, concluímos que es importante entender que *existe un equilibrio entre la estabilidad y la capacidad de generalización de nuestros modelo*.

<br>

| <h4>K-Cross Validation p/modelo</h4> | <h4>Resultados</h4> |
| ---|---|
| <h5>Logistic Regression</h5> | Cross-validation scores: [*0.75609756*, *0.58536585*, *0.775*, *0.725*, *0.825*, *0.625*, *0.775*, *0.775*, *0.825*, *0.875*, *0.725*, *0.75*, *0.825*, *0.825*, *0.775*]<br><br>Logistic Regression Mean Accuracy: **0.76** |
| <h5>K Nearest Neighbors</h5> | KNN Cross-validation scores [*0.70731707*, *0.6097561*, *0.775*, *0.775*, *0.825*, *0.775*, *0.75*, *0.775*, *0.8*, *0.75*, *0.775*, *0.825*, *0.8*, *0.8*, *0.725*]<br><br>KNN Mean Accuracy: **0.76** |
| <h5>Support Vector Machine</h5> | Cross validation scores [*0.70731707*, *0.65853659*, *0.75*, *0.8*, *0.825*, *0.7*, *0.85*, *0.8*, *0.775*, *0.85*, *0.75*, *0.825*, *0.825*, *0.825*, *0.775*]<br><br>SVM Mean Accuracy: **0.78** |
| <h5>Random Forest Classifier</h5> | Cross validation scores Random Forest: [*0.70731707*, *0.65853659*, *0.825*, *0.85*, *0.775*, *0.75*, *0.775*, *0.725*, *0.85*, *0.85*, *0.825*, *0.9*, *0.775*, *0.85*, *0.8*]<br><br>Random Forest Mean accuracy **0.79** |


### c. Métricas de evaluación:

- **Calcular la exactitud, precisión, matriz de confusión, curva ROC y AUC. Explicar cada una de estas métricas.**

<h4>Logistic Regression</h4>

| <h4>Métrica</h4> | <h4>Resultado </h4> | <h4>Explicación</h4> |
| ---|---|---|
| <h5>Precisión</h5> | <h4>0 -> 0.83</h4><br><h4>1 -> 0.77</h4> | <h6>La precisión mide la proporción de predicciones correctas en relación con el total de predicciones para cada clase. En este caso, el modelo tiene una alta precisión para la clase 0 (No sobrevivió) y una precisión ligeramente menor para la clase 1 (Sobrevivió). Esto significa que cuando el modelo predice que un pasajero no sobrevivió, acierta aproximadamente el 83% de las veces, y cuando predice que un pasajero sobrevivió, acierta aproximadamente el 77% de las veces.</h6> |
| <h5>Recall</h5> | <h4>0 -> 0.87</h4><br><h4>1 -> 0.71</h4> | <h6>El recall mide la proporción de instancias positivas (verdaderos positivos) que el modelo predijo correctamente en relación con el total de instancias positivas en los datos reales. En este caso, el modelo tiene un recall alto para la clase 0 (No sobrevivió), lo que significa que detecta aproximadamente el 87% de los casos en los que un pasajero no sobrevivió [True negatives]. Sin embargo, el recall para la clase 1 (Sobrevivió) es más bajo, lo que indica que el modelo pierde aproximadamente el 29% de los casos en los que un pasajero sobrevivió.</h6> |
| <h5>F1-Score</h5> | <h4>0 -> 0.85</h4><br><h4>1 -> 0.74</h4> | <h6>El F1-Score es una métrica que combina precisión y recall en una sola medida. Es útil cuando se busca un equilibrio entre la precisión y la capacidad del modelo para detectar correctamente los casos positivos. En este caso, el F1-Score para la clase 0 es alto, lo que indica un buen equilibrio entre precisión y recall para la clase 0. Para la clase 1, el F1-Score es un poco más bajo, lo que sugiere que podría haber margen para mejorar el equilibrio entre precisión y recall en esta clase.</h6> |
| <h5>Accuracy</h5> | <h4>0.81</h4> | <h6>La exactitud mide la proporción de predicciones correctas en relación con el total de predicciones en general. En este caso, el modelo tiene una exactitud global del 81%, lo que significa que acierta aproximadamente el 81% de las predicciones en todo el conjunto de datos. Sin embargo, la exactitud por sí sola sabemos que es engañosa ya que si no hubieramos equilibrado nuestras clases, el modelo podría predecir siempre la clase mayoritaria con mayor exactitud.</h6> |

<h4>K Nearest Neighbors</h4>

| <h4>Métrica</h4> | <h4>Resultado </h4> | <h4>Explicación</h4> |
| ---|---|---|
| <h5>Precisión</h5> | <h4>0 -> 0.83</h4><br><h4>1 -> 0.84</h4> | <h6>El modelo de KNN tiene una precisión similar para ambas clases, alrededor del 83% para la clase 0 y el 84% para la clase 1. Esto significa que cuando el modelo predice que un pasajero no sobrevivió, acierta aproximadamente el 83% de las veces, y cuando predice que un pasajero sobrevivió, acierta aproximadamente el 84% de las veces.</h6> |
| <h5>Recall</h5> | <h4>0 -> 0.92</h4><br><h4>1 -> 0.69</h4> | <h6>El modelo tiene un recall alto para la clase 0 (No sobrevivió), lo que significa que detecta aproximadamente el 92% de los casos en los que un pasajero no sobrevivió. Sin embargo, el recall para la clase 1 (Sobrevivió) es más bajo, alrededor del 69%.</h6> |
| <h5>F1-Score</h5> | <h4>0 -> 0.87</h4><br><h4>1 -> 0.76</h4> | <h6>El F1-Score es más alto para la clase 0 que para la clase 1, lo que sugiere un mejor equilibrio entre precisión y capacidad del modelo para detectar correctamente los casos positivos en la clase 0. Es decir, mejor acierta correctamente y es sensible ante la clase 'no-sobrevivió'.</h6> |
| <h5>Accuracy</h5> | <h4>0.83</h4> | <h6>El modelo de KNN tiene una exactitud global del 83%, lo que significa que acierta aproximadamente el 83% de las predicciones en todo el conjunto de datos.</h6> |

<h4>Support Vector Machine</h4>

| <h4>Métrica</h4> | <h4>Resultado </h4> | <h4>Explicación</h4> |
| ---|---|---|
| <h5>Precisión</h5> | <h4>0 -> 0.84</h4><br><h4>1 -> 0.82</h4> | <h6>El modelo de SVM tiene una precisión similar para ambas clases, alrededor del 84% para la clase 0 y el 82% para la clase 1. Esto significa que cuando el modelo predice que un pasajero no sobrevivió, acierta aproximadamente el 84% de las veces, y cuando predice que un pasajero sobrevivió, acierta aproximadamente el 82% de las veces.</h6> |
| <h5>Recall</h5> | <h4>0 -> 0.90</h4><br><h4>1 -> 0.73</h4> | <h6>El modelo tiene un recall alto para la clase 0 (No sobrevivió), lo que significa que detecta aproximadamente el 90% de los casos en los que un pasajero no sobrevivió. Sin embargo, el recall para la clase 1 (Sobrevivió) es más bajo, alrededor del 73%.</h6> |
| <h5>F1-Score</h5> | <h4>0 -> 0.87</h4><br><h4>1 -> 0.77</h4> | <h6>El F1-Score es más alto para la clase 0 que para la clase 1, lo que sugiere un mejor equilibrio entre precisión y capacidad del modelo para detectar correctamente los casos positivos en la clase 0.</h6> |
| <h5>Accuracy</h5> | <h4>0.84</h4> | <h6>En este caso, el modelo de SVM tiene una exactitud global del 84%, lo que significa que acierta aproximadamente el 84% de las predicciones en todo el conjunto de datos.</h6> |

<h4>Random Forest Classifier</h4>

| <h4>Métrica</h4> | <h4>Resultado </h4> | <h4>Explicación</h4> |
| ---|---|---|
| <h5>Precisión</h5> | <h4>0 -> 0.94</h4><br><h4>1 -> 0.95</h4> | <h6>El modelo de Random Forest tiene una alta precisión tanto para la clase 0 (No sobrevivió) como para la clase 1 (Sobrevivió). Esto significa que cuando el modelo predice que un pasajero no sobrevivió, acierta aproximadamente el 94% de las veces, y cuando predice que un pasajero sobrevivió, acierta aproximadamente el 95% de las veces.</h6> |
| <h5>Recall</h5> | <h4>0 -> 0.97</h4><br><h4>1 -> 0.90</h4> | <h6>En este caso, el modelo tiene un recall excepcionalmente alto para la clase 0 (No sobrevivió), lo que significa que detecta aproximadamente el 97% de los casos en los que un pasajero no sobrevivió. El recall para la clase 1 (Sobrevivió) también es alto, alrededor del 90%.</h6> |
| <h5>F1-Score</h5> | <h4>0 -> 0.95</h4><br><h4>1 -> 0.92</h4> | <h6>El F1-Score es alto tanto para la clase 0 como para la clase 1, lo que sugiere un excelente equilibrio entre precisión y capacidad del modelo para detectar correctamente los casos positivos.</h6> |
| <h5>Accuracy</h5> | <h4>0.94</h4> | <h6>El modelo de Random Forest tiene una alta exactitud global del 94%, lo que significa que acierta aproximadamente el 94% de las predicciones en todo el conjunto de datos.</h6> |

- **Con base en estas métricas, determinar el mejor clasificador y justificar la elección.**

El mejor clasificador fue: **A**

![Titanic](./assets/titanic.png)
