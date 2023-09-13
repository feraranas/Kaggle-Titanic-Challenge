# Kaggle Titanic Challenge
## **Equipo 4**:
| <h2>Alumno</h2> | <h2>Matricula</h2> |
| ---|---|
| <font size = 4 color ='336EFF'>Mauricio Juárez Sánchez</font> | <h1>A01660336</h1> |
| <font size = 4 color ='556E00'>Alfredo Jeong Hyun Park</font> | <h1>A01658259</h1> |
| <font size = 4 color ='00FF00'>Fernando Alfonso Arana Salas</font> | <h1>A01272933</h1> |
| <font size = 4 color ='3300FF'>Miguel Ángel Bustamante Pérez</font> | <h1>A01781583</h1> |

# Reto Kaggle – Titanic classification

El objetivo de este proyecto es resolver el problema Titanic - Machine Learning from Disaster de Kaggle Competition utilizando algoritmos de clasificación. Para ello, se seguirán las siguientes etapas:

## 1. Exploración y preprocesamiento de los datos:

### a. Distribuciones:

- <font size = 3 color ='336EFF'>Verificar si los datos están balanceados en las clases en las cuales se deben separar los datos.</font>

En conjunto de datos 'train' del reto Titanic, la tarea es predecir si un pasajero sobrevivió o no al naufragio. Es decir, las clases objetivo son "sobrevivió" y "no sobrevivió". El conjunto de datos no se considera equilibrado ya que el número de muestras de pasajeros que sobrevivieron es mayor al número de muestras de pasajeros que no sobrevivieron. El **porcentage qué 'sobrevivió' [549 personas] es**:  **61.62 %**. El **porcentage of 'no sobrevivió' [341 personas] ** es  **38.38%**. 

Está verificación del equilibrio de clases es importante en el análisis de datos y la modelización, ya que un desequilibrio significativo puede llevar a problemas de sesgo en los modelos de machine learning. Si una clase está sobrerepresentada en comparación con la otra, el modelo puede tener dificultades para aprender y predecir la clase minoritaria de manera efectiva. Por lo tanto, fue fundamental **evaluar y abordar el desequilibrio de clases en el preprocesamiento** de datos antes de entrenar cualquier modelo.

- <font size = 3 color ='336EFF'>Analizar la distribución de los datos categóricos y su relación con la clase "survived". Comienza a identificar características relevantes para una clasificación precisa.</font>

- <font size = 3 color ='336EFF'>Comprender la distribución de los datos numéricos y determinar si es necesario aplicar procesos de normalización o estandarización.</font>

### b. Datos faltantes:

- <font size = 3 color ='336EFF'>Identificar y visualizar los datos faltantes.</font>
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

- <font size = 3 color ='336EFF'>Decidir qué características requieren imputaciones y cuáles no. Justificar la decisión y eliminar las columnas no seleccionadas.</font>

- <font size = 3 color ='336EFF'>Aplicar técnicas de imputación para los datos faltantes. Seleccionar la mejor técnica y justificar la elección.</font>

### c. Análisis de correlación:

- Realizar un análisis de correlación para decidir qué características deben mantenerse y cuáles descartarse.

### d. Transformación de datos:

- Convertir los datos categóricos en numéricos. Explorar diferentes métodos y seleccionar el más adecuado. Justificar la elección.

## 2. Clasificación:

### a. Selección de clasificadores:

- Elegir tres algoritmos de clasificación que se utilizarán en el proyecto. Justificar la selección de cada algoritmo.

### b. Train-test-validate split:

- Utilizar k-cross validation para realizar la clasificación. Seleccionar el valor de "k" y justificar la elección.

### c. Métricas de evaluación:

- Calcular la exactitud, precisión, matriz de confusión, curva ROC y AUC. Explicar cada una de estas métricas.
- Con base en estas métricas, determinar el mejor clasificador y justificar la elección.


![Titanic](./assets/titanic.png)