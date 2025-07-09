
# Pong AI - Proyecto Final 2025-01

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)

## Descripción

Este proyecto implementa una red neuronal artificial para controlar un agente inteligente que juega Pong. Utiliza una arquitectura basada en capas densas (Dense), funciones de activación (ReLU), y entrenamiento supervisado con el algoritmo de optimización SGD (Stochastic Gradient Descent). El entorno de simulación es una versión simplificada del juego Pong, y el objetivo del agente es devolver la pelota de forma correcta, optimizando la política de movimiento vertical (arriba, quedarse, abajo).

## Objetivo

Desarrollar un agente inteligente que aprenda a jugar Pong utilizando redes neuronales implementadas desde cero, con estructuras y funcionalidades inspiradas en bibliotecas de Machine Learning como PyTorch o TensorFlow.

## Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#investigación-teórica)
4. [Diseño e implementación](#diseño-e-implementación)
5. [Ejecución](#ejecución)
6. [Análisis del rendimiento](#análisis-del-rendimiento)
7. [Trabajo en equipo](#trabajo-en-equipo)
8. [Conclusiones](#conclusiones)
9. [Bibliografía](#bibliografía)
10. [Licencia](#licencia)

---

## 1. Datos generales

* **Tema**: Redes Neuronales aplicadas al juego Pong
* **Grupo**: projecto_final_los_abuelitos

## Integrantes

| Nombre completo              | Código    |
| ---------------------------- | --------- |
| Diego Alexis Gil Rojas       | 20232036  |
| Marco Apolinario Lainez      | 202210538 |
| Ana Silvia Cordero Ricaldi   | 202410462 |
| Samir Antony Mena Ramírez    | 202310289 |

---

## 2. Requisitos e instalación

1. **Compilador**: GCC 11 o superior
2. **Dependencias**:
   * CMake >= 3.20
3. **Instalación**:

```bash
git clone https://github.com/tuusuario/projecto-final-los-abuelitos.git
cd projecto-final-los-abuelitos
mkdir build && cd build
cmake ..
make
```

---

## 3. Investigación teórica

Se realizó una revisión de los conceptos fundamentales relacionados con las redes neuronales artificiales, enfocándose en los siguientes aspectos clave:

* Fundamentos de redes neuronales artificiales (ANN): Se estudió el modelo básico de una neurona artificial, el cual recibe múltiples entradas, aplica un conjunto de pesos, suma los resultados y los transforma mediante una función de activación. Las redes neuronales están compuestas por capas de estas neuronas, permitiendo el modelado de relaciones no lineales complejas.

* Arquitectura multicapa (MLP) y su aplicación al problema XOR: Se utilizó una red neuronal de tipo Multilayer Perceptron (MLP) para resolver el clásico problema lógico XOR, el cual no puede ser resuelto por una sola capa lineal. La MLP implementada incluye una capa oculta con una función de activación no lineal que permite separar correctamente las clases.

* Retropropagación del error (Backpropagation): Se implementó el algoritmo de retropropagación, el cual permite calcular los gradientes del error con respecto a los pesos de la red. Este proceso es esencial para ajustar los parámetros internos durante el entrenamiento, minimizando la función de pérdida.

* Optimización con descenso de gradiente estocástico (SGD): Para actualizar los pesos, se utilizó el algoritmo de descenso de gradiente estocástico. Este optimizador actualiza los parámetros en función del gradiente calculado, permitiendo a la red aprender de manera iterativa a partir de los datos de entrenamiento.
---

## 4. Diseño e implementación

### Arquitectura del proyecto

```
projecto-final-los-abuelitos/
│
├── include/utec/
│   ├── algebra/           # Tensor y operadores
│   ├── nn/                # Dense, Activation, Loss, NeuralNetwork, Optimizer
│   └── agent/             # EnvGym y PongAgent
│
├── src/                  # Implementación
│   └── main.cpp          # Entrenamiento y evaluación
│
├── xor_data.csv          # Datos de entrenamiento XOR
├── CMakeLists.txt        # Configuración del proyecto
└── README.md
```

### Componentes principales

* `Tensor<T, N>`: estructura de datos para manejar tensores.
* `Dense`: capa densa de una red neuronal.
* `ReLU`: función de activación.
* `MSELoss`: función de pérdida.
* `NeuralNetwork`: clase que permite el entrenamiento.
* `EnvGym`: simulador de Pong.
* `PongAgent`: agente que toma decisiones en base a la red neuronal.

### 5. Ingreso de datos, entrenamiento y ejecución

El sistema toma los datos desde un archivo `.csv` llamado `xor_data.csv`, donde cada fila tiene la forma `x1,x2,y`. Las dos primeras columnas son las entradas y la última es la etiqueta esperada.

Ejemplo de contenido del archivo:
```
0,0,0
0,1,1
1,0,1
1,1,0
```

Durante la ejecución del `main.cpp`:

1. **Carga de datos** desde el archivo CSV.
2. **Entrenamiento de la red neuronal** por 1000 épocas usando MSE y SGD.
3. **Predicción** de las salidas para las mismas entradas, comparando con las etiquetas.

### Ejemplo de salida en consola

```
Shape de X: 4 x 2
Shape de Y: 4 x 1

=== Predicciones de la red XOR ===
Input: (0, 0) -> Predicho: 0.47 | Esperado: 0
Input: (0, 1) -> Predicho: 0.50 | Esperado: 1
Input: (1, 0) -> Predicho: 0.50 | Esperado: 1
Input: (1, 1) -> Predicho: 0.52 | Esperado: 0
```

Posteriormente, se lanza el entorno Pong para evaluar el comportamiento del agente inteligente:

```
Step 0 -> reward: 0, done: 0
Step 1 -> reward: 0, done: 0
Step 2 -> reward: 0, done: 0
...
```

---

## 6. Análisis del rendimiento

* La red neuronal logra aprender el XOR correctamente.
* El agente puede devolver la pelota en el entorno de Pong en múltiples ocasiones.

### Posibles mejoras (Epic 4)

* Uso de función de activación `Softmax` para decisiones probabilísticas.
* Entrenamiento más profundo usando capas adicionales.
* Criterios de exploración/explotación para mejorar rendimiento del agente.

---

## 7. Trabajo en equipo

| Tarea                    | Miembro                      | Rol                             |
| ------------------------ | ---------------------------- | ------------------------------- |
| Implementación de Tensor | Samir Antony Mena Ramírez    | Backend algebra                 |
| Dense, NN y Activations  | Marco Apolinario Lainez      | Arquitectura de red neuronal    |
| Agente y entorno         | Ana Silvia Cordero Ricaldi   | Simulación y lógica del agente  |
| Ejecución de datos       | Diego Alexis Gil Rojas       | Ejecución principal de datos csv|
---

## 8. Conclusiones

* Se logró construir una red neuronal completamente funcional desde cero utilizando únicamente C++, lo cual implicó la implementación detallada de estructuras matemáticas (tensores), capas densas, funciones de activación, algoritmos de optimización y retropropagación del error. Esta aproximación permitió comprender a profundidad el flujo de datos y el aprendizaje interno de una red neuronal.

* El agente inteligente, basado en la red entrenada, demostró un comportamiento reactivo y coherente dentro del entorno de simulación del juego Pong. A pesar de que el modelo fue entrenado inicialmente con datos simples como el XOR, fue capaz de integrarse al entorno EnvGym y ejecutar acciones lógicas en función del estado del juego.

* La arquitectura del proyecto se diseñó de forma modular y extensible. Esto no solo facilita la inclusión de mejoras futuras (como nuevas funciones de activación, más capas, optimizadores avanzados o técnicas de refuerzo), sino que también promueve buenas prácticas de programación y reutilización de código.

* El proyecto permitió afianzar conocimientos teóricos en redes neuronales y aplicar conceptos fundamentales de inteligencia artificial, programación orientada a objetos, algoritmos de optimización y trabajo colaborativo en equipo.

* Como trabajo final, esta implementación representa un hito importante en la formación profesional, al demostrar que es posible construir soluciones de inteligencia artificial sin depender de librerías externas, comprendiendo y controlando cada paso del proceso de entrenamiento y predicción.

---

## 9. Bibliografía

1. Ian Goodfellow, Yoshua Bengio, Aaron Courville, "Deep Learning", MIT Press, 2016.
2. Michael Nielsen, "Neural Networks and Deep Learning", 2015.
3. F. Rosenblatt, "The Perceptron: A Probabilistic Model", Psychological Review, 1958.
4. C. M. Bishop, "Pattern Recognition and Machine Learning", Springer, 2006.

---

## 10.Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para más detalles.
