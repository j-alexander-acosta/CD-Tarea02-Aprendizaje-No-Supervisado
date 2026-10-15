# Contexto 1: Tarea 02 - Aprendizaje No Supervisado
**Fecha de registro:** 2026-10-14

## 1. Instrucciones
* **Revisión del archivo submission.py:** Analizar y completar las 11 funciones requeridas para los algoritmos de aprendizaje no supervisado.
* **Restricciones de bibliotecas:** Utilizar exclusivamente `numpy` y la biblioteca estándar de Python (`collections.deque`, etc.). Queda estrictamente prohibido el uso de `scikit-learn` u otras bibliotecas externas de ML.
* **Preservación de interfaces:** Respetar los nombres, firmas de funciones y tipos de retorno exactos requeridos por el evaluador automático `grader.py`.
* **Inmutabilidad de entradas:** Ninguna función debe modificar los arreglos `X`, `C`, etc. recibidos como argumento.
* **Tolerancia y robustez numérica:** Manejar tipos de datos enteros y de punto flotante adecuadamente, prevenir divisiones por cero (como grupos vacíos en k-means o varianza nula en PCA), y aplicar las convenciones de desempate especificadas.

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Implementar y validar los algoritmos de k-means, inicialización k-means++, índice de Rand, PCA, DBSCAN y selección óptima de hiperparámetros.
* **Fases/Hitos principales:**
  1. **Parte 1 (k-means):** Implementación de `asignar_grupos`, `actualizar_centroides`, `inercia` y el bucle principal `kmeans`.
  2. **Parte 2 (k-means++):** Implementación de `inicializar_kmeanspp` con selección probabilística proporcional a $D(x)^2$.
  3. **Parte 3 (Índice de Rand):** Implementación de `indice_rand` calculando la concordancia de pares sin bucles manuales.
  4. **Parte 4 (PCA):** Implementación de `pca` (covarianza muestral, autovalores/autovectores vía `eigh`, convención de signos), `proyectar` y `reconstruir`.
  5. **Parte 5 (DBSCAN):** Implementación de `dbscan` con cola FIFO para densidad, diferenciando núcleo, borde y ruido (-1).
  6. **Parte 6 (Selección de método):** Definición de `elegir_metodo` con hiperparámetros que superen Rand >= 0.95 en "lunas" y "gaussianos".
  7. **Validación y Edge Cases:** Ejecución de `grader.py` (60/60 pts) y pruebas exhaustivas de casos borde (tipos enteros, listas, dimensiones y degeneraciones).
* **Entregables:**
  * Archivo `submission.py` completado y verificado con puntaje máximo.

## 3. Referencias
* **Path del plan original:** `submission.py`
