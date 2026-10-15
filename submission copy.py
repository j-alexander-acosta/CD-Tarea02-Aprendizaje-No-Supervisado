#!/usr/bin/env python3
"""Tarea: Aprendizaje no Supervisado. Este es el único archivo que completas.

Convenciones de toda la tarea (las mismas del enunciado):

* ``X`` es un arreglo de numpy de forma ``(n, d)``: una fila por observación.
* ``C`` es un arreglo de centroides de forma ``(k, d)``: una fila por grupo.
* Una **partición** es un arreglo de enteros de forma ``(n,)``: la posición
  ``i`` dice a qué grupo pertenece la observación ``i``. Los grupos se numeran
  desde 0. En DBSCAN, y solo ahí, la etiqueta ``-1`` significa ruido.
* Todas las distancias son euclidianas. Cuando basta comparar distancias
  conviene trabajar con la distancia **al cuadrado** y ahorrarse la raíz.
* Ninguna función modifica los arreglos que recibe.

Solo se permite numpy y la biblioteca estándar. No uses scikit-learn ni ninguna
otra biblioteca de aprendizaje: la gracia de esta tarea es escribir los cuatro
algoritmos a mano. No cambies las firmas: el corrector las llama tal como
están escritas.
"""
from __future__ import annotations

from collections import deque

import numpy as np


# ----------------------------------------------------------------------
# Parte 1: k-means
# ----------------------------------------------------------------------
def asignar_grupos(X, C):
    """Asigna cada observación al centroide más cercano.

    Devuelve un arreglo de enteros de forma ``(n,)`` con el índice del
    centroide más cercano a cada fila de ``X``. Si dos centroides quedan a la
    misma distancia, gana el de **índice menor**.

    Ejemplo::

        >>> asignar_grupos(np.array([[0.0, 0.0], [5.0, 5.0]]),
        ...                np.array([[0.0, 0.0], [6.0, 6.0]]))
        array([0, 1])
    """
    # BEGIN_TU_CODIGO
    raise NotImplementedError("Completa esta parte")
    # END_TU_CODIGO


def actualizar_centroides(X, etiquetas, k, C_previo):
    """Recalcula los ``k`` centroides como la media de su grupo.

    Devuelve un arreglo **nuevo** de forma ``(k, d)``. El centroide del grupo
    ``j`` es el promedio de las filas de ``X`` asignadas a ``j``.

    Un grupo puede quedar **vacío** (ninguna observación lo eligió). En ese
    caso el promedio no existe y la convención de esta tarea es conservar el
    centroide anterior, ``C_previo[j]``: así el algoritmo sigue trabajando con
    ``k`` centroides y no se cae con una división por cero.
    """
    # BEGIN_TU_CODIGO
    raise NotImplementedError("Completa esta parte")
    # END_TU_CODIGO


def inercia(X, C, etiquetas):
    """Inercia de la partición: la función que k-means minimiza.

    Es la suma, sobre todas las observaciones, de la distancia **al cuadrado**
    entre la observación y el centroide que tiene asignado::

        inercia = sum_i || x_i - c_{etiquetas[i]} ||^2

    Devuelve un ``float``. Es un número que solo baja: cada paso de k-means
    (reasignar y luego promediar) lo deja igual o más pequeño.
    """
    # BEGIN_TU_CODIGO
    raise NotImplementedError("Completa esta parte")
    # END_TU_CODIGO


def kmeans(X, k, C0, max_iter=100):
    """Corre k-means completo desde los centroides iniciales ``C0``.

    Cada iteración hace, en este orden:

    1. asignar cada observación a su centroide más cercano;
    2. si las asignaciones son **idénticas** a las de la iteración anterior,
       detenerse: el algoritmo convergió y nada más va a cambiar;
    3. si no, recalcular los centroides y seguir.

    La iteración en que se detecta la convergencia **sí se cuenta**. Con esta
    convención un conjunto que ya está en su solución termina en dos
    iteraciones: la primera asigna y la segunda comprueba que nada cambió.

    ``C0`` nunca se modifica: se trabaja sobre una copia.

    Devuelve un diccionario con exactamente estas cinco claves:

    ==================  ==================================================
    ``"centroides"``    arreglo ``(k, d)`` con los centroides finales
    ``"etiquetas"``     arreglo ``(n,)`` con la partición final
    ``"inercia"``       ``float``: la inercia de esa partición
    ``"iteraciones"``   entero: iteraciones recorridas, incluida la última
    ``"convergio"``     ``bool``: ``True`` solo si las asignaciones se
                        repitieron antes de agotar ``max_iter``
    ==================  ==================================================
    """
    # BEGIN_TU_CODIGO
    raise NotImplementedError("Completa esta parte")
    # END_TU_CODIGO


# ----------------------------------------------------------------------
# Parte 2: la inicialización k-means++
# ----------------------------------------------------------------------
def inicializar_kmeanspp(X, k, semilla):
    """Elige ``k`` centroides iniciales con el criterio de k-means++.

    La idea es sembrar los centroides lejos unos de otros, en vez de sortear
    ``k`` filas al azar: el primero se sortea uniformemente y cada siguiente
    se sortea con probabilidad proporcional a ``D(x)^2``, la distancia al
    cuadrado de ``x`` al centroide más cercano de los ya elegidos. Un punto
    que quedó lejos de todo lo elegido tiene mucha probabilidad de ser el
    próximo centroide; uno que ya está bien cubierto, casi ninguna.

    Para que el resultado sea reproducible, el sorteo se hace **exactamente**
    así::

        rng = np.random.default_rng(semilla)
        primero = int(rng.integers(len(X)))            # centroide 0
        ...                                            # centroides 1..k-1
        siguiente = int(rng.choice(len(X), p=d2 / d2.sum()))

    donde ``d2[i]`` es la distancia al cuadrado de ``X[i]`` al centroide más
    cercano de los ya elegidos. Caso degenerado: si ``d2`` suma cero (todos
    los puntos restantes son copias exactas de los centroides ya elegidos) no
    hay distribución que construir, y se sortea uniformemente con
    ``int(rng.integers(len(X)))``.

    Devuelve un arreglo ``(k, d)`` con copias de filas de ``X``.
    """
    # BEGIN_TU_CODIGO
    raise NotImplementedError("Completa esta parte")
    # END_TU_CODIGO


# ----------------------------------------------------------------------
# Parte 3: comparar dos particiones con el índice de Rand
# ----------------------------------------------------------------------
def indice_rand(a, b):
    """Índice de Rand entre dos particiones de los mismos ``n`` elementos.

    Se miran todos los pares de elementos ``(i, j)`` con ``i < j``, que son
    ``n (n - 1) / 2``, y se cuenta en cuántos las dos particiones están de
    acuerdo. Un par está de acuerdo si

    * las dos particiones lo ponen junto (``a[i] == a[j]`` y ``b[i] == b[j]``),
      o bien
    * las dos lo separan (``a[i] != a[j]`` y ``b[i] != b[j]``).

    El índice es la proporción de pares de acuerdo, un número entre 0 y 1.
    Vale 1 exactamente cuando las dos particiones son la misma salvo el nombre
    de los grupos, que es justo lo que se necesita para evaluar un
    agrupamiento: los nombres de los grupos no significan nada.

    Devuelve un ``float``. Lanza ``ValueError`` si las dos particiones no
    tienen el mismo largo, o si hay menos de dos elementos (sin al menos un
    par, la proporción no está definida).
    """
    # BEGIN_TU_CODIGO
    raise NotImplementedError("Completa esta parte")
    # END_TU_CODIGO


# ----------------------------------------------------------------------
# Parte 4: PCA por la matriz de covarianza
# ----------------------------------------------------------------------
def pca(X):
    """Análisis de componentes principales por la matriz de covarianza.

    Los cuatro pasos:

    1. calcular la media de cada columna y **centrar** los datos;
    2. calcular la matriz de covarianza muestral de los datos centrados,
       ``S = Xc.T @ Xc / (n - 1)``, de forma ``(d, d)``;
    3. obtener sus autovalores y autovectores (``S`` es simétrica, así que
       corresponde ``np.linalg.eigh``, que los devuelve reales);
    4. ordenarlos por autovalor **decreciente**.

    El autovector ``j`` es la dirección ``j`` de máxima varianza y su
    autovalor es exactamente la varianza de los datos proyectados sobre ella.

    Un autovector queda igual de bien con signo cambiado, así que hace falta
    una convención o el resultado no es reproducible: **la coordenada de mayor
    valor absoluto de cada componente debe quedar positiva** (si hay empate en
    el valor absoluto, manda la de índice menor).

    Devuelve un diccionario con exactamente estas cuatro claves:

    ========================  ============================================
    ``"media"``               arreglo ``(d,)`` con la media de cada columna
    ``"componentes"``         arreglo ``(d, d)``: la **fila** ``j`` es el
                              autovector ``j``, ya ordenado y con el signo
                              de la convención
    ``"varianzas"``           arreglo ``(d,)`` con los autovalores ordenados
                              de mayor a menor
    ``"varianza_explicada"``  arreglo ``(d,)`` con la proporción de varianza
                              de cada componente (los autovalores divididos
                              por su suma). Suma 1.
    ========================  ============================================

    Lanza ``ValueError`` si ``X`` tiene menos de dos filas (la covarianza
    muestral divide por ``n - 1``) o si la varianza total es cero (todas las
    filas son idénticas y no hay dirección que destacar).
    """
    # BEGIN_TU_CODIGO
    raise NotImplementedError("Completa esta parte")
    # END_TU_CODIGO


def proyectar(X, modelo, m):
    """Proyecta ``X`` sobre las primeras ``m`` componentes del ``modelo``.

    Se centra con la media guardada en el modelo y se proyecta sobre las
    ``m`` primeras filas de ``componentes``::

        Z = (X - media) @ componentes[:m].T

    Devuelve un arreglo ``(n, m)``: las coordenadas de cada observación en la
    nueva base. Esas ``m`` columnas son la representación reducida.
    """
    # BEGIN_TU_CODIGO
    raise NotImplementedError("Completa esta parte")
    # END_TU_CODIGO


def reconstruir(Z, modelo):
    """Vuelve al espacio original desde las coordenadas reducidas ``Z``.

    Es la operación inversa de ``proyectar``, con ``m = Z.shape[1]``::

        X_aprox = Z @ componentes[:m] + media

    Devuelve un arreglo ``(n, d)``. Con ``m = d`` la reconstrucción es exacta
    salvo error de redondeo; con ``m < d`` es la mejor aproximación de rango
    ``m`` que existe, y lo que se pierde es exactamente la varianza de las
    componentes descartadas.
    """
    # BEGIN_TU_CODIGO
    raise NotImplementedError("Completa esta parte")
    # END_TU_CODIGO


# ----------------------------------------------------------------------
# Parte 5: DBSCAN
# ----------------------------------------------------------------------
def dbscan(X, eps, min_muestras):
    """Agrupa por densidad con DBSCAN.

    Vocabulario, con ``eps`` el radio y ``min_muestras`` el mínimo de vecinos:

    * la **vecindad** de ``x`` son los puntos a distancia ``<= eps`` de ``x``,
      **incluido el propio** ``x``;
    * ``x`` es un punto **de núcleo** si su vecindad tiene al menos
      ``min_muestras`` puntos;
    * un punto **de borde** no es de núcleo, pero cae en la vecindad de
      alguno que sí lo es;
    * todo lo demás es **ruido**.

    El algoritmo recorre los puntos por índice creciente. Al encontrar un
    punto de núcleo que todavía no pertenece a ningún grupo, abre un grupo
    nuevo y lo hace crecer: mete su vecindad en una cola FIFO y, mientras la
    cola tenga elementos, saca el primero, lo agrega al grupo si aún no
    pertenece a ninguno y, si además es de núcleo, encola su vecindad (en
    orden de índice creciente). Cuando la cola se vacía, el grupo está
    cerrado y la búsqueda sigue con el punto siguiente.

    Ese recorrido fija dos cosas que el corrector comprueba: los grupos se
    numeran ``0, 1, 2, ...`` en el orden en que se abren, y un punto de borde
    alcanzado desde dos grupos se queda en el **primero** que lo alcanzó.

    Devuelve un arreglo de enteros de forma ``(n,)`` con el grupo de cada
    punto y ``-1`` en los puntos de ruido.
    """
    # BEGIN_TU_CODIGO
    raise NotImplementedError("Completa esta parte")
    # END_TU_CODIGO


# ----------------------------------------------------------------------
# Parte 6: elegir el método
# ----------------------------------------------------------------------
def elegir_metodo(caso):
    """Decide qué método y qué parámetros usar en cada uno de dos casos.

    ``caso`` es una de estas dos cadenas:

    ``"lunas"``
        Dos grupos alargados y entrelazados, con poco ruido, en el plano.
        Cada grupo tiene una densidad pareja y los dos están igual de densos.
        Es el conjunto que devuelve ``util.datos_lunas()``.

    ``"gaussianos"``
        Tres nubes redondas alrededor de tres centros, de densidades muy
        distintas entre sí. Es el conjunto que devuelve
        ``util.datos_gaussianos()``.

    Devuelve un diccionario con exactamente estas dos claves:

    ==================  ==================================================
    ``"metodo"``        ``"kmeans"`` o ``"dbscan"``
    ``"parametros"``    diccionario con los argumentos del método elegido:
                        ``{"k": ...}`` para k-means,
                        ``{"eps": ..., "min_muestras": ...}`` para DBSCAN
    ==================  ==================================================

    Lanza ``ValueError`` si ``caso`` no es ninguna de las dos cadenas.

    El corrector no se conforma con el nombre del método: **corre el método
    que elegiste, con los parámetros que entregaste, sobre el conjunto que
    describe el caso**, y compara el resultado con los grupos verdaderos
    usando tu ``indice_rand``. Los parámetros tienen que servir de verdad.
    """
    # BEGIN_TU_CODIGO
    raise NotImplementedError("Completa esta parte")
    # END_TU_CODIGO


if __name__ == "__main__":
    print("Este archivo solo define funciones.")
    print("Para revisar tu trabajo: python grader.py")
