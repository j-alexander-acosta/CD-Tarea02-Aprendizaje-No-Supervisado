"""Utilidades de la tarea «Aprendizaje no Supervisado». No se modifica.

Aquí viven los conjuntos de datos. Todos se generan localmente con semilla
fija: la tarea no descarga nada y la corrección es determinista, así que dos
personas que corran el mismo comando obtienen exactamente los mismos números.

La única dependencia es numpy.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "datos_juguete",
    "datos_gaussianos",
    "datos_cinco_grupos",
    "datos_lunas",
    "datos_plano",
    "particiones_ejemplo",
    "centroides_iniciales",
]


def datos_juguete():
    """Los seis puntos del ejemplo que el enunciado desarrolla a mano.

    El orden de las filas es el del enunciado, así que el resultado de
    ``kmeans`` se puede comparar directamente con el desarrollo en papel.

    Devuelve ``X`` de forma ``(6, 2)``.
    """
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [5.0, 5.0],
            [6.0, 5.0],
            [5.0, 6.0],
        ]
    )


def datos_gaussianos(semilla=3):
    """Tres grupos gaussianos con densidades muy distintas entre sí.

    La gracia del conjunto es que los tres grupos no se parecen: dos son
    compactos y están relativamente cerca, y el tercero es ancho y disperso.
    Esa diferencia de densidad es lo que hace que un único radio no le sirva
    igual de bien a los tres grupos.

    ================  ========  ==========  =================================
    grupo             puntos    centro      desviación por coordenada
    ================  ========  ==========  =================================
    0 (denso)         120       (0, 0)      0,30
    1 (denso)         80        (1,6, 0,3)  0,28
    2 (disperso)      40        (0,8, 3,6)  1,10
    ================  ========  ==========  =================================

    Devuelve ``(X, y)`` con ``X`` de forma ``(240, 2)`` y ``y`` de forma
    ``(240,)`` con el grupo verdadero de cada punto (0, 1 o 2). Las etiquetas
    verdaderas están para poder evaluar, no para agrupar: ningún método de
    esta tarea las mira.
    """
    rng = np.random.default_rng(semilla)
    centros = [(0.0, 0.0), (1.6, 0.3), (0.8, 3.6)]
    tamanos = [120, 80, 40]
    sigmas = [0.30, 0.28, 1.10]

    bloques = []
    etiquetas = []
    for grupo, (centro, tamano, sigma) in enumerate(zip(centros, tamanos, sigmas)):
        bloques.append(np.array(centro) + sigma * rng.normal(size=(tamano, 2)))
        etiquetas.append(np.full(tamano, grupo))
    return np.vstack(bloques), np.concatenate(etiquetas)


def datos_cinco_grupos(semilla=5):
    """Cinco grupos redondos y bien separados, con tamaños muy desiguales.

    Uno de los grupos tiene 240 puntos y los otros cuatro tienen 20 cada uno.
    El conjunto es fácil a la vista (los cinco grupos están despegados), pero
    es duro para la inicialización: si los centroides iniciales se sortean
    uniformemente entre las filas, la mayoría cae dentro del grupo grande y
    algún grupo pequeño se queda sin ninguno. Es el conjunto donde se aprecia
    lo que aporta k-means++.

    Devuelve ``(X, y)`` con ``X`` de forma ``(320, 2)``.
    """
    rng = np.random.default_rng(semilla)
    centros = [(0.0, 0.0), (7.0, 0.0), (0.0, 7.0), (7.0, 7.0), (3.5, 3.5)]
    tamanos = [240, 20, 20, 20, 20]
    sigmas = [0.35, 0.30, 0.30, 0.30, 0.30]

    bloques = []
    etiquetas = []
    for grupo, (centro, tamano, sigma) in enumerate(zip(centros, tamanos, sigmas)):
        bloques.append(np.array(centro) + sigma * rng.normal(size=(tamano, 2)))
        etiquetas.append(np.full(tamano, grupo))
    return np.vstack(bloques), np.concatenate(etiquetas)


def datos_lunas(n=200, ruido=0.06, semilla=7):
    """El conjunto clásico de «dos lunas»: dos arcos entrelazados.

    Cada luna es media circunferencia con ruido gaussiano de desviación
    ``ruido``. Los dos grupos son evidentes a la vista y ninguno de los dos es
    una nube redonda alrededor de un centro, que es exactamente el supuesto
    que hace k-means.

    Devuelve ``(X, y)`` con ``X`` de forma ``(2n, 2)`` y ``y`` con la luna de
    cada punto (0 o 1). Las primeras ``n`` filas son la luna 0.
    """
    rng = np.random.default_rng(semilla)
    angulos = np.linspace(0.0, np.pi, n)
    luna0 = np.column_stack([np.cos(angulos), np.sin(angulos)])
    luna1 = np.column_stack([1.0 - np.cos(angulos), 0.5 - np.sin(angulos)])
    X = np.vstack([luna0, luna1]) + ruido * rng.normal(size=(2 * n, 2))
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    return X, y


def datos_plano(n=150, semilla=11):
    """Puntos de R^5 que en realidad viven casi sobre un plano de R^2.

    Se construyen como ``media + a * u1 + b * u2 + ruido``, con ``u1`` y
    ``u2`` dos direcciones fijas y ortonormales de R^5. El coeficiente ``a``
    tiene desviación 3,0 y ``b`` desviación 1,0, así que la primera dirección
    concentra bastante más varianza que la segunda. El ruido, de desviación
    0,02, reparte una cantidad ínfima de varianza en las otras tres
    direcciones.

    Es el conjunto donde se ve que dos componentes bastan para reconstruir
    cinco coordenadas casi sin error.

    Devuelve ``X`` de forma ``(n, 5)``.
    """
    rng = np.random.default_rng(semilla)
    u1 = np.array([1.0, 1.0, 0.0, -1.0, 0.0])
    u2 = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
    u1 = u1 / np.linalg.norm(u1)
    u2 = u2 - (u2 @ u1) * u1
    u2 = u2 / np.linalg.norm(u2)

    media = np.array([2.0, -1.0, 0.5, 3.0, -2.0])
    a = 3.0 * rng.normal(size=(n, 1))
    b = 1.0 * rng.normal(size=(n, 1))
    return media + a * u1 + b * u2 + 0.02 * rng.normal(size=(n, 5))


def particiones_ejemplo():
    """Dos particiones de seis elementos, para comparar con el índice de Rand.

    La primera es ``[0, 0, 0, 1, 1, 1]`` y la segunda ``[0, 0, 1, 1, 2, 2]``:
    coinciden en la mitad de los pares y difieren en la otra mitad, así que
    sirven de control de escritorio.

    Devuelve ``(a, b)``, dos arreglos de forma ``(6,)``.
    """
    a = np.array([0, 0, 0, 1, 1, 1])
    b = np.array([0, 0, 1, 1, 2, 2])
    return a, b


def centroides_iniciales(X, k, semilla):
    """Inicialización uniforme: ``k`` filas distintas de ``X``, al azar.

    Es la inicialización ingenua, la que se usa como punto de comparación
    contra k-means++. Devuelve un arreglo ``(k, d)`` que es una copia: quien
    la reciba puede modificarla sin tocar ``X``.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(semilla)
    indices = rng.choice(len(X), size=k, replace=False)
    return X[indices].copy()
