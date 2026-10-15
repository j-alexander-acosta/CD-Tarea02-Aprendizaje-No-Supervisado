#!/usr/bin/env python3
"""Corrector automático de la tarea «Aprendizaje no Supervisado».

Se ejecuta sin conexión a internet y sin más dependencia que numpy::

    python grader.py

Imprime el detalle de cada prueba, el subtotal por parte y el puntaje final.
Las pruebas que ves aquí suman 60 de los 100 puntos de la tarea y cubren al
menos un caso de cada una de las once funciones. El corrector del curso, que es
el que pone la nota, corre esas mismas pruebas más otras sobre esas mismas
funciones, con los mismos criterios, y con ellas reparte los 40 puntos
restantes. Los 100 puntos se corrigen solos.

Todo lo que se evalúa está en ``submission.py``, que es lo único que se
entrega.
"""
from __future__ import annotations

import os
import re
import signal
import sys

# Nada de .pyc en la carpeta de una entrega: el bytecode de submission
# es la solución de referencia en forma decompilable.
sys.dont_write_bytecode = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from util import (  # noqa: E402
    centroides_iniciales,
    datos_cinco_grupos,
    datos_gaussianos,
    datos_juguete,
    datos_lunas,
    datos_plano,
    particiones_ejemplo,
)

import submission as sub  # noqa: E402


# ----------------------------------------------------------------------
# Infraestructura de pruebas
# ----------------------------------------------------------------------
PARTES = [
    (1, "k-means", 24),
    (2, "La inicialización k-means++", 12),
    (3, "El índice de Rand", 14),
    (4, "PCA por la matriz de covarianza", 22),
    (5, "DBSCAN", 22),
    (6, "Elegir el método", 6),
]
PUNTAJE_TOTAL = sum(puntos for _, _, puntos in PARTES)

PRUEBAS = []


def prueba(parte, descripcion, puntos, extra=False):
    """Decorador que registra una función de prueba."""

    def decorador(funcion):
        PRUEBAS.append(
            {
                "parte": parte,
                "descripcion": descripcion,
                "puntos": puntos,
                "funcion": funcion,
                "extra": extra,
            }
        )
        return funcion

    return decorador


def cerca(obtenido, esperado, tol=1e-9):
    """Compara arreglos o escalares con tolerancia absoluta."""
    return np.allclose(
        np.asarray(obtenido, dtype=float),
        np.asarray(esperado, dtype=float),
        rtol=0.0,
        atol=tol,
    )


def afirmar(condicion, mensaje):
    if not condicion:
        raise AssertionError(mensaje)


def resumen_particion(etiquetas):
    """(cantidad de grupos sin contar el ruido, cantidad de puntos de ruido)."""
    etiquetas = np.asarray(etiquetas, dtype=int)
    grupos = sorted(set(etiquetas.tolist()) - {-1})
    return len(grupos), int((etiquetas == -1).sum())


def mejor_kmeans(X, k, reinicios=10):
    """Corre k-means con varios reinicios de k-means++ y devuelve el mejor.

    Es lo que se hace en la práctica: como el resultado depende de dónde
    partieron los centroides, se prueban varios y se conserva el de menor
    inercia.
    """
    mejor = None
    for semilla in range(reinicios):
        resultado = sub.kmeans(X, k, sub.inicializar_kmeanspp(X, k, semilla))
        if mejor is None or resultado["inercia"] < mejor["inercia"]:
            mejor = resultado
    return mejor


# ----------------------------------------------------------------------
# Parte 1: k-means
# ----------------------------------------------------------------------
@prueba(1, "la entrega usa solo numpy y la biblioteca estándar", 2)
def p1_dependencias():
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission.py")
    codigo = open(ruta, encoding="utf-8").read()
    prohibidas = re.findall(
        r"^\s*(?:import|from)\s+(sklearn|scipy|pandas|torch|tensorflow)\b",
        codigo,
        re.M,
    )
    afirmar(
        not prohibidas,
        f"submission.py importa {sorted(set(prohibidas))}: esta tarea se "
        "resuelve con numpy y la biblioteca estándar, porque la gracia es "
        "escribir los algoritmos a mano",
    )
    # Sin esta segunda mitad, una entrega en blanco se lleva estos dos puntos:
    # no importa nada prohibido justamente porque no hace nada.
    sin_hacer = len(re.findall(r"^\s*raise NotImplementedError", codigo, re.M))
    afirmar(
        sin_hacer == 0,
        f"quedan {sin_hacer} funciones sin implementar en submission.py: esta "
        "revisión mira que la entrega respete las bibliotecas permitidas, y una "
        "entrega sin resolver no cuenta",
    )


@prueba(1, "asignar_grupos: valores, forma y arreglos intactos", 3)
def p1_asignar():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    C = np.array([[0.0, 0.0], [6.0, 6.0]])
    copia_X, copia_C = X.copy(), C.copy()
    obtenido = sub.asignar_grupos(X, C)

    afirmar(
        np.asarray(obtenido).shape == (4,),
        f"asignar_grupos debe devolver un arreglo de forma (4,); dio "
        f"{np.asarray(obtenido).shape}",
    )
    afirmar(
        np.asarray(obtenido).tolist() == [0, 0, 0, 1],
        f"los tres primeros puntos están más cerca del centroide 0 y el "
        f"cuarto del centroide 1: se esperaba [0, 0, 0, 1] y dio "
        f"{np.asarray(obtenido).tolist()}",
    )
    afirmar(
        cerca(X, copia_X, 0.0) and cerca(C, copia_C, 0.0),
        "asignar_grupos no debe modificar X ni C",
    )


@prueba(1, "actualizar_centroides: promedios y grupo vacío", 4)
def p1_actualizar():
    X = datos_juguete()
    etiquetas = np.array([0, 1, 0, 1, 1, 1])
    C_previo = np.array([[0.0, 0.0], [1.0, 0.0]])
    copia = C_previo.copy()
    obtenido = sub.actualizar_centroides(X, etiquetas, 2, C_previo)

    afirmar(
        np.asarray(obtenido).shape == (2, 2),
        f"se esperaba forma (2, 2) y dio {np.asarray(obtenido).shape}",
    )
    afirmar(
        cerca(obtenido, [[0.0, 0.5], [4.25, 4.0]]),
        "el centroide de un grupo es el promedio de sus puntos: con esa "
        f"asignación se esperaba [[0, 0,5], [4,25, 4]] y dio {obtenido}",
    )
    afirmar(cerca(C_previo, copia, 0.0),
            "actualizar_centroides no debe modificar C_previo")

    # Grupo vacío: nadie eligió el centroide 1, que debe quedar como estaba.
    vacio = sub.actualizar_centroides(
        np.array([[0.0, 0.0], [2.0, 0.0]]),
        np.array([0, 0]),
        2,
        np.array([[9.0, 9.0], [-3.0, 7.0]]),
    )
    afirmar(
        cerca(vacio, [[1.0, 0.0], [-3.0, 7.0]]),
        "un grupo vacío conserva su centroide anterior (no es cero ni NaN): "
        f"se esperaba [[1, 0], [-3, 7]] y dio {vacio}",
    )


@prueba(1, "inercia: contrato y valor sobre el ejemplo del enunciado", 2)
def p1_inercia():
    X = datos_juguete()
    C = np.array([[0.0, 0.5], [4.25, 4.0]])
    etiquetas = np.array([0, 1, 0, 1, 1, 1])
    obtenido = sub.inercia(X, C, etiquetas)
    afirmar(
        isinstance(obtenido, float) or np.isscalar(obtenido),
        f"inercia debe devolver un escalar; devolvió {type(obtenido).__name__}",
    )
    afirmar(
        cerca(obtenido, 37.25, 1e-9),
        f"la suma de distancias al cuadrado al centroide asignado da 37,25 "
        f"y dio {obtenido}",
    )


@prueba(1, "kmeans: el ejemplo del enunciado, contrato y conteo de iteraciones", 3)
def p1_kmeans():
    X = datos_juguete()
    C0 = np.array([[0.0, 0.0], [1.0, 0.0]])
    copia = C0.copy()
    r = sub.kmeans(X, 2, C0)

    afirmar(isinstance(r, dict),
            f"kmeans debe devolver un diccionario; devolvió {type(r).__name__}")
    afirmar(
        set(r) == {"centroides", "etiquetas", "inercia", "iteraciones", "convergio"},
        "las claves deben ser exactamente centroides, etiquetas, inercia, "
        f"iteraciones y convergio; trajo {sorted(r)}",
    )
    afirmar(cerca(C0, copia, 0.0), "kmeans no debe modificar C0")
    afirmar(bool(r["convergio"]),
            "sobre estos seis puntos k-means converge: debía informarlo")
    afirmar(
        np.asarray(r["etiquetas"]).tolist() == [0, 0, 0, 1, 1, 1],
        "la partición final separa los tres puntos de abajo de los tres de "
        f"arriba: [0, 0, 0, 1, 1, 1] y dio {np.asarray(r['etiquetas']).tolist()}",
    )
    afirmar(
        cerca(r["centroides"], [[1 / 3, 1 / 3], [16 / 3, 16 / 3]], 1e-9),
        "los centroides finales son los promedios de cada trío, (1/3, 1/3) y "
        f"(16/3, 16/3); dio {r['centroides']}",
    )
    afirmar(
        cerca(r["inercia"], 8.0 / 3.0, 1e-9),
        f"la inercia final es 8/3 = 2,667 y dio {r['inercia']}",
    )
    afirmar(
        int(r["iteraciones"]) == 3,
        "son tres iteraciones: dos que mueven los centroides y una tercera "
        f"que comprueba que la asignación ya no cambia; contó {r['iteraciones']}",
    )


# ----------------------------------------------------------------------
# Parte 2: k-means++
# ----------------------------------------------------------------------
@prueba(2, "inicializar_kmeanspp: contrato, filas de X y primer sorteo", 4)
def p2_contrato():
    X, _ = datos_cinco_grupos()
    copia = X.copy()
    C = sub.inicializar_kmeanspp(X, 5, 0)

    afirmar(
        np.asarray(C).shape == (5, 2),
        f"se esperaba forma (5, 2) y dio {np.asarray(C).shape}",
    )
    afirmar(cerca(X, copia, 0.0), "inicializar_kmeanspp no debe modificar X")

    for j, centroide in enumerate(np.asarray(C)):
        distancias = ((X - centroide) ** 2).sum(axis=1)
        afirmar(
            float(distancias.min()) < 1e-18,
            f"el centroide {j} no es ninguna fila de X: k-means++ elige "
            "puntos del conjunto, no puntos inventados",
        )
    afirmar(
        len({tuple(fila) for fila in np.asarray(C).tolist()}) == 5,
        "los cinco centroides deben ser puntos distintos; se repitió alguno",
    )

    esperado_primero = X[int(np.random.default_rng(0).integers(len(X)))]
    afirmar(
        cerca(np.asarray(C)[0], esperado_primero),
        "el primer centroide se sortea uniformemente con "
        "int(rng.integers(len(X))) sobre un rng creado con la semilla que se "
        f"recibe; se esperaba {esperado_primero} y dio {np.asarray(C)[0]}",
    )


@prueba(2, "inicializar_kmeanspp: baja la inercia frente al sorteo uniforme", 2)
def p2_inercia():
    X, _ = datos_cinco_grupos()
    con_pp = [sub.kmeans(X, 5, sub.inicializar_kmeanspp(X, 5, s))["inercia"]
              for s in range(15)]
    con_uniforme = [sub.kmeans(X, 5, centroides_iniciales(X, 5, s))["inercia"]
                    for s in range(15)]
    mediana_pp = float(np.median(con_pp))
    mediana_uniforme = float(np.median(con_uniforme))
    afirmar(
        mediana_pp < 0.5 * mediana_uniforme,
        "sobre este conjunto la inercia mediana de 15 corridas con k-means++ "
        "debe ser menos de la mitad de la del sorteo uniforme; dio "
        f"{mediana_pp:.1f} contra {mediana_uniforme:.1f}",
    )


# ----------------------------------------------------------------------
# Parte 3: el índice de Rand
# ----------------------------------------------------------------------
@prueba(3, "indice_rand: valor, partición consigo misma y renombre de grupos", 5)
def p3_valores():
    a, b = particiones_ejemplo()
    obtenido = sub.indice_rand(a, b)
    afirmar(
        cerca(obtenido, 10.0 / 15.0, 1e-12),
        "de los 15 pares de estas dos particiones hay 10 de acuerdo, así que "
        f"el índice es 10/15 = 0,6667; dio {obtenido}",
    )
    afirmar(
        cerca(sub.indice_rand(a, a), 1.0, 1e-12),
        "una partición comparada consigo misma da exactamente 1",
    )
    renombrada = np.array([7, 7, 7, 2, 2, 2])
    afirmar(
        cerca(sub.indice_rand(a, renombrada), 1.0, 1e-12),
        "el índice de Rand no mira los nombres de los grupos: [0,0,0,1,1,1] y "
        f"[7,7,7,2,2,2] son la misma partición y deben dar 1; dio "
        f"{sub.indice_rand(a, renombrada)}",
    )
    afirmar(
        cerca(sub.indice_rand(b, a), obtenido, 1e-12),
        "el índice debe ser simétrico: indice_rand(a, b) e indice_rand(b, a) "
        "tienen que coincidir",
    )


@prueba(3, "indice_rand: rango y sensibilidad sobre una partición real", 3)
def p3_rango():
    X, y = datos_gaussianos()
    perfecta = sub.indice_rand(y, y)
    afirmar(cerca(perfecta, 1.0, 1e-12), f"indice_rand(y, y) debe dar 1; dio {perfecta}")

    rng = np.random.default_rng(0)
    azar = rng.integers(0, 3, size=len(y))
    al_azar = sub.indice_rand(y, azar)
    afirmar(
        0.0 <= al_azar <= 1.0,
        f"el índice siempre está entre 0 y 1; dio {al_azar}",
    )
    afirmar(
        al_azar < 0.8,
        "una partición sorteada al azar no puede parecerse tanto a la "
        f"verdadera: dio {al_azar:.3f}, y con tres grupos al azar lo esperable "
        "ronda 0,55",
    )
    casi = y.copy()
    casi[:5] = (casi[:5] + 1) % 3
    afirmar(
        al_azar < sub.indice_rand(y, casi) < 1.0,
        "cambiar cinco etiquetas debe dar un índice alto pero menor que 1, y "
        "en todo caso mayor que el de una partición al azar",
    )


# ----------------------------------------------------------------------
# Parte 4: PCA
# ----------------------------------------------------------------------
@prueba(4, "pca: contrato y valores exactos sobre una matriz pequeña", 6)
def p4_valores():
    X = np.array([[2.0, 0.0], [-2.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    copia = X.copy()
    modelo = sub.pca(X)

    afirmar(isinstance(modelo, dict),
            f"pca debe devolver un diccionario; devolvió {type(modelo).__name__}")
    afirmar(
        set(modelo) == {"media", "componentes", "varianzas", "varianza_explicada"},
        "las claves deben ser exactamente media, componentes, varianzas y "
        f"varianza_explicada; trajo {sorted(modelo)}",
    )
    afirmar(cerca(X, copia, 0.0), "pca no debe modificar X")
    afirmar(
        cerca(modelo["media"], [0.0, 0.0]),
        f"la media de estas cuatro filas es (0, 0); dio {modelo['media']}",
    )
    afirmar(
        cerca(modelo["varianzas"], [8.0 / 3.0, 2.0 / 3.0], 1e-9),
        "con covarianza muestral (dividir por n - 1) los autovalores son 8/3 "
        f"y 2/3; dio {modelo['varianzas']}. Si te dio 2 y 0,5, dividiste por n",
    )
    afirmar(
        cerca(modelo["varianza_explicada"], [0.8, 0.2], 1e-9),
        f"la varianza explicada es [0,8, 0,2] y dio {modelo['varianza_explicada']}",
    )
    afirmar(
        np.asarray(modelo["componentes"]).shape == (2, 2),
        f"componentes debe tener forma (2, 2), una fila por componente; dio "
        f"{np.asarray(modelo['componentes']).shape}",
    )
    afirmar(
        cerca(modelo["componentes"], [[1.0, 0.0], [0.0, 1.0]], 1e-9),
        "las direcciones principales son los ejes, y con la convención de "
        "signo (coordenada de mayor magnitud positiva) quedan [[1, 0], [0, 1]]; "
        f"dio {modelo['componentes']}",
    )


@prueba(4, "proyectar: forma y coordenadas en la nueva base", 3)
def p4_proyectar():
    X = np.array([[2.0, 0.0], [-2.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    modelo = sub.pca(X)
    Z = sub.proyectar(X, modelo, 1)
    afirmar(
        np.asarray(Z).shape == (4, 1),
        f"con m = 1 la proyección tiene forma (4, 1); dio {np.asarray(Z).shape}",
    )
    afirmar(
        cerca(Z, [[2.0], [-2.0], [0.0], [0.0]], 1e-9),
        f"sobre la primera componente las coordenadas son 2, -2, 0 y 0; dio "
        f"{np.asarray(Z).ravel()}",
    )

    Xp = datos_plano()
    modelo = sub.pca(Xp)
    Z = sub.proyectar(Xp, modelo, 2)
    afirmar(
        np.asarray(Z).shape == (len(Xp), 2),
        f"se esperaba forma ({len(Xp)}, 2) y dio {np.asarray(Z).shape}",
    )
    afirmar(
        cerca(np.asarray(Z).mean(axis=0), [0.0, 0.0], 1e-8),
        "los datos se centran antes de proyectar, así que la proyección tiene "
        f"media cero; dio {np.asarray(Z).mean(axis=0)}",
    )
    afirmar(
        cerca(np.asarray(Z).var(axis=0, ddof=1),
              np.asarray(modelo["varianzas"])[:2], 1e-8),
        "la varianza de la columna j de la proyección tiene que ser el "
        "autovalor j: eso es lo que significa «dirección de máxima varianza»",
    )


@prueba(4, "reconstruir: exacta con todas las componentes y de rango bajo con dos", 5)
def p4_reconstruir():
    Xp = datos_plano()
    modelo = sub.pca(Xp)

    Z = sub.proyectar(Xp, modelo, 5)
    exacta = sub.reconstruir(Z, modelo)
    afirmar(
        np.asarray(exacta).shape == Xp.shape,
        f"reconstruir debe devolver forma {Xp.shape}; dio "
        f"{np.asarray(exacta).shape}",
    )
    afirmar(
        cerca(exacta, Xp, 1e-8),
        "con las cinco componentes la reconstrucción tiene que devolver los "
        "datos originales (¿falta sumar la media de vuelta?)",
    )

    Z2 = sub.proyectar(Xp, modelo, 2)
    aprox = sub.reconstruir(Z2, modelo)
    centrado = Xp - Xp.mean(axis=0)
    error = float(np.sum((Xp - aprox) ** 2) / np.sum(centrado ** 2))
    afirmar(
        error < 0.01,
        "estos datos de cinco columnas viven casi sobre un plano: dos "
        "componentes deben reconstruirlos con menos de 1% de error relativo, "
        f"y el error fue {100 * error:.2f}%",
    )
    perdida = 1.0 - float(np.asarray(modelo["varianza_explicada"])[:2].sum())
    afirmar(
        abs(error - perdida) < 1e-6,
        "el error relativo de la reconstrucción de rango 2 tiene que coincidir "
        "con la varianza explicada que se descartó "
        f"({error:.6f} contra {perdida:.6f}): esa igualdad es la que hace que "
        "la varianza explicada sirva para elegir cuántas componentes usar",
    )


# ----------------------------------------------------------------------
# Parte 5: DBSCAN
# ----------------------------------------------------------------------
def _conjunto_dbscan():
    """Ocho puntos en una recta: dos grupos densos y un punto aislado."""
    return np.array(
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0],
         [5.0, 0.0], [8.0, 0.0], [8.5, 0.0], [9.0, 0.0]]
    )


@prueba(5, "dbscan: núcleo, borde y ruido sobre un conjunto pequeño", 6)
def p5_pequeno():
    X = _conjunto_dbscan()
    copia = X.copy()
    obtenido = np.asarray(sub.dbscan(X, 0.6, 3))

    afirmar(
        obtenido.shape == (8,),
        f"dbscan debe devolver un arreglo de forma (8,); dio {obtenido.shape}",
    )
    afirmar(cerca(X, copia, 0.0), "dbscan no debe modificar X")
    afirmar(
        obtenido.tolist() == [0, 0, 0, 0, -1, 1, 1, 1],
        "con eps = 0,6 y min_muestras = 3 hay dos grupos y un punto de ruido: "
        f"se esperaba [0, 0, 0, 0, -1, 1, 1, 1] y dio {obtenido.tolist()}. "
        "Recuerda que la vecindad incluye al propio punto y que el punto "
        "aislado en x = 5 no alcanza a ser núcleo ni borde",
    )


@prueba(5, "dbscan: numeración de grupos y efecto de min_muestras", 5)
def p5_parametros():
    X = _conjunto_dbscan()

    todos_nucleo = np.asarray(sub.dbscan(X, 0.6, 1))
    afirmar(
        todos_nucleo.tolist() == [0, 0, 0, 0, 1, 2, 2, 2],
        "con min_muestras = 1 hasta el punto aislado es núcleo y abre su "
        "propio grupo; los grupos se numeran en el orden en que se abren, "
        f"así que se esperaba [0, 0, 0, 0, 1, 2, 2, 2] y dio {todos_nucleo.tolist()}",
    )

    exigente = np.asarray(sub.dbscan(X, 0.6, 5))
    afirmar(
        exigente.tolist() == [-1] * 8,
        "con min_muestras = 5 ningún punto llega a tener cinco vecinos dentro "
        f"del radio: todo es ruido. Dio {exigente.tolist()}",
    )

    uno_solo = np.asarray(sub.dbscan(X, 20.0, 3))
    grupos, ruido = resumen_particion(uno_solo)
    afirmar(
        grupos == 1 and ruido == 0,
        f"con un radio enorme todo queda en un único grupo sin ruido; dio "
        f"{grupos} grupo(s) y {ruido} punto(s) de ruido",
    )

    # Punto de borde alcanzable desde dos grupos: se queda en el primero.
    Y = np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0],
                  [1.0, 0.0],
                  [1.6, 0.0], [1.8, 0.0], [2.0, 0.0]])
    etiquetas = np.asarray(sub.dbscan(Y, 0.6, 3))
    afirmar(
        etiquetas.tolist() == [0, 0, 0, 0, 1, 1, 1],
        "el punto del medio es de borde y lo alcanzan los dos grupos: se "
        "queda en el que lo alcanzó primero, que es el 0. Se esperaba "
        f"[0, 0, 0, 0, 1, 1, 1] y dio {etiquetas.tolist()}",
    )


@prueba(5, "dbscan: densidades distintas, donde un solo radio no alcanza", 3)
def p5_densidades():
    X, y = datos_gaussianos()

    estrecho = np.asarray(sub.dbscan(X, 0.25, 5))
    grupos_e, ruido_e = resumen_particion(estrecho)
    afirmar(
        ruido_e > 30,
        "con eps = 0,25 el grupo disperso queda casi entero como ruido: se "
        f"esperaban más de 30 puntos de ruido y hubo {ruido_e}",
    )

    ancho = np.asarray(sub.dbscan(X, 0.6, 5))
    rand_ancho = sub.indice_rand(ancho, y)
    afirmar(
        rand_ancho < 0.8,
        "con eps = 0,6 los dos grupos densos se fusionan y la partición se "
        f"aleja de la verdadera; el índice de Rand dio {rand_ancho:.4f}, "
        "demasiado alto para lo que debería pasar",
    )
    afirmar(
        grupos_e >= 1,
        "con eps = 0,25 debería quedar al menos un grupo además del ruido",
    )


# ----------------------------------------------------------------------
# Parte 6: elegir el método
# ----------------------------------------------------------------------
def aplicar_eleccion(caso, X):
    """Corre el método que la entrega eligió, con los parámetros que dio."""
    decision = sub.elegir_metodo(caso)
    afirmar(
        isinstance(decision, dict)
        and set(decision) == {"metodo", "parametros"},
        "elegir_metodo debe devolver un diccionario con exactamente las "
        f"claves metodo y parametros; trajo "
        f"{sorted(decision) if isinstance(decision, dict) else type(decision).__name__}",
    )
    afirmar(
        decision["metodo"] in {"kmeans", "dbscan"},
        f"el método debe ser «kmeans» o «dbscan»; dijo {decision['metodo']!r}",
    )

    parametros = decision["parametros"]
    if decision["metodo"] == "dbscan":
        afirmar(
            set(parametros) == {"eps", "min_muestras"},
            "para DBSCAN los parámetros son eps y min_muestras; trajo "
            f"{sorted(parametros)}",
        )
        return np.asarray(sub.dbscan(X, parametros["eps"],
                                     parametros["min_muestras"]))
    afirmar(
        set(parametros) == {"k"},
        f"para k-means el único parámetro es k; trajo {sorted(parametros)}",
    )
    return np.asarray(mejor_kmeans(X, int(parametros["k"]))["etiquetas"])


@prueba(6, "elegir_metodo: contrato y caso desconocido", 2)
def p6_contrato():
    for caso in ("lunas", "gaussianos"):
        decision = sub.elegir_metodo(caso)
        afirmar(
            isinstance(decision, dict) and "metodo" in decision,
            f"elegir_metodo({caso!r}) no devolvió un diccionario con la clave "
            "metodo",
        )
    try:
        sub.elegir_metodo("otra cosa")
    except ValueError:
        pass
    else:
        afirmar(False, "con un caso desconocido debía lanzar ValueError")


@prueba(6, "elegir_metodo: la elección funciona sobre las dos lunas", 2)
def p6_lunas():
    X, y = datos_lunas()
    etiquetas = aplicar_eleccion("lunas", X)
    grupos, ruido = resumen_particion(etiquetas)
    rand = sub.indice_rand(etiquetas, y)
    afirmar(
        grupos == 2 and ruido <= 0.02 * len(X) and rand >= 0.95,
        "el método y los parámetros que elegiste tienen que recuperar las dos "
        f"lunas: se esperaban 2 grupos, a lo más {int(0.02 * len(X))} puntos "
        f"de ruido e índice de Rand sobre 0,95, y se obtuvieron {grupos} "
        f"grupo(s), {ruido} de ruido y Rand {rand:.3f}",
    )


# ----------------------------------------------------------------------
# Ejecución
# ----------------------------------------------------------------------
TOPE_SEGUNDOS = 120

# El temporizador se apoya en SIGALRM, que existe en Linux y macOS pero no en
# Windows. Si no está disponible, el corrector funciona igual, solo que sin
# tope de tiempo.
HAY_TEMPORIZADOR = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")


class TiempoAgotado(Exception):
    """Se levanta cuando una prueba supera el tope de tiempo."""


def _sono_la_alarma(_numero, _marco):
    raise TiempoAgotado


def ejecutar_una(caso):
    """Corre una prueba y devuelve (puntos obtenidos, detalle del error).

    Cada prueba tiene un tope de ``TOPE_SEGUNDOS``: un bucle escrito como
    ``while True`` dejaría al corrector colgado para siempre, y en una
    corrección por lotes una sola entrega así bloquearía todo lo demás.
    """
    anterior = None
    if HAY_TEMPORIZADOR:
        anterior = signal.signal(signal.SIGALRM, _sono_la_alarma)
        signal.setitimer(signal.ITIMER_REAL, TOPE_SEGUNDOS)
    try:
        caso["funcion"]()
    except TiempoAgotado:
        return 0, (f"la prueba superó los {TOPE_SEGUNDOS} s: "
                   "¿el bucle de k-means respeta max_iter?")
    except NotImplementedError:
        return 0, "sin implementar todavía"
    except AssertionError as exc:
        return 0, str(exc)
    except Exception as exc:  # noqa: BLE001 — el corrector nunca debe caerse
        return 0, f"{type(exc).__name__}: {exc}"
    finally:
        if HAY_TEMPORIZADOR:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, anterior)
    return caso["puntos"], ""


def main():
    print("=" * 70)
    print(" Corrector — Tarea: Aprendizaje no Supervisado")
    print("=" * 70)

    obtenido_total = 0
    maximo_total = 0
    hay_extra = any(c["extra"] for c in PRUEBAS)

    for numero, titulo, _ in PARTES:
        casos = [c for c in PRUEBAS if c["parte"] == numero]
        if not casos:
            continue
        print(f"\nParte {numero}: {titulo}")
        obtenido_parte = maximo_parte = 0
        for caso in casos:
            puntos, detalle = ejecutar_una(caso)
            obtenido_parte += puntos
            maximo_parte += caso["puntos"]
            marca = "OK   " if puntos == caso["puntos"] else "FALLA"
            sufijo = "  (adicional)" if caso["extra"] else ""
            print(f"  [{marca}] {caso['descripcion']}{sufijo}"
                  f"  ->  {puntos}/{caso['puntos']}")
            if detalle:
                print(f"           {detalle}")
        print(f"  Subtotal parte {numero}: {obtenido_parte}/{maximo_parte}")
        obtenido_total += obtenido_parte
        maximo_total += maximo_parte

    print("\n" + "=" * 70)
    if hay_extra:
        print(f" PUNTAJE TOTAL: {obtenido_total}/{maximo_total}")
    else:
        print(f" PUNTAJE EN LAS PRUEBAS VISIBLES: {obtenido_total}/{maximo_total}")
        print(f" La tarea vale {PUNTAJE_TOTAL} puntos: los "
              f"{PUNTAJE_TOTAL - maximo_total} restantes se obtienen en las")
        print(" pruebas adicionales que el corrector del curso corre sobre las")
        print(" mismas funciones, con los mismos criterios.")
    print("=" * 70)
    return 0 if obtenido_total == maximo_total else 1


if __name__ == "__main__":
    sys.exit(main())
