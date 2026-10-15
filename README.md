# Tarea: Aprendizaje no Supervisado

Ciencia de Datos — Universidad Católica del Norte

## Archivos

| Archivo | Qué es |
|---|---|
| `enunciado.pdf` | El enunciado completo. Léelo primero. |
| `submission.py` | **El único archivo que debes editar y entregar.** |
| `util.py` | Los conjuntos de datos. No se modifica. |
| `grader.py` | Corrector automático. Corre sin conexión a internet. |
| `requirements.txt` | La única dependencia: `numpy`. |

## Antes de empezar

`COMO-TRABAJAR.pdf` trae las instrucciones generales: qué instalar, cómo abrir
una terminal en esta carpeta, qué significa el puntaje del corrector y los
errores más frecuentes. Sirve para todas las tareas del curso.

Un detalle que conviene saber de entrada: en Windows el intérprete se llama
`python` (o `py`), y en Linux y macOS suele ser `python3`. Los comandos de acá
usan `python`: si tu sistema responde «command not found», prueba con
`python3`.

## Cómo trabajar

Basta con Python 3 y numpy:

```
pip install -r requirements.txt
```

Cada vez que completes una parte, corre el corrector desde esta carpeta:

```
python grader.py
```

El corrector imprime el detalle de cada prueba, el subtotal por parte y el
puntaje total. Cuando una prueba falla, la línea siguiente dice qué se esperaba
y qué se obtuvo: esa línea es la que hay que leer antes de volver al código.
Las pruebas visibles suman 60 de los 100 puntos y tocan las once funciones, así
que sirven para depurar cada una; los 40 restantes salen de pruebas adicionales
que el corrector del curso corre sobre esas mismas funciones, con los mismos
criterios. Los 100 puntos se corrigen solos, sin revisión a mano.

Los datos se generan localmente con semillas fijas, así que el corrector es
determinista: dos corridas seguidas dan exactamente el mismo puntaje, y no hace
falta conexión a internet en ningún momento.

## Entrega

Un solo archivo: `submission.py` con las once funciones completadas. No se
entrega nada más. No cambies las firmas ni el nombre del archivo: el corrector
las llama tal como están escritas. No hay que subir datos ni incluir la carpeta
`__pycache__`.
