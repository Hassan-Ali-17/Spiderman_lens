# Filters

Es un proyecto de realidad aumentada que genera un portal interactivo en el video de la cámara utilizando el seguimiento de manos en tiempo real. A través de un gesto natural, el usuario puede alternar entre 8 filtros visuales distintos que se renderizan en vivo dentro del portal.

---

## Características

- Seguimiento de manos en tiempo real mediante MediaPipe Hands.
- Portal en perspectiva construido dinámicamente a partir de las puntas de índice y pulgar de ambas manos.
- Ocho filtros visuales aplicados exclusivamente dentro del área del portal.
- Cambio de filtro mediante gesto: acercar las manos activa la transición al siguiente filtro de la secuencia.
- Sistema de histéresis para evitar cambios accidentales por temblores o imprecisión del tracking.

## Filtros incluidos

| Filtro | Descripción |
|--------|-------------|
| `filtro_grid` | Superposición de cuadrícula sobre la imagen original |
| `filtro_1` | Duotono segmentado por umbrales de luminosidad |
| `filtro_2` | Trama de puntos (halftone) en blanco y negro |
| `filtro_3` | Aberración cromática con separación de canales RGB |
| `filtro_5` | Simulación de cámara térmica mediante colormap |
| `filtro_6` | Estilo vintage sepia con viñeteado y grano |
| `filtro_blanco` | Efecto de vidrio esmerilado sobre la imagen |
| `filtro_rosa` | Halftone en duotono rosa-magenta |

## Instalación

Clonar el repositorio:

```bash
git clone https://github.com/mishu006/Filters.git
cd Filters
```

Crear un entorno virtual:

```bash
python -m venv venv
```

Activarlo:

```bash
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS / Linux
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Uso

```bash
python main.py
```

Con la cámara activa, levanta ambas manos con el índice y el pulgar extendidos: el portal se genera automáticamente entre ellas. Acércalas para "cerrarlas" y avanzar al siguiente filtro de la lista. Presiona **`q`** con la ventana activa para finalizar la ejecución.

## Estructura del proyecto

```
Filters/
├── main.py            Punto de entrada: bucle de captura y ciclo de filtros
├── hand_tracking.py    Detección de dedos extendidos a partir de los landmarks
├── geometry.py          Geometría del portal y detección del gesto de cierre
├── filters.py            Definición de los ocho filtros disponibles
├── requirements.txt
└── README.md
```

## Extender el proyecto

Para agregar un nuevo filtro, basta con definir una función en `filters.py` que reciba un recorte en formato BGR (`numpy.ndarray`) y devuelva un recorte del mismo tamaño:

```python
def filtro_nuevo(roi: np.ndarray) -> np.ndarray:
    return roi
```

Después, incorpórala a la lista `FILTROS` al final del archivo. El ciclo de filtros se ajusta automáticamente a la cantidad de elementos que contenga esa lista.

## Stack técnico

- Python 3.10
- OpenCV
- MediaPipe
- NumPy

## Licencia

Este proyecto se distribuye bajo licencia MIT.
