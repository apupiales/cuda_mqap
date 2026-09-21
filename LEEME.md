# cuda_mqap — NSGA-II paralelo + Greedy 2-opt adaptado en CUDA para el mQAP

[English](README.md) | **Español**

Implementación en CUDA C del algoritmo evolutivo multiobjetivo **NSGA-II**, combinado con una búsqueda
local **Greedy 2-opt adaptada**, para resolver instancias del **Problema de Asignación Cuadrática
Multiobjetivo** (mQAP, *multiobjective Quadratic Assignment Problem*). La evaluación del fitness, los pasos
de NSGA-II, la selección, la mutación y la búsqueda local se ejecutan en la GPU.

Esta rama contiene la **implementación original**: una sola unidad de traducción, `kernel.cu`, con las
instancias compiladas dentro del programa. En otras ramas hay versiones reescritas y optimizadas, que cargan
las instancias en tiempo de ejecución, incluyen pruebas, corrigen los
[problemas conocidos](#problemas-conocidos-del-código-original) y admiten poblaciones mayores: ver
[Ramas del repositorio](#ramas-del-repositorio).

---

## Índice

1. [Características](#características)
2. [Ramas del repositorio](#ramas-del-repositorio)
3. [El problema: mQAP](#el-problema-mqap)
4. [Cómo funciona el programa](#cómo-funciona-el-programa)
5. [Estructura del repositorio](#estructura-del-repositorio)
6. [Requisitos](#requisitos)
7. [Abrir en Visual Studio 2026 (plug and play)](#abrir-en-visual-studio-2026-plug-and-play)
8. [Configuración](#configuración)
9. [Salida](#salida)
10. [Análisis de resultados](#análisis-de-resultados)
11. [Problemas conocidos del código original](#problemas-conocidos-del-código-original)
12. [Créditos y citas](#créditos-y-citas)
13. [Licencia](#licencia)

---

## Características

1. Creación de la población inicial (permutaciones aleatorias).
2. Cálculo del fitness de cada objetivo.
3. NSGA-II paralelo:
   - 3.1 Matriz de dominancia.
   - 3.2 Dominancia total.
   - 3.3 Frentes de Pareto.
   - 3.4 Rango.
   - 3.5 Fitness de la población ordenado por objetivo (bitonic sort, usado en el cálculo del crowding).
   - 3.6 Crowding distance.
   - 3.7 Población descendiente.
4. Selección por torneo binario.
5. Mutación: intercambio (dos pasadas) y transposición (inversión de un segmento).
6. Búsqueda local Greedy 2-opt adaptada.

Admite instancias de 2 o 3 objetivos y hasta 60 instalaciones.

---

## Ramas del repositorio

El programa original está en esta rama. Las demás ramas mantienen el mismo algoritmo (NSGA-II con un
Greedy 2-opt adaptado sobre el mQAP) pero reescriben su implementación; cada una parte de la anterior.

| Rama | Qué contiene | Diferencias principales frente a esta versión original |
|---|---|---|
| `master` (esta) | Implementación original (2019): todo en `kernel.cu`, con las instancias compiladas dentro del programa, más el proyecto de Visual Studio y esta documentación | — |
| [`develop_with_claude_opus_5`](https://github.com/apupiales/cuda_mqap/tree/develop_with_claude_opus_5) | Reescritura modular y optimización en GPU del mismo algoritmo | Código repartido en `include/`, `src/` y `tests/`; las instancias se leen de los `.dat` en tiempo de ejecución y los parámetros se pasan por línea de comandos; fitness en O(n²) en lugar de tres productos de matrices densas; 2-opt con evaluación incremental en O(n); todo NSGA-II dentro de un bloque por ejecución; tres lanzamientos de kernel por generación sin sincronizar con el host; `--runs` ejecuta ejecuciones independientes de forma concurrente; pruebas automáticas, `--verify` y los [problemas conocidos](#problemas-conocidos-del-código-original) corregidos. KC30-3fl-1rl: 42,4 s → 0,14 s en una RTX 2060 |
| [`develop_p512_single_block`](https://github.com/apupiales/cuda_mqap/tree/develop_p512_single_block) | La anterior, con una supervivencia que no guarda la matriz de dominancia | Población hasta 512 en cualquier GPU (la memoria compartida crece linealmente con P en lugar de cuadráticamente). Con P ≤ 256, resultados idénticos y menor tiempo de GPU |
| [`develop_large_population_multiblock`](https://github.com/apupiales/cuda_mqap/tree/develop_large_population_multiblock) | La versión refactorizada con la supervivencia repartida en varios bloques | Población hasta 65536: lanzamiento cooperativo para los frentes de Pareto y ordenaciones por segmentos (CUB) para el crowding y la selección. Con P ≤ 256 usa el mismo kernel de un bloque, con resultados idénticos |

Las dos últimas ramas existen porque una población mayor explora más el frente de Pareto: en
KC10-2fl-3uni, la fracción media del frente óptimo encontrada por ejecución pasa del 68 % con P = 256 al
82 % con P = 4096 (100 ejecuciones de cada). Su README documenta las mediciones y los límites de cada GPU.

---

## El problema: mQAP

Hay `n` instalaciones (*facilities*) que deben asignarse a `n` ubicaciones (*locations*). Un cromosoma es
una permutación (`short[FACILITIES_LOCATIONS]`). Cada objetivo `k` tiene su propia matriz de flujos `Fk`, y
todos comparten la matriz de distancias `D`. El programa calcula cada objetivo como la traza de un producto
de matrices:

```
f_k = Trace(Fk · X · Dᵀ · Xᵀ)          k = 1..m   (m = 2 o 3)
```

donde `X` es la permutación escrita como matriz binaria `n × n`. Todos los objetivos se minimizan, así que
el resultado es una aproximación del **frente de Pareto**, es decir, del conjunto de soluciones no dominadas.

Las instancias son los conjuntos de prueba KC10, KC20 y KC30 de Knowles y Corne (ver
[Créditos y citas](#créditos-y-citas)) y están en `mQAPData/*.dat`. Para las instancias de diez
instalaciones, `mQAPData/*.PO` contiene los frentes de Pareto óptimos publicados.

---

## Cómo funciona el programa

Todo está en `kernel.cu`. `main` repite el algoritmo genético completo `TIMES` veces. Cada ejecución hace
`ITERATIONS + 1` iteraciones sobre un array de `NSGA2_POPULATION_SIZE = 2 · POPULATION_SIZE` cromosomas
(`Rt` = padres + descendencia):

1. **`parallelPopulationFitnessCalculation`**: construye la matriz binaria de cada cromosoma y ejecuta la
   cadena de productos `multiplicationWithFlowMatrix` → `multiplicationWithTranposedDistanceMatrix` →
   `matrixMultiplication` → `calculateTrace`. Las filas de fitness tienen `OBJECTIVES + 1` columnas: la
   columna extra guarda el índice original para que sobreviva a las ordenaciones.
2. **`parallelNSGA2`**:
   - matriz de dominancia (`get2ObjectivePopulationDominanceMatrix` o `get3ObjectivePopulationDominanceMatrix`; por eso solo se admiten 2 y 3 objetivos);
   - dominancia total, frentes de Pareto y rango;
   - bitonic sort por objetivo (`bitonicSortStep`) y crowding distance;
   - selección de los siguientes `POPULATION_SIZE` padres por frente y crowding distance.
3. **`BinaryTournamentSelection`**: los ganadores se copian en la segunda mitad del array de población.
4. **Variación de la mitad descendiente** `[POPULATION_SIZE, NSGA2_POPULATION_SIZE)`:
   - dos pasadas de `exchangeMutation`;
   - `transpositionMutation`;
   - `greedy2Opt`, cuyo criterio de aceptación se elige al azar entre un único objetivo y el promedio de
     los objetivos.

Al final de cada ejecución, la población final se añade al fichero de resultados (ver [Salida](#salida)).

Convenciones: los arrays duplicados en host y device usan los prefijos `h_` / `d_`. `kernel.cu` contiene
bloques comentados con poblaciones de prueba fijas, que sirvieron para validar el fitness (por ejemplo,
F0 = 228322 y F1 = 193446 para una permutación fija de KC10).

---

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| `kernel.cu` | Todo el programa: kernels, código de host y `main` (una sola unidad de traducción) |
| `general_dev_settings.cu` | `TIMES` (repeticiones de la ejecución completa), `DEV_MODE` y los indicadores de depuración `PRINT_*` |
| `settings_KC*_*fl_*.cu` | Un fichero por instancia: tamaños, población, iteraciones, probabilidades de mutación y las matrices de la instancia en memoria `__constant__` |
| `mQAPData/` | Instancias (`.dat`), frentes de Pareto óptimos (`.PO`) y su procedencia (`README.txt`) |
| `mQAPMetrics/` | Scripts de Node.js: métrica de distancia y gráficos 3D |
| `comparative_results_kcX_datasets.xlsx` | Resultados comparativos de las instancias |
| `cuda_mqap.slnx`, `cuda_mqap.vcxproj` | Solución y proyecto de Visual Studio |
| `cuda_mqap.props`, `cuda_toolkit.props` | Configuración del proyecto y detección de la versión de CUDA instalada |
| `.vsconfig`, `.gitattributes` | Componentes de Visual Studio y finales de línea |

Los `.cu` distintos de `kernel.cu` se incluyen con `#include` y nunca se compilan por separado.

---

## Requisitos

- GPU NVIDIA con *compute capability* ≥ 7.5 (GeForce RTX 20xx o posterior) y un driver actualizado.
- **Visual Studio 2026** con la carga de trabajo *Desarrollo para el escritorio con C++*. Al abrir la
  solución, Visual Studio lee `.vsconfig` y ofrece instalar los componentes que falten.
- **CUDA Toolkit 12.x o 13.x** (≥ 11.8), instalado **después** de Visual Studio para que se añada su
  *Visual Studio Integration*. El proyecto se desarrolló con CUDA 13.4.

---

## Abrir en Visual Studio 2026 (plug and play)

```
git clone https://github.com/apupiales/cuda_mqap.git
cd cuda_mqap
start cuda_mqap.slnx
```

1. Selecciona `Release | x64` y pulsa **F5** (o Ctrl+F5).
2. El programa ejecuta la instancia seleccionada en `kernel.cu` (KC10-2fl-1rl por defecto), imprime la
   población inicial y la solución final, y escribe `result_KCX_Yfl_Z_nsga2_greedy_2opt.txt` en la raíz
   del repositorio (ignorado por git).
3. El ejecutable se genera en `build\x64\<Configuración>\cuda_mqap.exe`.

Nada depende de la máquina donde se creó el proyecto:

| Qué | Cómo se adapta |
|---|---|
| Versión de CUDA | `cuda_toolkit.props` la toma de `CUDA_PATH` (p. ej. `...\CUDA\v12.6` → `CUDA 12.6.props`). Para usar otra versión instalada: `set CudaVersion=12.6` antes de abrir Visual Studio, o `msbuild /p:CudaVersion=12.6` |
| Falta la integración de CUDA | La compilación se detiene con un mensaje que explica cómo arreglarlo |
| Toolset de C++ | `$(DefaultPlatformToolset)` del Visual Studio que lo abre (v145 en VS 2026); SDK de Windows `10.0` (el más reciente instalado) |
| GPU | Código nativo para `sm_75`, `sm_80`, `sm_86` y `sm_89`, más el PTX que el driver compila para GPUs más nuevas (RTX 50xx) |
| Pila | Se reservan 8 MB: `parallelPopulationFitnessCalculation` guarda ~0,8–0,9 MB de arrays en la pila con KC20/KC30, cerca del máximo de 1 MB que Windows reserva por defecto |
| Rutas | Relativas al repositorio; salidas en `build\` (ignorado por git) |

**Línea de comandos** (desde una consola *x64 Native Tools*, porque `nvcc` necesita `cl.exe`):

```
nvcc -O3 -arch=sm_75 kernel.cu -o cuda_mqap.exe
```

---

## Configuración

### Selección de la instancia

`kernel.cu` tiene un bloque de líneas `#include "settings_KC*_*fl_*.cu"` con **una sola** sin comentar.
Para ejecutar otra instancia, comenta la línea activa, descomenta la que quieras y vuelve a compilar.
Cada fichero de configuración define:

- `FACILITIES_LOCATIONS` y `OBJECTIVES` (2 para `2fl`, 3 para `3fl`);
- `POPULATION_SIZE`, que **debe ser potencia de dos** (bitonic sort), e `ITERATIONS`;
- `EXCHANGE_MUTATION_PROBABILITY` y `TRANSPOSITION_MUTATION_PROBABILITY`;
- las matrices de distancias y de flujos, copiadas del `.dat` correspondiente (no se lee nada en tiempo de
  ejecución).

| Fichero de configuración | `POPULATION_SIZE` | `ITERATIONS` |
|---|---|---|
| KC10-2fl-1rl, 3rl, 4rl, 5rl | 64 | 70 |
| KC10-2fl-1uni, 2rl | 16 | 70 |
| KC10-2fl-2uni | 4 | 70 |
| KC10-2fl-3uni | 128 | 25 |
| KC20-2fl-1rl, 1uni, 2uni, 3uni | 64 | 300 |
| KC30-3fl-1rl, 1uni, 2uni | 32 | 70 |

Los comentarios de cabecera de algunos ficheros de configuración citan el `.dat` equivocado, y
`settings_KC20_2fl_1rl.cu` contiene las matrices de otra instancia (ver
[Problemas conocidos](#problemas-conocidos-del-código-original)).

### Salida de depuración

`general_dev_settings.cu` contiene `TIMES`, `DEV_MODE` y los indicadores `PRINT_*`. La mayoría de las
impresiones requieren `DEV_MODE true` **y** el indicador correspondiente.
`PRINT_FIRST_POPULATION_WITH_FITNESS` (activo por defecto) imprime la población inicial con su fitness.

---

## Salida

La población final de cada ejecución (`TIMES` ejecuciones por llamada al programa) se **añade** a
`result_KCX_Yfl_Z_nsga2_greedy_2opt.txt` en el directorio de trabajo, como un literal de diccionario de
Python/JavaScript:

```
{
'0361948752': [5925064, 2282788],
'5134062879': [1665490, 5884156],
...
},
```

La clave es la permutación, con los genes escritos sin separador, y el valor son sus objetivos. Con más de
10 instalaciones la clave es ambigua, porque hay genes de dos dígitos; la salida por consola separa los
genes con espacios. Al final se imprime el tiempo de ejecución (`Time Spent`).

---

## Análisis de resultados

- **`comparative_results_kcX_datasets.xlsx`**: una pestaña por instancia con los frentes obtenidos
  (NSGA-II, NSGA-II + Greedy 2-opt, población inicial y, en KC10, el frente de Pareto óptimo) y su
  gráfico, además de la pestaña *Distance Metric*.
- **`mQAPMetrics/distance_metric_*.js`**: scripts de Node.js con los frentes obtenidos pegados dentro.
  Para cada ejecución calculan la distancia euclídea media de cada solución obtenida al punto más cercano
  del frente `.PO`; después, la media y la desviación típica entre ejecuciones, para NSGA-II y para
  NSGA-II + Greedy 2-opt. Se ejecutan con `node distance_metric_<instancia>.js`.
- **`mQAPMetrics/3D_plot-*.js`**: gráficos 3D de los frentes de 3 objetivos con LightningChart JS
  (`npm install @arction/lcjs @arction/xydata`).

---

## Problemas conocidos del código original

Una revisión de esta versión (con compute-sanitizer, Nsight Systems y comprobaciones de consistencia)
encontró los siguientes problemas. Aquí solo se documentan: **no se han modificado en esta rama**. Las
correcciones mínimas están en el commit `3f3a187`, y la reescritura completa, en la rama
`develop_with_claude_opus_5`.

| # | Problema | Efecto |
|---|---|---|
| B1 | `cudaMalloc(&d_state, sizeof(curandState))` reserva 1 estado, pero `curand_setup` inicializa hasta 8 192 | Escrituras fuera de límites en memoria de GPU. Con CUDA 13.4, **ejecutar sin el Greedy 2-opt produce fitness imposibles** (negativos o de miles de millones); `compute-sanitizer --tool memcheck` lo detecta |
| B2 | `binaryTournament` se lanza con `NSGA2_POPULATION_SIZE` bloques sobre arrays de `POPULATION_SIZE` elementos | Lecturas y escrituras fuera de límites |
| B3 | `settings_KC20_2fl_1rl.cu` contiene las matrices de **KC20-2fl-2rl** | Los resultados etiquetados como KC20-2fl-1rl son de KC20-2fl-2rl |
| B4 | El generador aleatorio del torneo se inicializa con `time(NULL)` en cada generación | Las generaciones que caen en el mismo segundo repiten adversarios |
| B5 | `shufflePopulationGenes` comparte estados curand entre bloques | Números aleatorios correlacionados y barajado sesgado |
| B6 | Crowding: `(unsigned int)HUGE_VALF`, extremo de frente no detectado, posible división por cero | Comportamiento indefinido en casos límite |
| B7 | `cudaMalloc` por generación sin su `cudaFree` | Fugas de memoria (783 reservas en una ejecución de KC10) |
| B8 | El bucle del greedy recorre los pares (i, j) y (j, i) | El doble de trabajo en la búsqueda local |
| B9 | ~0,8–0,9 MB de arrays en la pila del host | Cerca del máximo de 1 MB de Windows (el proyecto reserva 8 MB) |
| B10 | `DEV_MODE \|\| PRINT_*` en lugar de `&&`, y un `sizeof` incorrecto | Salida de depuración inesperada |
| B11 | La última iteración solo conserva el primer frente más filas obsoletas | La población final mezcla soluciones no dominadas y obsoletas |
| B12 | Soluciones fuera del frente actual pueden ganar la ordenación por crowding | Selecciones erróneas en casos raros |

Rendimiento: en una ejecución de KC10-2fl-1rl (70 iteraciones, 3,6 s en una RTX 2060), la GPU solo está
ocupada ~9 % del tiempo. Casi todo el tiempo se va en lanzamientos de kernels, copias de depuración y
sincronizaciones tras cada kernel (87 510 lanzamientos y 80 558 `cudaMemcpy`).

---

## Créditos y citas

### Autor

**Andrés Pupiales Arévalo** — <apupiales@gmail.com> — <https://github.com/apupiales>. Proyecto iniciado en mayo de 2019.

### Conjuntos de prueba (datos de terceros)

Los ficheros de `mQAPData/` **no forman parte del código fuente del programa** y su licencia no los
cubre. Se redistribuyen sin modificar, para uso académico y de investigación, dando crédito a sus autores
(detalles en [`mQAPData/README.txt`](mQAPData/README.txt)):

- **Instancias (`.dat`)**: conjunto de pruebas del mQAP de **Joshua D. Knowles y David W. Corne**,
  generado con sus generadores de instancias `makeQAPuni` y `makeQAPrl` ((C) J. Knowles, 2002). La página
  original ya no está en línea; [copia archivada](https://web.archive.org/web/2019/http://www.cs.bham.ac.uk/~jdk/mQAP/).
- **Frentes de Pareto óptimos (`.PO`)**: enumeración de los óptimos de Pareto de las instancias de diez
  instalaciones, realizada por **Gary Lamont** y publicada en la misma página.
- **Copia utilizada**: [fredizzimo/keyboardlayout](https://github.com/fredizzimo/keyboardlayout/tree/master/tests/mQAPData)
  (Fred Sundvik, 2015). La licencia MIT de ese repositorio no cubre estos datos, cuyos autores son los
  citados arriba.
- **Condiciones**: no se publicó una licencia explícita para las instancias. La página original ofrece
  los generadores como software libre *"for academic or educational use"* y pide contactar con el autor
  para uso comercial; el mismo criterio se aplica a las instancias.

Cita el conjunto de pruebas cuando lo uses:

> J. D. Knowles y D. W. Corne. *Instance Generators and Test Suites for the Multiobjective Quadratic
> Assignment Problem*. En Evolutionary Multi-Criterion Optimization (EMO 2003), Lecture Notes in Computer
> Science, vol. 2632, pp. 295–310. Springer, 2003.

```bibtex
@InProceedings{Knowles2003mQAP,
  author    = {Joshua D. Knowles and David W. Corne},
  title     = {Instance Generators and Test Suites for the Multiobjective Quadratic Assignment Problem},
  booktitle = {Evolutionary Multi-Criterion Optimization (EMO 2003)},
  series    = {Lecture Notes in Computer Science},
  volume    = {2632},
  pages     = {295--310},
  publisher = {Springer},
  year      = {2003}
}
```

Trabajo relacionado de los mismos autores (análisis del paisaje de búsqueda del mQAP):

> J. D. Knowles y D. W. Corne. *Towards Landscape Analyses to Inform the Design of a Hybrid Local
> Search for the Multiobjective Quadratic Assignment Problem*. En A. Abraham, J. Ruiz-del-Solar y
> M. Köppen (eds.), Soft Computing Systems: Design, Management and Applications, pp. 271–279. IOS Press,
> Ámsterdam, 2002.

### Referencias algorítmicas

- La adaptación del criterio del Greedy 2-opt a varios objetivos sigue el trabajo citado en `kernel.cu`:
  <https://arxiv.org/abs/1109.1276>.
- Los scripts de métricas usan [LightningChart JS](https://lightningchart.com/js-charts/) (`@arction/lcjs`)
  para los gráficos 3D.

---

## Licencia

Copyright (C) 2019-2026 Andrés Pupiales Arévalo.

Este programa se distribuye bajo la **GNU General Public License v3** (ver [`LICENSE`](LICENSE)). La
cabecera de `kernel.cu` indica *"version 2 of the License, or (at your option) any later version"*, lo que
permite distribuirlo bajo la versión 3. Se distribuye con la esperanza de que sea útil, pero **sin ninguna
garantía**.

El CUDA Toolkit de NVIDIA (compilador `nvcc`, runtime `cudart` y biblioteca cuRAND) no forma parte de este
repositorio: es software propietario de NVIDIA, distribuido bajo su propia licencia (CUDA EULA). Los datos
de `mQAPData/` son datos de terceros (ver [Conjuntos de prueba](#conjuntos-de-prueba-datos-de-terceros)).
