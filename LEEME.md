# cuda_mqap — NSGA-II + Greedy 2-opt adaptado en CUDA para el mQAP

[English](README.md) | **Español**

Implementación paralela en GPU (CUDA C++) del algoritmo evolutivo multiobjetivo **NSGA-II**,
combinado con una búsqueda local **Greedy 2-opt adaptada**, para resolver instancias del
**Problema de Asignación Cuadrática Multiobjetivo** (mQAP, *multiobjective Quadratic Assignment Problem*).

Todo el algoritmo se ejecuta en la GPU: la evaluación del fitness, la ordenación no dominada, el
crowding distance, la selección, la mutación y la búsqueda local. Cada generación son **3 lanzamientos
de kernel sin sincronización con el host**. Además, se pueden ejecutar **varias ejecuciones
independientes de forma concurrente** en una sola llamada al programa.

---

## Índice

1. [Características](#características)
2. [El problema: mQAP](#el-problema-mqap)
3. [El algoritmo](#el-algoritmo)
4. [Arquitectura del proyecto](#arquitectura-del-proyecto)
5. [Diseño en GPU](#diseño-en-gpu)
6. [Requisitos](#requisitos)
7. [Compilación](#compilación)
8. [Uso](#uso)
9. [Experimentos y métricas](#experimentos-y-métricas)
10. [Pruebas y validación](#pruebas-y-validación)
11. [Rendimiento](#rendimiento)
12. [Mejoras respecto a la versión original](#mejoras-respecto-a-la-versión-original)
13. [Límites del tamaño de población y recursos de la GPU](#límites-del-tamaño-de-población-y-recursos-de-la-gpu)
14. [Limitaciones y trabajo futuro](#limitaciones-y-trabajo-futuro)
15. [Solución de problemas](#solución-de-problemas)
16. [Créditos y licencia](#créditos-y-licencia)

---

## Características

**Algoritmo**
- NSGA-II completo: ordenación no dominada rápida, crowding distance y selección elitista (μ + λ).
- Selección por torneo binario, mutación por intercambio y mutación por transposición (inversión de un segmento).
- Greedy 2-opt adaptado a varios objetivos: en cada generación se elige al azar si el criterio de mejora
  es la suma de todos los objetivos o un único objetivo.
- Instancias de 2 y 3 objetivos (flujos) y hasta 64 instalaciones.

**Rendimiento en GPU**
- Fitness en **O(n²)** por cromosoma (un *warp* por cromosoma, con las matrices en *shared memory*),
  en lugar de tres productos de matrices densas de O(n³).
- NSGA-II completo **en la GPU**: un bloque por ejecución hasta P = 256 (matriz de dominancia empaquetada en
  bits en *shared memory*) y una supervivencia multibloque con lanzamiento cooperativo y ordenaciones por
  segmentos (CUB) para P hasta 65536.
- Greedy 2-opt con **evaluación incremental (delta) en O(n)** de cada intercambio; la búsqueda local de
  toda la descendencia es un único kernel.
- Estados aleatorios Philox persistentes, que se inicializan una sola vez.
- **Ejecuciones independientes en paralelo** (`--runs R`) para aprovechar toda la GPU en las campañas de experimentos.

**Ingeniería**
- Separación estricta host/device: `main.cpp` no contiene código CUDA, y los kernels se exponen mediante funciones lanzadoras.
- Instancias leídas de los ficheros `.dat` en tiempo de ejecución; los parámetros se pasan por línea de comandos.
- Control de errores `CUDA_CHECK`/`CUDA_CHECK_KERNEL` y gestión de memoria RAII (`DeviceBuffer<T>`).
- **Plug and play en Visual Studio 2026**: clonar, abrir `cuda_mqap.slnx` y pulsar F5. La versión de CUDA,
  el toolset de C++ y las arquitecturas de GPU se adaptan a la máquina. También `CMakeLists.txt` con `ctest`.
- Batería de pruebas que compara cada kernel con una implementación independiente en CPU, y la opción
  `--verify`, que valida los resultados de cada ejecución.
- Resultados reproducibles mediante semilla (`--seed`).

---

## El problema: mQAP

Hay `n` instalaciones (*facilities*) que deben asignarse a `n` ubicaciones (*locations*). Una solución es
una permutación `p`, donde `p[i]` es la ubicación de la instalación `i`. Cada objetivo `k` tiene su propia
matriz de flujos `Fk`, y todos comparten la matriz de distancias `D`. Se minimizan simultáneamente los `m` costes:

```
cost_k(p) = Σ_i Σ_j  Fk[i][j] · D[p(i)][p(j)]          k = 1..m   (m = 2 o 3)
```

Esta expresión es equivalente a la formulación matricial `Trace(Fk · X · Dᵀ · Xᵀ)` que usaba la versión
original, donde `X` es la matriz de permutación; las pruebas verifican esa equivalencia. Como los
objetivos están en conflicto, el resultado no es una única solución sino una aproximación del **frente de
Pareto**: el conjunto de soluciones no dominadas.

### Instancias (`mQAPData/`)

Las instancias son el conjunto de pruebas del mQAP de Knowles y Corne
([página archivada](https://web.archive.org/web/2019/http://www.cs.bham.ac.uk/~jdk/mQAP/)), tomadas de la copia de
<https://github.com/fredizzimo/keyboardlayout/tree/master/tests/mQAPData>. Son datos de terceros, no cubiertos
por la licencia de este proyecto: ver [Datos de terceros](#datos-de-terceros).

| Fichero | Contenido |
|---|---|
| `KC<n>-<m>fl-<tipo>.dat` | Cabecera (`facilities = 10 objectives = 2 …` o `facilities: 10 objectives: 2 …`), la matriz de distancias `n×n` y `m` matrices de flujo `n×n` |
| `KC10-2fl-*.PO` | Frente de Pareto óptimo publicado: en cada línea, una permutación en base 1 y sus `m` costes |

`rl` son instancias con distancias y flujos del tipo *real-like*, y `uni` con valores uniformes.
Hay 23 instancias con n = 10, 20 y 30, y los frentes óptimos de las 8 instancias KC10.

---

## El algoritmo

```mermaid
flowchart TD
    A[Población inicial aleatoria<br/>2P permutaciones · Fisher-Yates] --> B[Fitness de las 2P soluciones]
    B --> C{{"Supervivencia NSGA-II (Rt = Pt ∪ Qt)<br/>frentes · crowding · mejores P"}}
    C -->|¿última iteración?| Z[Frente no dominado final]
    C --> D["Reproducción<br/>Pt+1 = supervivientes<br/>Qt+1 = ganadores del torneo binario + mutaciones"]
    D --> E["Greedy 2-opt adaptado sobre Qt+1<br/>(deja el fitness actualizado)"]
    E --> C
```

Cada generación hace lo siguiente:

1. **Supervivencia NSGA-II** sobre las `2P` soluciones de `Rt = Pt ∪ Qt`:
   - *Ordenación no dominada*: rango 1 para el primer frente de Pareto, rango 2 para el siguiente, etc.
   - *Crowding distance* de cada frente. Para cada objetivo se ordenan los miembros del frente; los
     extremos reciben ∞ y los puntos interiores suman `(f[siguiente] − f[anterior]) / (máx − mín)`,
     con el máximo y el mínimo calculados sobre toda la población.
   - Se seleccionan las `P` mejores soluciones por (rango ascendente, crowding descendente).
2. **Reproducción**. Las `P` supervivientes forman `Pt+1`. Para cada una se celebra un **torneo binario**
   contra otra superviviente elegida al azar: gana el rango menor y, en caso de empate, el mayor crowding.
   El ganador se copia y se muta:
   - **Mutación por intercambio**: se intercambian dos genes al azar; se aplica 2 veces.
   - **Mutación por transposición**: se invierte el segmento comprendido entre dos posiciones aleatorias.
3. **Greedy 2-opt adaptado** sobre cada descendiente. Se recorren en orden todos los pares de posiciones
   `(r < s)` y se conserva el intercambio si no empeora el criterio de la generación, elegido al azar para
   cada ejecución y generación: la suma de todos los objetivos o un único objetivo `k`. La idea de adaptar
   el criterio proviene de <https://arxiv.org/ftp/arxiv/papers/1109/1109.1276.pdf>.

Parámetros:

| Parámetro | Dónde | Valor por defecto |
|---|---|---|
| Tamaño de población `P` | `--population` | 64 (potencia de 2 entre 16 y 65536) |
| Generaciones | `--iterations` | 70 |
| Ejecuciones independientes | `--runs` | 1 |
| Semilla | `--seed` | aleatoria (se imprime) |
| Mutaciones por intercambio por hijo | `include/config.h` (`kExchangeMutations`) | 2 |
| Probabilidad de intercambio / transposición | `include/config.h` | 1.0 / 1.0 |

---

## Arquitectura del proyecto

```
cuda_mqap/
├── include/
│   ├── config.h            Límites (n, P, objetivos) y parámetros de los operadores
│   ├── cuda_check.cuh      CUDA_CHECK / CUDA_CHECK_KERNEL
│   ├── device_buffer.cuh   DeviceBuffer<T>: memoria de GPU con RAII
│   ├── device_common.cuh   Funciones __device__ compartidas (coste por warp, delta 2-opt, bitonic sort)
│   ├── instance.h          Struct Instance, loadInstance(), cost() de referencia en CPU
│   ├── kernels.cuh         Declaración de los lanzadores de kernels y del layout de memoria
│   ├── solver.h            SolverOptions, Solution, RunResult, solve()
│   └── survival_workspace.cuh   Búferes de la supervivencia multibloque (se reservan una vez)
├── src/
│   ├── main.cpp            Línea de comandos, fichero de resultados y --verify (solo host)
│   ├── instance.cpp        Parser de los ficheros .dat y validación (incluido el desbordamiento del fitness)
│   ├── solver.cu           Orquestación en el host: reservas, bucle de generaciones y recogida de resultados
│   ├── fitness.cu          Kernel de fitness
│   ├── nsga2.cu            Kernel de supervivencia NSGA-II (un bloque por ejecución, P ≤ 256)
│   ├── nsga2_multiblock.cu Supervivencia NSGA-II repartida en varios bloques (P > 256)
│   ├── operators.cu        RNG, población inicial, torneo y mutaciones
│   └── local_search.cu     Kernel Greedy 2-opt
├── tests/test_kernels.cu   Pruebas de cada kernel contra referencias en CPU
├── scripts/run_experiments.ps1   Campaña de experimentos con los parámetros de cada instancia
├── scripts/run_convergence.ps1   Trazas de cada generación y su análisis (ver más abajo)
├── scripts/analyze_convergence.py   Hipervolumen, estancamiento y cobertura del frente óptimo
├── mQAPData/               Instancias (.dat) y frentes óptimos (.PO)
├── mQAPMetrics/            Scripts Node.js de métricas y gráficos 3D
├── comparative_results_kcX_datasets.xlsx   Resultados comparativos
├── cuda_mqap.slnx, cuda_mqap.vcxproj, test_kernels.vcxproj   Solución y proyectos de Visual Studio
├── cuda_mqap.props, cuda_toolkit.props   Configuración común y detección de la versión de CUDA
├── .vsconfig, .gitattributes             Componentes de Visual Studio y finales de línea
└── CMakeLists.txt
```

**Separación host/device.** El programa se organiza en tres capas:

| Capa | Ficheros | Responsabilidad |
|---|---|---|
| Aplicación (host) | `main.cpp`, `instance.cpp` | Argumentos, lectura de la instancia, escritura de resultados y verificación en CPU |
| Orquestación (host) | `solver.cu` | Reserva de toda la memoria una sola vez, secuencia de lanzamientos y copia final de resultados |
| Kernels (device) | `fitness.cu`, `nsga2.cu`, `operators.cu`, `local_search.cu` | Kernels `template<int OBJ>` (instanciados para 2 y 3 objetivos) y sus lanzadores |

Los kernels viven en el espacio de nombres `mqap::detail`. Fuera de su `.cu` solo se ven los lanzadores
declarados en `kernels.cuh` (`launchFitness`, `launchSurvival`, `launchReproduce`, `launchGreedy2Opt`…),
y cada uno comprueba el lanzamiento con `CUDA_CHECK_KERNEL()`.

---

## Diseño en GPU

### Layout de memoria

Para `R` ejecuciones, población `P`, `n` instalaciones y `OBJ` objetivos:

| Buffer | Tipo y forma | Descripción |
|---|---|---|
| `genes` (×2, doble búfer) | `short [R][2P][n]` | Filas `[0, P)`: supervivientes; filas `[P, 2P)`: descendencia |
| `fitness` (×2) | `unsigned int [R][2P][OBJ]` | Coste de cada objetivo |
| `survivorIndex / Rank / Crowding` | `[R][P]` | Resultado de la supervivencia, ordenado por (rango, −crowding) |
| `rng` | `curandStatePhilox4_32_10_t [R][2P]` | Estados aleatorios persistentes |
| `flow`, `dist` | `int [OBJ][n][n]`, `int [n][n]` | Matrices de la instancia |

Toda la memoria se reserva **una vez** con `DeviceBuffer<T>` y se libera automáticamente. Entre
generaciones solo se intercambian los punteros del doble búfer.

### Kernels

| Kernel | Grid × bloque | Paralelismo | Técnicas |
|---|---|---|---|
| `fitnessKernel<OBJ>` | `(⌈2P/4⌉, R)` × 128 | 1 warp por cromosoma | `F` y `D` en *shared memory* (carga coalescente); lecturas de `F` consecutivas por carril (sin conflictos de banco); reducción con `__shfl_down_sync` |
| `survivalKernel<OBJ>` (P ≤ 256) | `R` × `2P` | 1 bloque por ejecución, 1 hilo por individuo | Dominancia en bits (`2P × 2P/32` palabras), frentes con `__ballot_sync` + `__popc`, bitonic sort de claves de 64 bits `(rango, fitness)` y `(rango, −crowding)` en *shared memory* |
| `reproduceKernel<OBJ>` | `(⌈P/128⌉, R)` × 128 | 1 hilo por descendiente | Estado Philox en registros; torneo, mutaciones y copia en un solo paso |
| `greedy2OptKernel<OBJ>` | `(⌈P/4⌉, R)` × 128 | 1 warp por descendiente | Matrices en *shared memory*; delta O(n) repartido entre los 32 carriles; criterio uniforme en el warp (sin divergencia) |
| `initPopulationKernel` | `(⌈2P/128⌉, R)` × 128 | 1 hilo por cromosoma | Fisher-Yates sin sesgo |
| `rngInitKernel` | `⌈R·2P/128⌉` × 128 | 1 hilo por estado | Una subsecuencia Philox independiente por hilo |
| Supervivencia multibloque (P > 256) | `(⌈2P/256⌉, R)` × 256, más un lanzamiento cooperativo | 1 hilo por individuo en toda la malla | Conteo de dominadores con teselas de fitness en *shared memory*, pelado de frentes con `grid.sync()`, crowding y selección con ordenaciones por segmentos (CUB) |

*Shared memory* por bloque:
- **Fitness y 2-opt:** `(OBJ + 1)·n²·4 + 4·n·2` bytes, por ejemplo 14,6 KB para n = 30 y 3 objetivos.
  Si hace falta más de 48 KB se solicita automáticamente el máximo *opt-in* del dispositivo
  (`cudaFuncAttributeMaxDynamicSharedMemorySize`), lo que permite n = 60 con 3 objetivos en Turing.
- **Supervivencia:** hasta ~46 KB con P = 256 (camino de un bloque). El camino multibloque (P > 256) solo usa
  teselas de fitness de unos 3 KB; ver [Límites del tamaño de población](#límites-del-tamaño-de-población-y-recursos-de-la-gpu).

### Evaluación incremental del 2-opt

Intercambiar las posiciones `r` y `s` de `p` solo modifica los términos del coste en los que aparecen `r` o `s`:

```
Δ(r,s) = (F_rr − F_ss)(D_{ps ps} − D_{pr pr}) + (F_rs − F_sr)(D_{ps pr} − D_{pr ps})
       + Σ_{k≠r,s} [ (F_kr − F_ks)(D_{pk ps} − D_{pk pr}) + (F_rk − F_sk)(D_{ps pk} − D_{pr pk}) ]
```

Cada carril del warp calcula una parte del sumatorio y el resultado se reduce con `__shfl_down_sync`.
Así, cada una de las `n(n−1)/2` evaluaciones cuesta O(n) en lugar de recalcular el fitness completo.
Los acumuladores son de 64 bits.

### Sincronización

- Todos los kernels se lanzan en el *default stream*, que ya garantiza el orden entre ellos, así que no se
  usa `cudaDeviceSynchronize` durante la ejecución.
- El host solo espera al final (`cudaEventSynchronize`), para medir el tiempo y copiar los resultados.
- Dentro de los kernels, `__syncthreads()` solo separa fases que comparten *shared memory*, y `__syncwarp()`
  hace visible a todo el warp el intercambio aplicado en el 2-opt.
- En Debug, `MQAP_SYNC_CHECK` hace que `CUDA_CHECK_KERNEL()` sincronice después de cada kernel, de modo
  que un error de ejecución se reporta en el lanzamiento que lo causó.

---

## Requisitos

- GPU NVIDIA con *compute capability* ≥ 7.5 (GeForce RTX 20xx o posterior) y un driver actualizado.
- **Visual Studio 2026** con la carga de trabajo *Desarrollo para el escritorio con C++*. Al abrir la
  solución, Visual Studio lee `.vsconfig` y ofrece instalar los componentes que falten.
- **CUDA Toolkit 12.x o 13.x** (≥ 11.8), instalado **después** de Visual Studio para que se añada su
  *Visual Studio Integration*. El proyecto detecta automáticamente la versión instalada; se desarrolló
  con CUDA 13.4.
- Alternativa sin el IDE: CMake ≥ 3.24 + Ninja (incluidos con Visual Studio).

---

## Compilación

### Abrir en Visual Studio 2026 (plug and play)

```
git clone https://github.com/apupiales/cuda_mqap.git
cd cuda_mqap
git checkout develop_with_claude_opus_5
start cuda_mqap.slnx
```

1. Visual Studio abre la solución con sus dos proyectos: `cuda_mqap` (el programa, proyecto de inicio) y
   `test_kernels` (las pruebas).
2. Selecciona `Release | x64` y pulsa **F5** (o Ctrl+F5). El programa se ejecuta con
   `mQAPData\KC10-2fl-1rl.dat --verify`, usando la raíz del repositorio como directorio de trabajo. Los
   argumentos se cambian en *Proyecto → Propiedades → Depuración*.
3. Para ejecutar las pruebas: clic derecho en `test_kernels` → *Establecer como proyecto de inicio* → Ctrl+F5.
4. Los ejecutables se generan en `build\x64\<Configuración>\`.

Nada depende de la máquina donde se creó el proyecto:

| Qué | Cómo se adapta |
|---|---|
| Versión de CUDA | `cuda_toolkit.props` la toma de `CUDA_PATH` (p. ej. `...\CUDA\v12.6` → `CUDA 12.6.props`). Para usar otra versión instalada: `set CudaVersion=12.6` antes de abrir VS, o `msbuild /p:CudaVersion=12.6` |
| Falta la integración de CUDA | La compilación se detiene con un mensaje que explica cómo arreglarlo (en lugar de "tipo de elemento CudaCompile desconocido") |
| Toolset de C++ | `$(DefaultPlatformToolset)` del Visual Studio que lo abre (v145 en VS 2026); SDK de Windows `10.0` (el más reciente instalado) |
| GPU | Código nativo para `sm_75`, `sm_80`, `sm_86` y `sm_89`, más el PTX que el driver compila para GPUs más nuevas (RTX 50xx) |
| Rutas | Todas relativas al repositorio (`$(MSBuildThisFileDirectory)`); salidas en `build\` (ignorado por git) |
| Finales de línea | `.gitattributes` mantiene CRLF en los ficheros de Visual Studio |

La configuración común está en `cuda_mqap.props`: C++17, `/W4`, `-lineinfo` en Release, y `-G` más
`MQAP_SYNC_CHECK` en Debug.

### CMake

```
cmake -S . -B build/cmake -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake
ctest --test-dir build/cmake --output-on-failure
```

Por defecto compila `sm_75`, `sm_80`, `sm_86` y `sm_89` más PTX; para compilar solo para tu GPU usa
`-DCMAKE_CUDA_ARCHITECTURES=native`. En Visual Studio también sirve *Archivo → Abrir → Carpeta*.

### nvcc directo

Desde una consola *x64 Native Tools*:

```
nvcc -O3 -arch=sm_75 -std=c++17 -Iinclude src\main.cpp src\instance.cpp src\solver.cu src\fitness.cu ^
     src\nsga2.cu src\nsga2_multiblock.cu src\operators.cu src\local_search.cu -o cuda_mqap.exe
```

---

## Uso

```
cuda_mqap <instance.dat> [opciones]
  --population P   tamaño de población, potencia de 2 en [16, 65536] (defecto 64)
  --iterations N   generaciones (defecto 70)
  --runs R         ejecuciones independientes concurrentes (defecto 1)
  --seed S         semilla (defecto: aleatoria, se imprime en la salida)
  --output FILE    fichero de resultados, en modo append (defecto result_<instancia>_nsga2_greedy_2opt.txt)
  --trace FILE     escribe el frente de cada generacion en FILE (CSV, se sobrescribe)
  --trace-max N    puntos guardados por ejecucion y generacion en la traza (defecto 4096)
  --trace-every K  guarda el frente cada K generaciones, mas la ultima (defecto 1)
  --verify         verifica las poblaciones finales en CPU
  --quiet          no imprime las soluciones finales
```

Ejemplos:

```
:: Una ejecución con verificación
build\x64\Release\cuda_mqap.exe mQAPData\KC10-2fl-1rl.dat --verify

:: 30 ejecuciones independientes en paralelo, reproducibles
build\x64\Release\cuda_mqap.exe mQAPData\KC20-2fl-1rl.dat --iterations 300 --runs 30 --seed 2026 --quiet

:: Instancia de 3 objetivos
build\x64\Release\cuda_mqap.exe mQAPData\KC30-3fl-1rl.dat --population 32 --runs 10
```

Salida por consola (resumida):

```
Instance KC10-2fl-1rl: n = 10, objectives = 2 | population = 64, iterations = 70, runs = 1, seed = 42

FINAL SOLUTION (run 0, 64 non-dominated)
0 3 6 1 9 4 8 7 5 2 5925064 2282788
5 1 3 4 0 6 2 8 7 9 1665490 5884156
5 1 6 3 0 2 8 9 7 4 1869616 4670952
...
Verification: OK

Results appended to result_KC10-2fl-1rl_nsga2_greedy_2opt.txt
Time Spent: 0.148970 s (GPU 13.494 ms)
```

### Fichero de resultados

Mantiene el formato original, así que los scripts de `mQAPMetrics` siguen funcionando. Cada ejecución
añade un bloque con las soluciones no dominadas (rango 1) de la población final: la clave es la
permutación y el valor, sus costes.

```
{
'0361948752': [5925064, 2282788],
'5134062879': [1665490, 5884156],
'5163028974': [1869616, 4670952],
},
```

La población final suele contener la misma solución muchas veces, porque converge y los supervivientes
son copias unos de otros. **El frente se escribe sin repeticiones:** cada solución no dominada distinta
aparece una vez, en el orden que le da NSGA-II. La población completa, con sus repeticiones, es la que
comprueba `--verify` y la que constituye la población final de la ejecución.

### Verificación (`--verify`)

Recalcula en CPU, de forma independiente, cada solución de la población final de cada ejecución:
- que la permutación sea válida;
- que el fitness coincida exactamente con `cost()` en CPU;
- que el rango 1 corresponda exactamente a las soluciones no dominadas (y que toda solución con rango > 1
  esté dominada por alguna superviviente).

Si algo falla, el código de salida es 1.

---

## Experimentos y métricas

`scripts/run_experiments.ps1` ejecuta la campaña de `comparative_results_kcX_datasets.xlsx` con la
población y las iteraciones que usaba cada instancia en la versión original:

```
.\scripts\run_experiments.ps1 -Runs 30                              # todas las instancias
.\scripts\run_experiments.ps1 -Runs 10 -Seed 2026 -Instances KC10-2fl-1rl,KC30-3fl-1rl
```

### Cuántas generaciones necesita cada instancia (`--trace`)

`--trace FICHERO` escribe un CSV con `run,generation,f1,f2[,f3]`: las soluciones no dominadas distintas
de **cada** generación de cada ejecución, desde la supervivencia de la población inicial (generación 0)
hasta el frente final. Copia los supervivientes al host una vez por generación, así que sincroniza con el
dispositivo y **el tiempo de una ejecución con traza no es comparable** con el de una normal; la copia es
despreciable frente a una generación con población grande, y se nota con población pequeña.

`scripts/run_convergence.ps1` graba las trazas y las analiza con `scripts/analyze_convergence.py`, que
informa, por instancia:

| Indicador | Significado |
|---|---|
| `t_stall` | Primera generación tras la cual el hipervolumen crece menos de `--epsilon` (relativo) durante `--patience` generaciones. La supervivencia es elitista, así que el hipervolumen solo puede crecer: una curva plana es estancamiento real, no ruido |
| `t_final` | Primera generación cuyo frente ya es igual al último. No necesita datos de referencia y dice cuándo deja de encontrarse algo nuevo |
| `t_optimum` | Primera generación que cubre el frente `.PO` publicado. Solo lo tienen las instancias KC10 |
| `coverage` | Fracción del frente óptimo encontrada al final |

El punto de referencia del hipervolumen es fijo para todo el fichero (la esquina peor de la generación 0),
de modo que las generaciones y las ejecuciones son comparables entre sí. Los tres números de generación son
variables aleatorias —cada ejecución estanca en un punto distinto—, así que el resumen da la mediana y el
percentil 90 sobre las ejecuciones, nunca un único valor.

```
.\scripts\run_convergence.ps1 -Population 1024 -Iterations 200 -Runs 30
.\scripts\run_convergence.ps1 -Population 4096 -Iterations 300 -Instances KC10-2fl-1rl
python scripts\analyze_convergence.py results\convergence\*.csv --patience 30 --epsilon 1e-5
```

`--iterations` tiene que estar claramente por encima del estancamiento esperado o la medición informará
del tope en su lugar; el análisis avisa cuando una ejecución sigue mejorando en la última generación. Las
curvas por generación se escriben en `results/convergence/curves/<instancia>_curve.csv` (hipervolumen
mediano, su fracción del valor final y el tamaño mediano del frente), listas para graficar.

El número de generaciones depende mucho de la población, así que la respuesta útil es el coste total: el
tiempo de una generación se conoce para cada P (ver [Rendimiento](#rendimiento)), de modo que
`generaciones × tiempo por generación` indica qué pareja (P, generaciones) alcanza antes el objetivo.

#### Resultados de la campaña (RTX 2060, 2026-09-21)

Tres poblaciones, subiendo el tope de `--iterations` hasta que el estancamiento fuera observable:
P = 1024 con 30 ejecuciones y tope de 2000 generaciones, P = 16384 con 10 ejecuciones y tope de 300 (1500
y 5 ejecuciones en las instancias de 3 objetivos), y P = 65536 —el máximo de esta rama— con 5 ejecuciones,
tope de 300 y `--trace-every 5`, así que sus valores tienen una resolución de 5 generaciones. `t_stall` es
la mediana sobre las ejecuciones, y la cobertura es la fracción del frente óptimo publicado encontrada al
final con P = 1024.

| Instancia | `t_stall` P = 1024 | P = 16384 | P = 65536 | El frente deja de cambiar (P = 1024) | Frente óptimo | Cobertura |
|---|---|---|---|---|---|---|
| KC10-2fl-2uni | **1** | **1** | ≤ 5 | 1 | 30/30 ejecuciones en la generación 1 | 100 % |
| KC10-2fl-2rl | **6** | **4,5** | **5** | 9 | 30/30 ejecuciones en la generación 9 | 100 % |
| KC10-2fl-1uni | **6,5** | **5,5** | **10** | 6,5 | nunca | 92,3 % |
| KC10-2fl-3rl | **13** | **6,5** | **10** | 845 | nunca | 72,7 % |
| KC10-2fl-1rl | **14** | **5,5** | **10** | 53 | nunca | 79,3 % |
| KC10-2fl-4rl | **16,5** | **6** | **10** | 320 | nunca | 73,6 % |
| KC10-2fl-5rl | **29** | **20** | **10** | 1054 | nunca | 78,6 % |
| KC10-2fl-3uni | **52** | **22,5** | **10** | 1182 | nunca | 86,3 % |
| KC20-2fl-2uni | **46,5** | **28,5** | **40** | 65 | — | — |
| KC20-2fl-1rl | **98** | **55,5** | **45** | 1817 | — | — |
| KC20-2fl-1uni | **185** | **100** | **80** | 1892 | — | — |
| KC20-2fl-3uni | **373** | **144** | **170** | 1987 | — | — |
| KC30-3fl-2uni | **550** | **360** | sin medir | ~tope | — | — |
| KC30-3fl-1rl | **1476** | > 5000 | sin medir | ~tope | — | — |
| KC30-3fl-1uni | **1886** | > 5000 | sin medir | ~tope | — | — |

Las instancias de 3 objetivos no se ejecutaron con P = 65536: a ese tamaño una generación de KC30 cuesta
unos 280 ms y necesitan miles, así que la campaña serían horas por instancia.

Lo que dice la campaña:

- **Las 70 generaciones heredadas de la versión original encajan en las instancias KC10** (estancan entre 1
  y 52), se quedan **cortas en KC20** (46 a 373) y **claramente cortas en KC30** (550 a 1886, y dos de
  ellas no estancan ni en 2000).
- **`t_stall` no depende del tope**: multiplicarlo por cuatro (500 → 2000) no movió los valores de KC10 ni
  de KC20, así que son la respuesta real y no un artefacto del presupuesto.
- **`t_final` no converge** (845, 1054, 1182, 1817, 1987…): el frente sigue admitiendo micro-refinamientos
  casi indefinidamente, sin ganancia medible. Para decidir cuántas generaciones usar, el indicador es
  `t_stall`.
- **Más generaciones casi no compran calidad.** Pasar de 500 a 2000 generaciones movió la cobertura entre
  0,0 y 3,8 puntos.
- **Más población recorta las generaciones, pero no sale a cuenta en tiempo.** Las instancias KC10
  estancan en 10 generaciones o menos con P = 65536, frente a 1 a 52 con P = 1024, pero cada generación
  cuesta 86 ms en lugar de 0,25 ms por ejecución. Llegar a la meseta de KC10-2fl-1rl cuesta 3,5 ms por
  ejecución con P = 1024 y 860 ms con P = 65536: unas 245 veces más para el mismo 79,3 % del frente
  óptimo.
- **Hay un techo de cobertura que no rompe ni la población ni las generaciones**: KC10-2fl-1rl se queda en
  el 79,3 % con P = 1024, con P = 16384 y con P = 65536 (46 de sus 58 puntos óptimos). Multiplicar por 64
  la población mueve las demás instancias menos de tres puntos (KC10-2fl-5rl 78,6 % → 81,2 %, KC10-2fl-3uni
  86,3 % → 87,4 %, KC10-2fl-4rl 73,6 % → 74,3 %). La población sí manda en el rango bajo —de P = 64 (38 de
  58) a P = 1024— y a partir de ahí se aplana. Lo que queda es el algoritmo: esta combinación de NSGA-II
  con el greedy 2-opt converge a un subconjunto del frente óptimo.

**Las dos instancias de 3 objetivos marcadas `> 5000` no se estancan con P = 16384.** Una ejecución con
ese tope (5 ejecuciones, `--trace-every 25`, 16 minutos cada una) muestra una asíntota en lugar de una
parada. Hipervolumen como fracción del valor que alcanza en la generación 5000, y tamaño del frente:

| Generación | 200 | 400 | 800 | 1600 | 2400 | 4000 | 5000 |
|---|---|---|---|---|---|---|---|
| KC30-3fl-1rl | 97,0 % · 2252 puntos | 97,7 % · 3156 | 98,6 % · 4243 | 99,2 % · 5567 | 99,5 % · 6323 | 99,9 % · 7444 | 100 % · 7855 |
| KC30-3fl-1uni | sin muestra | 95,6 % · 931 puntos | 97,2 % · 1332 | 98,5 % · 1716 | 99,2 % · 1956 | 99,8 % · 2187 | 100 % · 2306 |

La ganancia útil llega muy pronto —el 97 % del resultado de 5000 generaciones en la generación 200 de
KC30-3fl-1rl— y lo que sigue es refinamiento lento con un frente que no deja de crecer. Para esas dos, el
número de generaciones es una decisión de presupuesto, no una medición. Con P = 1024 las mismas instancias
sí estancan, en 1476 y 1886, porque su meseta es más baja.

Salvedades de la medición: con 3 objetivos y población grande casi toda la población es no dominada, así
que esas trazas se grabaron con `--trace-every 10` o `25`, y calcular el hipervolumen de frentes de varios
miles de puntos es lo que limita la resolución de la curva. El análisis imprime la ventana que acaba
usando, porque la prueba de estancamiento pasa a exigir que no haya crecimiento en ella, y `--hv-runs N`
limita el hipervolumen a las primeras N ejecuciones: la tabla anterior sale de `--hv-runs 1 --hv-every 8`,
que da cinco veces más resolución al mismo coste (26 muestras en lugar de 3). Los demás indicadores usan
siempre todas las ejecuciones.

| Instancias | P | Generaciones |
|---|---|---|
| KC10-2fl-1rl, 3rl, 4rl, 5rl | 64 | 70 |
| KC10-2fl-1uni, 2rl, 2uni ¹ | 16 | 70 |
| KC10-2fl-3uni | 128 | 25 |
| KC20-2fl-1rl, 1uni, 2uni, 3uni | 64 | 300 |
| KC30-3fl-1rl, 1uni, 2uni | 32 | 70 |

¹ KC10-2fl-2uni usaba P = 4; ahora el mínimo es 16.

Los resultados se guardan en `results\result_<instancia>_nsga2_greedy_2opt.txt`. En la RTX 2060, la
campaña completa (15 instancias × 3 ejecuciones) tarda unos 2 segundos.

**Métricas (`mQAPMetrics/`)** — scripts de Node.js que contienen los frentes obtenidos, copiados de los ficheros de resultados:
- `distance_metric_*.js`: distancia generacional, es decir, la media de la distancia euclídea de cada
  solución obtenida al punto más cercano del frente óptimo `.PO`. Reporta la media y la desviación típica
  entre ejecuciones, para NSGA-II y para NSGA-II + Greedy 2-opt.
- `3D_plot-*.js`: gráficos 3D de los frentes de las instancias de 3 objetivos, con LightningChart JS
  (`@arction/lcjs`).

### Resultados en el libro de Excel

El 2026-09-19 se añadieron a `comparative_results_kcX_datasets.xlsx` los resultados de esta versión:

- **Pestañas de instancia (KC10-\*, KC20-\*):** cada pestaña tiene un bloque nuevo a la derecha de los
  existentes, con 20 o 10 genes y 2 objetivos por fila, y una serie **verde** en su gráfico:
  *"CUDA NSGA-II Paralelo + Greedy 2opt, N iteraciones (optimizado con claude)"*. Se usaron la población y
  las iteraciones de cada pestaña, con `--verify`.
  - En las KC10 se muestra la primera de 100 ejecuciones concurrentes; en las KC20, una única ejecución.
  - Debajo de cada bloque hay una nota con el comando, la semilla y el tiempo.
  - KC10-2fl-2uni se ejecutó con P = 16 (las series originales usaron P = 2).
- **Distance Metric:** columnas F–G (media y desviación típica) y columna H de la segunda tabla, con la
  distancia gama de esta versión sobre 100 ejecuciones por instancia KC10. Se calcula igual que
  `mQAPMetrics/distance_metric_*.js` y se trunca a 2 decimales, como los valores existentes.

**Población máxima de la rama (2026-09-21).** Los mismos 12 experimentos se repitieron con
`--population 65536`, el tope de esta rama, manteniendo las iteraciones de cada pestaña (70, 30 o 25 en
KC10; 300 en KC20), una ejecución, semilla 20260920 y `--verify` OK. Cada pestaña tiene un segundo bloque
nuevo y una serie **roja** en su gráfico: *"CUDA NSGA-II Paralelo + Greedy 2opt, N iteraciones (optimizado
con claude, poblacion 65536)"*.

Con esa población toda la población final es no dominada, así que el frente que escribe el programa tiene
65 536 filas, de las cuales solo entre 1 y 212 son soluciones distintas; el bloque y la serie guardan las
distintas, porque las repeticiones dibujarían los mismos puntos. La nota bajo cada bloque recoge el
comando, la semilla, el número de puntos distintos y el tiempo de la ejecución.

Calidad frente al frente óptimo publicado (`.PO`), junto a la serie verde de la misma rama. *Encontrados*
cuenta cuántos puntos del óptimo reproduce exactamente la ejecución; la gama es la distancia calculada
igual que `mQAPMetrics/distance_metric_*.js` (menor es mejor):

| Instancia | Puntos `.PO` | Verde: encontrados | Verde: gama | Roja: encontrados | Roja: gama |
|---|---|---|---|---|---|
| KC10-2fl-1rl | 58 | 38 | 685,46 | **46** | **568,78** |
| KC10-2fl-1uni | 13 | 7 | 241,93 | **12** | **0,00** |
| KC10-2fl-2rl | 15 | 12 | 0,00 | **15** | 0,00 |
| KC10-2fl-2uni | 1 | 1 | 0,00 | 1 | 0,00 |
| KC10-2fl-3rl | 55 | 28 | 24 466,49 | **40** | **17 757,94** |
| KC10-2fl-3uni | 130 | 66 | 420,75 | **114** | **11,47** |
| KC10-2fl-4rl | 53 | 29 | 7992,15 | **40** | **3793,44** |
| KC10-2fl-5rl | 49 | 26 | 24 552,85 | **39** | **2629,05** |

Las instancias KC20 no tienen frente publicado; sus series rojas tienen 86, 68, 8 y 212 puntos distintos
(1rl, 1uni, 2uni y 3uni). Cada permutación graficada se comprobó en el host: su coste recalculado coincide
con el fitness que escribió el programa, y cada frente es no dominado.

Tiempos en la RTX 2060: 6,6–6,8 s por pestaña KC10 de 70 generaciones y 38–39 s por pestaña KC20 de 300;
201 s las doce.

**Distance Metric.** Las columnas H e I de la primera tabla (media y desviación típica) y la columna I de
la segunda contienen la distancia gama de esta población, medida con el mismo protocolo que las columnas
verdes: **100 ejecuciones por instancia KC10** con las iteraciones de su pestaña, `--seed 20260921`, y la
distancia calculada igual que `mQAPMetrics/distance_metric_*.js` (por ejecución, la media sobre sus
permutaciones únicas de la distancia al punto más cercano del frente `.PO`; después, media y desviación
típica entre ejecuciones). La nota de A11 recoge el comando. Media / desviación típica, verde frente a roja:

| Instancia | Verde (población de la pestaña) | Roja (P = 65536) |
|---|---|---|
| KC10-2fl-1rl | 850,56 / 656,20 | 557,92 / 56,58 |
| KC10-2fl-1uni | 192,02 / 342,12 | 0,00 / 0,00 |
| KC10-2fl-2rl | 10.941,55 / 7.430,27 | 0,00 / 0,00 |
| KC10-2fl-2uni | 532,56 / 1.806,81 | 0,00 / 0,00 |
| KC10-2fl-3rl | 20.531,69 / 2.896,06 | 16.132,53 / 2.893,64 |
| KC10-2fl-3uni | 376,23 / 77,04 | 12,51 / 5,86 |
| KC10-2fl-4rl | 7.515,98 / 2.082,63 | 3.001,91 / 1.277,88 |
| KC10-2fl-5rl | 26.891,28 / 10.070,77 | 3.409,13 / 921,20 |

Una gama de 0,00 significa que **todas las soluciones encontradas en cada una de las 100 ejecuciones están
exactamente sobre el frente óptimo publicado**, no que se haya encontrado el frente entero: la serie roja
de KC10-2fl-1uni tiene 12 de sus 13 puntos. La ejecución que se dibuja en los gráficos es una ejecución
única aparte, con semilla 20260920; esta tabla compara los lotes de 100 ejecuciones.

**Una semilla para el lote, un flujo aleatorio por ejecución.** `--seed` no repite la misma aleatoriedad en
todas las ejecuciones. `curand_init(seed, id, 0, ...)`, en `src/operators.cu`, da a cada uno de los
`runs × 2P` estados su propia subsecuencia de Philox, así que la ejecución *r* saca sus números del tramo
`[r·2P, (r+1)·2P)` y dos ejecuciones nunca comparten números: con la misma semilla, tres ejecuciones de
P = 16 y cero generaciones ya dan tres poblaciones distintas. Lo que se repite entre ejecuciones es la
convergencia, no la aleatoriedad. Contando los frentes distintos de las 100 ejecuciones de cada lote:

| Instancia | Frentes distintos / 100 | Tamaños |
|---|---|---|
| KC10-2fl-3uni | 60 | 113–119 puntos |
| KC10-2fl-1rl | 21 | 47–50 puntos |
| KC10-2fl-1uni | 2 | 12 y 13 puntos |

En KC10-2fl-1rl, 64 de las 100 ejecuciones acaban exactamente en el mismo frente de 47 puntos, porque con
P = 65536 la búsqueda converge a él; en KC10-2fl-1uni uno de los dos frentes tiene 12 de los 13 puntos
óptimos publicados y el otro los 13. Por eso también la desviación típica vale 0,00 en 1uni, 2rl y 2uni:
no porque las ejecuciones sean iguales, sino porque en todas ellas cada solución encontrada está sobre el
frente óptimo, de modo que la distancia es 0 en todas.

Una semilla fija mantiene el lote reproducible: repetir el comando de la nota de A11 devuelve exactamente
las cifras de la tabla. Una semilla tomada del reloj no añadiría independencia entre ejecuciones —ya la
tienen— y sí se perdería eso. Lo que tampoco responde un timestamp es si el resultado depende de la
semilla concreta; eso se comprueba repitiendo el lote con una segunda semilla fija y comparando las medias.

**Calidad frente al Greedy 2-opt original.** Los resultados son mixtos:

| Instancia | Métrica | Original | Esta versión |
|---|---|---|---|
| KC10-2fl-1rl | distancia gama (menor es mejor), 100 ejecuciones | 1.484,66 | **850,56** |
| KC10-2fl-3rl | ídem | 22.541,34 | **20.531,69** |
| KC10-2fl-4rl | ídem | 12.399,20 | **7.515,98** |
| KC10-2fl-5rl | ídem | 32.418,89 | **26.891,28** |
| KC10-2fl-3uni | ídem | 381,15 | 376,23 |
| KC10-2fl-1uni | ídem | **79,32** | 192,02 |
| KC10-2fl-2rl | ídem | **4.451,70** | 10.941,55 |
| KC10-2fl-2uni | ídem (P distinto, no comparable) | 1.346,43 | 532,56 |
| KC20-2fl-1uni | hipervolumen, 1 ejecución (mayor es mejor) | 3,5249·10¹⁰ | 3,5253·10¹⁰ |
| KC20-2fl-1rl | ídem | **6,3518·10¹³** | 6,3061·10¹³ (−0,7 %) |
| KC20-2fl-2uni | ídem | **8,4911·10⁹** | 7,8799·10⁹ (−7,2 %) |
| KC20-2fl-3uni | ídem | **7,6649·10¹⁰** | 7,4938·10¹⁰ (−2,2 %) |

Las cifras de las KC20 salen de una sola ejecución por versión, así que no permiten conclusiones estadísticas.

**Correcciones del libro (2026-09-19)**
- **KC20-2fl-3uni:** las series NSGA-II, Greedy 2opt y la serie oculta del óptimo de Pareto del gráfico
  apuntaban a la pestaña KC20-2fl-1uni, y la celda A1 decía "KC20-2fl-1uni Pareto Optimal". Ahora usan los
  datos de su propia pestaña.
- **KC20-2fl-1rl:** las series originales contenían resultados de la instancia **KC20-2fl-2rl**, porque el
  antiguo `settings_KC20_2fl_1rl.cu` tenía esas matrices (bug B3). Se volvieron a ejecutar con la instancia
  correcta, P = 64 y 300 iteraciones, usando el código original (`kernel.cu` de `ec882da`) con **solo** los
  arreglos de memoria B1 y B2:
  - NSGA-II: la llamada a `greedy2Opt` comentada (2,8 s);
  - NSGA-II + Greedy 2opt: 104,5 s;
  - población inicial: la de la ejecución con Greedy.

  Las 320 filas de la pestaña tienen un fitness igual a su coste en KC20-2fl-1rl. Las notas están en X67 y
  CS67, y los datos antiguos siguen en el historial de git.
- **Datos de 2019:** en las 12 pestañas, cada fila de los bloques NSGA-II, Greedy y población inicial
  tiene exactamente el coste de su permutación, así que los resultados originales son internamente
  coherentes.
- **Pendiente:** en la segunda columna de Distance Metric, el valor de NSGA-II para KC10-2fl-2uni
  (27.629,49) no coincide con los datos actuales de `mQAPMetrics/distance_metric_KC10_2fl_2uni.js`
  (15.195,66).

> **Aviso sobre el código original.** Con CUDA 13.4, el `kernel.cu` de `ec882da` ejecutado sin el Greedy
> 2-opt produce fitness imposibles (negativos o de miles de millones). La causa son las escrituras fuera
> de límites de `curand_setup` (bug B1), que corrompen los buffers de NSGA-II. Si necesitas reproducir la
> versión original, usa el commit `3f3a187` o aplica al menos los arreglos B1 y B2, y compruébalo con
> `compute-sanitizer --tool memcheck`.

---

## Pruebas y validación

`test_kernels` (proyecto `test_kernels` en Visual Studio, o `ctest`) compara cada kernel con una
implementación independiente en CPU:

| Prueba | Qué verifica |
|---|---|
| Carga de instancias | Los 23 `.dat` se leen (en ambos formatos de cabecera) y `n`/`m` coinciden con el nombre del fichero |
| Frentes óptimos `.PO` | Las **374 soluciones óptimas publicadas** tienen exactamente su coste publicado, tanto en CPU como en GPU |
| Fitness | El kernel coincide con la `Trace(F·X·Dᵀ·Xᵀ)` literal de la versión original en KC10, KC20 y KC30, con varias ejecuciones |
| Supervivencia NSGA-II | Los rangos, el crowding y la selección coinciden con un NSGA-II en CPU para P = 16, 64 y 256, con 2 y 3 objetivos |
| Supervivencia multibloque | La misma comprobación forzando el camino multibloque con P = 16, 64 y 256, y con P = 512, 1024 y 2048 |
| Poblaciones grandes | Con P = 32768 y P = 65536 (más de 32767 individuos por ejecución): los supervivientes son distintos, están ordenados por (rango, crowding) y nadie de la población domina a un superviviente de rango 1 |
| Greedy 2-opt | La permutación resultante es **idéntica** a la de un greedy en CPU que recalcula el coste completo (n = 10, 30 y 60, este último con más de 48 KB de *shared memory*) |
| Reproducción | Los supervivientes y su fitness se copian correctamente y los hijos son permutaciones válidas |
| Población inicial | Todas las permutaciones son válidas y están barajadas |

```
build\x64\Release\test_kernels.exe mQAPData

:: --quick omite las dos pruebas con más de 32767 individuos (demasiado lentas bajo compute-sanitizer)
build\x64\Release\test_kernels.exe mQAPData --quick
```

Validación adicional realizada con `compute-sanitizer` sobre el programa y sobre las pruebas:

```
compute-sanitizer --tool memcheck --leak-check full build\x64\Release\cuda_mqap.exe mQAPData\KC30-3fl-1rl.dat --population 32 --iterations 5 --runs 2 --verify
compute-sanitizer --tool racecheck  ...
compute-sanitizer --tool synccheck  ...
compute-sanitizer --tool initcheck  ...
```

Resultado: 0 errores, 0 fugas y 0 *hazards*.

---

## Rendimiento

GeForce RTX 2060 (sm_75, 30 SM), builds Release, CUDA 13.4. La columna "Original corregida" es el código
monolítico anterior (`kernel.cu`, commit `3f3a187`) con sus errores de memoria corregidos.

| Caso | Original corregida | Esta versión | Aceleración (tiempo de pared) |
|---|---|---|---|
| KC10-2fl-1rl, P=64, 70 gen., 1 ejecución | 2,2 s | 0,15 s (15 ms de GPU) | ~15× |
| KC10-2fl-1rl, P=64, 70 gen., 10 ejecuciones | 23,7 s | 0,12 s (21 ms de GPU) | ~200× |
| KC20-2fl-1rl, P=64, 300 gen., 1 ejecución | 48,8 s | 0,15 s (55 ms de GPU) | ~325× |
| KC30-3fl-1rl, P=32, 70 gen., 1 ejecución | 42,4 s | 0,14 s (40 ms de GPU) | ~300× |
| KC30-3fl-1rl, P=32, 70 gen., 30 ejecuciones | ~21 min (estimado) | 0,23 s (135 ms de GPU) | ~5 500× |

En esta versión, el tiempo de pared está dominado por la creación del contexto CUDA (~0,1 s), así
que el tiempo de GPU refleja mejor el coste del algoritmo.

Perfil con Nsight Systems (KC10-2fl-1rl, 70 generaciones, 1 ejecución):

| Métrica | Original (`ec882da`) | Esta versión |
|---|---|---|
| Tiempo total | 3,64 s | 0,15 s |
| Lanzamientos de kernel | 87 510 | 214 |
| `cudaMemcpy` | 80 558 | 6 |
| `cudaDeviceSynchronize` | 68 335 | 0 |
| `cudaMalloc` / `cudaFree` | 21 507 / 20 724 (783 fugas) | 11 / 11 |
| Tiempo total en kernels | ~340 ms | ~6,4 ms |

**Calidad de las soluciones.** Se mide como la fracción de puntos del frente óptimo `.PO` encontrados
exactamente y como IGD normalizado, con los mismos parámetros en ambas versiones:

| Instancia | Original corregida | Esta versión |
|---|---|---|
| KC10-2fl-1rl (P=64, 70 gen.) | 68,4 % · IGD 0,0053 | 68,4 % · IGD 0,0054 |
| KC10-2fl-3uni (P=128, 25 gen.) | 44,2 % · IGD 0,0057 | 45,0 % · IGD 0,0055 |

En estas dos instancias la calidad es equivalente. En la campaña completa del libro de Excel los resultados
son mixtos: ver [Resultados en el libro de Excel](#resultados-en-el-libro-de-excel).

---

## Mejoras respecto a la versión original

### Errores corregidos

| # | Error en la versión original | Corrección |
|---|---|---|
| B1 | Se reservaba 1 `curandState` pero se inicializaban hasta 8 192, escribiendo fuera de límites en memoria de GPU (con CUDA 13.4 corrompe los resultados de la variante sin Greedy) | Estados Philox dimensionados por hilo (`[R][2P]`) y persistentes |
| B2 | El torneo binario usaba 2P bloques sobre arrays de tamaño P (accesos fuera de límites) | Un hilo por descendiente, con guarda de límites |
| B3 | `settings_KC20_2fl_1rl.cu` contenía las matrices de KC20-2fl-2rl | Las instancias se leen directamente de los `.dat` |
| B4 | La semilla del torneo era `time(NULL)` en cada generación, así que ~19 generaciones seguidas repetían adversarios | RNG persistente con una sola semilla de 64 bits |
| B5 | El barajado inicial usaba estados curand compartidos entre bloques y estaba sesgado | Fisher-Yates con un estado por cromosoma |
| B6 | En el crowding, `(unsigned)HUGE_VALF` (comportamiento indefinido), un extremo de frente mal detectado y posible división por cero | Crowding reescrito con ∞ real y comprobación del rango |
| B7 | 11 reservas de GPU por generación sin liberar | `DeviceBuffer<T>` RAII; todas las reservas se hacen una vez |
| B8 | El greedy evaluaba cada par dos veces ((i,j) y (j,i)) | Cada par `r < s` se evalúa una sola vez |
| B9 | ~0,9 MB de arrays de depuración en la pila del host (1 MB en Windows) | Eliminados |
| B10 | `DEV_MODE \|\| PRINT_*` en lugar de `&&`, y `sizeof` incorrecto | Eliminados junto con el código de depuración |
| B11 | La última iteración sacaba solo el primer frente mezclado con filas obsoletas | La salida es exactamente el frente no dominado de la población final |
| B12 | Soluciones fuera del frente podían ganar la ordenación por crowding | Selección por clave compuesta (rango, −crowding) |

### Optimizaciones

| Área | Antes | Ahora |
|---|---|---|
| Fitness | 3 productos de matrices densas O(n³), bloques de 32×32 hilos (≈10 % útiles con n = 10), accesos no coalescentes y memoria constante serializada | O(n²), un warp por cromosoma, matrices en *shared memory*, reducción con *shuffles* |
| NSGA-II | Bucle en el host por frente, con ~10 kernels y copias por frente; bitonic sort con bloques de 2 hilos (28 lanzamientos por ordenación) | Un único kernel por generación, todo en *shared memory* |
| Greedy 2-opt | ~50 llamadas a la API por par evaluado (fitness completo, `cudaMalloc`/`cudaFree`, copias) | Un lanzamiento por generación, delta O(n) |
| Configuración de lanzamiento | 13 kernels con 1 hilo por bloque (1/32 de eficiencia SIMT) | 1 hilo o 1 warp por elemento, bloques de 128 hilos |
| Transferencias | Copias de depuración siempre activas (~1 150 por generación) | Solo 6 copias al final de la ejecución |
| Sincronización | `cudaDeviceSynchronize` tras cada kernel | Ninguna durante la ejecución |
| Escalabilidad | Ejecuciones en serie (bucle `TIMES`) | `--runs R` concurrentes (`blockIdx.y = run`) |

### Ingeniería

- `kernel.cu` monolítico (2 081 líneas) → módulos con separación host/device.
- 15 `settings_*.cu` recompilados por instancia → instancia y parámetros en tiempo de ejecución.
- Errores ignorados → `CUDA_CHECK` / `CUDA_CHECK_KERNEL` que abortan con fichero y línea.
- Proyecto de Visual Studio no versionado (excluido por `.gitignore`) → `cuda_mqap.slnx` + `CMakeLists.txt` versionados.
- Sin pruebas → `test_kernels` + `--verify` + `compute-sanitizer`.

### Diferencias de comportamiento

- El greedy 2-opt evalúa cada par una sola vez. Con el criterio "todos los objetivos" compara la suma
  exacta de las variaciones, en lugar de medias truncadas a entero.
- El fichero de resultados contiene solo las soluciones no dominadas de la población final.
- La población mínima es 16 (antes KC10-2fl-2uni usaba 4) y debe ser potencia de 2.
- El adversario del torneo se elige de forma uniforme entre las P supervivientes.

---

## Límites del tamaño de población y recursos de la GPU

**En esta rama la población máxima es P = 65536 en cualquier GPU, para las 15 instancias.** Hasta P = 256 la
supervivencia NSGA-II de cada ejecución la sigue haciendo un único bloque de 2P hilos (`nsga2.cu`); por
encima se usa la supervivencia multibloque de `nsga2_multiblock.cu`:

1. `countDominatorsKernel`: cuántos individuos dominan a cada uno, leyendo el fitness en teselas en memoria
   compartida.
2. `peelFrontsKernel`: **lanzamiento cooperativo**, para que toda la malla pueda sincronizarse
   (`grid.sync()`). En cada frente, los individuos sin dominadores restantes se añaden a una lista y los
   demás descuentan a los miembros que los dominaban, hasta que todos tienen rango.
3. Crowding distance: una **ordenación por segmentos** (CUB) por objetivo según (rango, fitness), que deja
   cada frente contiguo, y después la misma acumulación que en el kernel de un bloque.
4. Selección: ordenación por segmentos según (rango ascendente, crowding descendente); sobreviven los P
   primeros de cada ejecución.

Todos los búferes son O(N) por ejecución (N = 2P), no O(N²), así que **la VRAM tampoco es el límite**: el
coste de una población mayor es el tiempo, que crece como N². Los índices y rangos de los supervivientes
son `int`, así que nada se rompe al pasar de 32767 individuos; el tope de 65536 es práctico: más allá, una
sola generación ya cuesta cientos de milisegundos (ver la tabla siguiente).

### Medido en la RTX 2060

- **Con P ≤ 256 los resultados son idénticos a `develop_with_claude_opus_5`** (mismo kernel de un bloque):
  con la misma semilla, los ficheros de resultados coinciden byte a byte en KC10, KC20 y KC30.
- **Las poblaciones grandes funcionan en las 23 instancias `.dat`** con `--verify` OK (P = 8192), y también
  KC10-2fl-1rl y KC30-3fl-1rl con P = 65536. Tiempo de GPU de una ejecución, con las iteraciones de cada
  pestaña del libro:

| Instancia (iteraciones) | P = 256 | P = 512 | P = 1024 | P = 2048 | P = 4096 |
|---|---|---|---|---|---|
| KC10-2fl-1rl (70) | 23 ms | 45 ms | 64 ms | 87 ms | 153 ms |
| KC30-3fl-1rl (70) | 64 ms | 138 ms | 206 ms | 367 ms | 683 ms |

  Con las poblaciones más grandes, tiempo de GPU de 10 generaciones de una ejecución (semilla 12345) y el
  tiempo por generación que resulta:

| Instancia | P = 8192 | P = 16384 | P = 32768 | P = 65536 |
|---|---|---|---|---|
| KC10-2fl-1rl | 71 ms (7,1 ms/gen.) | 167 ms (16,7) | 353 ms (35,3) | 1183 ms (118,3) |
| KC30-3fl-1rl | 206 ms (20,6 ms/gen.) | 439 ms (43,9) | 1049 ms (104,9) | 2800 ms (280,0) |

  Entre P = 32768 y P = 65536 el tiempo se multiplica aproximadamente por tres: el conteo de dominancia, que
  es O(N²), empieza a dominar sobre la parte O(N) de la generación. Una ejecución completa de 70
  generaciones con P = 65536 cuesta 6,7 s de GPU en KC10-2fl-1rl y 17,7 s en KC30-3fl-1rl.

### Efecto del tamaño de población en la calidad

Instancias KC10, 70 iteraciones y 100 ejecuciones por caso. La primera cifra es la distancia gama al frente
de Pareto óptimo publicado (menor es mejor, calculada como `mQAPMetrics/distance_metric_*.js`); el
porcentaje es la fracción media del frente óptimo encontrada por ejecución.

| Instancia | P = 256 | P = 512 | P = 1024 | P = 2048 | P = 4096 |
|---|---|---|---|---|---|
| KC10-2fl-1rl | 782,85 / 75 % | 692,33 / 77 % | 651,89 / 79 % | 573,58 / 79 % | 557,29 / 79 % |
| KC10-2fl-5rl | 16.866,45 / 62 % | 13.451,29 / 66 % | 10.487,07 / 70 % | 7.567,27 / 72 % | 6.224,17 / 74 % |
| KC10-2fl-3uni | 160,14 / 68 % | 119,95 / 72 % | 75,45 / 76 % | 54,64 / 80 % | 29,96 / 82 % |

La fracción del frente óptimo encontrada crece con P en todas las instancias, y la distancia gama baja. El
tiempo de GPU del lote completo (100 ejecuciones) crece aproximadamente de forma lineal con P: 0,6 s con
P = 512 y 6,1 s con P = 4096 en KC10-2fl-1rl.

La serie roja de `comparative_results_kcX_datasets.xlsx` muestra el mismo efecto con el tope de la rama,
P = 65536, en las doce instancias del libro: ver
[Resultados en el libro de Excel](#resultados-en-el-libro-de-excel).

### Cómo calcular el límite en otra GPU

1. **Tope del código:** P ≤ 65536 (`kMaxPopulation` en `include/config.h`). Es un límite de tiempo, no de
   memoria ni del tipo de los índices: los índices y rangos de los supervivientes son `int`.
2. **Memoria compartida:** solo importa con P ≤ 256 (camino de un bloque, 47 KB como máximo). El camino
   multibloque usa teselas de fitness de unos 3 KB.
3. **Lanzamiento cooperativo:** necesario para el pelado de frentes; lo admiten todas las GPU NVIDIA desde
   Pascal.
4. **VRAM:** determina cuántas ejecuciones concurrentes caben:

```
ejecuciones máximas   = min(65 535, VRAM libre / memoria por ejecución)
memoria por ejecución = 2P·(4n + 8·OBJ + 64) + 12P + 40·2P bytes  (el último término es el espacio de trabajo)
```

Con P = 4096 en KC30, una ejecución ocupa unos 2 MB, así que 100 ejecuciones concurrentes necesitan unos
200 MB. El espacio de trabajo solo se reserva cuando P > 256.

| GPU | VRAM | P máx. (esta rama) | Ejecuciones concurrentes de KC30 con P = 4096 (calculado) |
|---|---|---|---|
| RTX 2060 (medida) | 6 GB | **65536** | ~2600 |
| RTX 3070 Laptop | 8 GB | **65536** | ~3500 |
| RTX 3080 | 10–12 GB | **65536** | ~4400–5200 |
| RTX 4080 / 4090 | 16 / 24 GB | **65536** | ~7000–10 500 |
| RTX 5080 / 5090 | 16 / 32 GB | **65536** | ~7000–14 000 |

Solo la fila de la RTX 2060 está medida; las demás usan el 85 % de la VRAM de cada GPU. Estas cifras quedan
muy por encima de lo que permite el tiempo: con P = 4096, cada ejecución de KC30 cuesta unos 0,7 s de GPU.

### Cómo explorar más soluciones

1. **Más ejecuciones concurrentes:** `--runs` (islas sin migración, por ahora).
2. **Población mayor:** hasta 65536 en esta rama. A partir de unos pocos miles de individuos, el conteo de
   dominancia O(N²) domina el coste de la generación, así que el límite útil es el tiempo que se quiera
   gastar, no la memoria.
3. **Ambas cosas:** el producto (ejecuciones × P) lo limita la VRAM y, en la práctica, el tiempo.

---

## Limitaciones y trabajo futuro

**Límites actuales:**
- n ≤ 64. La *shared memory* disponible también influye: con 3 objetivos, n ≤ 63 en GPUs con 64 KB *opt-in*.
- P es una potencia de 2 entre 16 y 65536; hasta 256 la supervivencia usa un bloque de 2P hilos
  (ver [Límites del tamaño de población y recursos de la GPU](#límites-del-tamaño-de-población-y-recursos-de-la-gpu)).
- Solo se admiten 2 o 3 objetivos (los kernels están instanciados para esos valores).
- Los costes se almacenan como enteros de 32 bits; el cargador rechaza las instancias que podrían desbordarlos.

**Posibles mejoras:**
- Modelo de islas con migración entre las ejecuciones concurrentes.
- CUDA Graphs para capturar la generación; con 3 kernels por generación, el beneficio esperado es pequeño.
- Análisis con Nsight Compute del `survivalKernel`, que al ser un solo bloque está limitado por la latencia.
- Más operadores de cruce y variantes del criterio del 2-opt.

---

## Solución de problemas

| Síntoma | Causa y solución |
|---|---|
| `CUDA Toolkit X.Y Visual Studio integration not found` | El CUDA Toolkit se instaló antes que Visual Studio, o sin su *Visual Studio Integration*: vuelve a ejecutar el instalador de CUDA (instalación personalizada → Visual Studio Integration). Si tienes varios toolkits instalados, elige uno con `CudaVersion` (ver [Abrir en Visual Studio 2026](#abrir-en-visual-studio-2026-plug-and-play)) |
| `no kernel image is available for execution on the device` | La GPU es anterior a `sm_75`, o el driver es demasiado antiguo para compilar el PTX: actualiza el driver o añade la arquitectura en `cuda_mqap.props` (`CodeGeneration`) |
| Visual Studio pide instalar componentes al abrir la solución | Viene de `.vsconfig`: acepta para instalar la carga de trabajo de C++ y el SDK de Windows |
| `population must be a power of two in [16, 65536]` | Usa una potencia de 2 entre 16 y 65536 |
| `instance too large: … shared memory` | La instancia no cabe en la *shared memory* del bloque (ver límites) |
| `costs may overflow 32-bit fitness values` | La instancia podría desbordar el fitness de 32 bits |
| Ejecución muy lenta en Debug | Es lo esperado: Debug compila el device con `-G` y sincroniza tras cada kernel. Usa Release para medir |
| `[CUDA] … at <fichero>:<línea>` | Error de CUDA con su ubicación exacta; para más detalle, ejecuta bajo `compute-sanitizer` |

---

## Créditos y licencia

### Créditos

- **Autor:** Andrés Pupiales Arévalo — <apupiales@gmail.com> — <https://github.com/apupiales>. Proyecto iniciado en mayo de 2019.
- **Instancias mQAP:** J. Knowles y D. Corne; frentes óptimos de G. Lamont (ver [Datos de terceros](#datos-de-terceros)).
- **Refactorización y optimización de la versión actual:** realizadas con la asistencia de Claude (Anthropic).

### Licencia

Copyright (C) 2019-2026 Andrés Pupiales Arévalo.

Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los términos de la
**GNU General Public License**, publicada por la Free Software Foundation, en su **versión 3** o, a tu
elección, cualquier versión posterior (`SPDX-License-Identifier: GPL-3.0-or-later`). Se distribuye con
la esperanza de que sea útil, pero **sin ninguna garantía**. Consulta el texto completo en [`LICENSE`](LICENSE).

Todos los ficheros fuente llevan la cabecera correspondiente.

**Sobre CUDA.** El CUDA Toolkit de NVIDIA (compilador `nvcc`, runtime `cudart` y cabeceras de cuRAND)
**no forma parte de este repositorio**: es software propietario de NVIDIA, distribuido bajo su propia
licencia (CUDA EULA), y se necesita para compilar y ejecutar el programa. La licencia GPL v3 cubre
únicamente el código de este proyecto.

### Datos de terceros

Los ficheros de `mQAPData/` **no están cubiertos por la licencia GPL v3** de este proyecto: son datos
de *benchmark* de terceros, redistribuidos sin modificar para uso académico y de investigación.

- **Instancias (`.dat`):** conjunto de pruebas del mQAP de Joshua Knowles y David Corne, generado con
  sus generadores `makeQAPuni`/`makeQAPrl` ((C) J. Knowles, 2002). La página original ya no está en
  línea; [copia archivada](https://web.archive.org/web/2019/http://www.cs.bham.ac.uk/~jdk/mQAP/).
- **Frentes óptimos (`.PO`):** enumeración de las instancias de 10 instalaciones realizada por Gary Lamont,
  publicada en la misma página.
- **Copia utilizada:** [fredizzimo/keyboardlayout](https://github.com/fredizzimo/keyboardlayout/tree/master/tests/mQAPData)
  (ficheros idénticos). La licencia MIT de ese repositorio no cubre estos datos, cuyos autores son los citados arriba.
- **Condiciones:** no se ha publicado una licencia explícita. La página original ofrece los generadores
  como software libre *"for academic or educational use"* y pide contactar con el autor para uso
  comercial; aplica el mismo criterio a las instancias.
- **Cita:** J. D. Knowles y D. W. Corne, *Instance Generators and Test Suites for the Multiobjective
  Quadratic Assignment Problem*, EMO 2003, LNCS 2632, pp. 295–310, Springer, 2003.

Detalles y entrada BibTeX en [`mQAPData/README.txt`](mQAPData/README.txt).
