# Comparación con la CPU y con otros MOEA

[English](COMPARISON.md) | **Español**

Este documento pertenece a la rama `develop_comparison_vs_cpu_and_moeas`, que parte de
`develop_large_population_multiblock` y conserva todo lo que describe su [LEEME](LEEME.md). Responde a dos
preguntas que el resto del proyecto dejaba abiertas:

1. **¿Cuánto aporta la GPU?** Las aceleraciones del LEEME se miden contra el código original de 2019, cuyo tiempo se
   iba en llamadas a la API. Aquí el mismo algoritmo se ejecuta en la GPU y en una CPU multinúcleo.
2. **¿Qué tan bueno es el resultado frente a otros algoritmos?** Aquí `cuda_mqap` se compara con NSGA-II, NSGA-III y
   MOEA/D de [pymoo](https://pymoo.org), cada uno con y sin el mismo greedy 2-opt, a **igual trabajo** y a **igual
   tiempo de reloj**, en las 23 instancias de Knowles–Corne y en 16 instancias mayores con n = 60.

---

## Índice

1. [Qué añade esta rama](#qué-añade-esta-rama)
2. [Requisitos](#requisitos)
3. [Protocolo](#protocolo)
4. [Cómo reproducirlo](#cómo-reproducirlo)
5. [Resultados](#resultados)
6. [Qué dicen los resultados](#qué-dicen-los-resultados)
7. [Limitaciones](#limitaciones)

---

## Qué añade esta rama

| Pieza | Dónde | Qué hace |
|---|---|---|
| Versión CPU del algoritmo | `src/solver_cpu.cpp`, `--cpu`, `--threads N` | El mismo algoritmo con OpenMP: misma disposición de datos, inicialización Fisher-Yates, fitness O(n²), supervivencia por conteo de dominadores y pelado de frentes, crowding, torneo y mutaciones, greedy 2-opt con la delta O(n), el mismo recorrido de pares y la misma compuerta. Solo cambia el generador aleatorio (xoshiro256** por fila en lugar de Philox) |
| Compuerta compartida | `include/gate.h` | `greedyApplies` sin dependencias de CUDA, para que la compartan los kernels, la versión CPU y el host |
| Contador de trabajo | `SolveStats`, línea `Evaluations:` de la salida | Evaluaciones completas O(n²) y deltas de intercambio O(n) del greedy, contadas en el host a partir de la compuerta determinista, para poder dar el mismo presupuesto a algoritmos distintos |
| Instancias de Garrett | `scripts/fetch_gar60.ps1` → `data/gar60/` | Descarga las 16 instancias con n = 60 y 2 o 3 objetivos (Garrett y Dasgupta, 2009), tal como se publicaron con PasMoQAP (Sanhueza et al., CEC 2017). Ese repositorio no declara licencia, así que no se versionan (`data/` está ignorado) |
| Algoritmos de pymoo | `scripts/baselines/pymoo_mqap.py` | NSGA-II, NSGA-III y MOEA/D con operadores de permutación, y cada uno con el greedy 2-opt de `cuda_mqap` como reparación (`-ls`). Presupuesto en generaciones, evaluaciones o segundos; salida en el formato del fichero de resultados |
| Campaña | `scripts/run_comparison.ps1` | Los cuatro bloques del [protocolo](#protocolo), reanudable |
| Análisis | `scripts/analyze_comparison.py` | Hipervolumen, cobertura, Mann–Whitney con corrección de Holm, A12 de Vargha–Delaney, victorias/empates/derrotas y prueba de Friedman |
| Pruebas | `tests/test_kernels.cu` | Dos comprobaciones de la versión CPU: permutaciones válidas, fitness exacto, rangos coherentes con la dominancia, frente sin repeticiones y el mismo conteo de trabajo que la versión GPU. **29 comprobaciones** en total |

La versión CPU es el mismo algoritmo, no una aproximación, y eso se comprobó: 30 ejecuciones por versión en
KC20-2fl-1rl y KC20-2fl-3uni (P = 64, 300 generaciones) no difieren en cobertura (Mann–Whitney p = 0,14 y 0,51), y
la única diferencia de hipervolumen con p < 0,05 (0,085 puntos, p = 0,022) deja de ser significativa con la
corrección de Holm sobre las cuatro pruebas.

Los algoritmos de pymoo usan la misma función de coste que `cuda_mqap`, que reproduce los 374 costes óptimos
publicados de las KC10, y sus variantes meméticas usan la misma búsqueda local: el mismo recorrido de pares, la
misma aceptación y un criterio por generación elegido entre la suma y cada objetivo.

## Requisitos

- La compilación Release del [LEEME](LEEME.md#compilación). La versión CPU necesita OpenMP, que Visual Studio y
  CMake activan por su cuenta.
- Python 3 con `numpy`, `scipy`, `numba` y `pymoo` (medido con pymoo 0.6.2):

```
pip install numpy scipy numba pymoo
```

- Conexión, una vez, para descargar las instancias de Garrett: `.\scripts\fetch_gar60.ps1`.

## Protocolo

Treinta ejecuciones por algoritmo e instancia, semilla 20261006, y cada frente final medido contra el mismo frente
de referencia de su instancia.

| Bloque | Qué mide | Ajustes |
|---|---|---|
| `speedup` | El mismo algoritmo en la GPU y en 12 hilos de CPU | KC10-2fl-1rl, KC20-2fl-1rl y KC30-3fl-1rl; P = 64 … 16 384; 20 generaciones; mediana de 3 ejecuciones; greedy en todos los descendientes |
| `budget` | **Igual trabajo** en las 23 instancias de Knowles–Corne | P = 64 y las generaciones de la versión original (70 en KC10, 300 en KC20 y KC30); `cuda_mqap --untuned` |
| `gar60` | Igual trabajo **e** igual tiempo en las 16 instancias de Garrett | Igual trabajo: P = 64, 100 generaciones. Igual tiempo: `cuda_mqap` con P = 4096 y 100 generaciones; cada ejecución de pymoo, con P = 100, recibe el tiempo de reloj de una de esas ejecuciones (11 a 16 s) |
| `time` | **Igual tiempo de reloj** en las 23 instancias de Knowles–Corne | `cuda_mqap` con la [ejecución por defecto](LEEME.md#la-ejecución-por-defecto-de-cada-instancia) de cada instancia; cada ejecución de pymoo, con P = 100, recibe el tiempo de una de esas ejecuciones |

- **Algoritmos de referencia.** pymoo 0.6.2: muestreo aleatorio de permutaciones, cruce de orden y mutación por
  inversión; eliminación de duplicados en NSGA-II y NSGA-III; direcciones de referencia de energía de Riesz en
  NSGA-III y MOEA/D, con 20 vecinos y probabilidad de cruce entre vecinos de 0,9 en MOEA/D. Las variantes `-ls`
  aplican el greedy 2-opt de `cuda_mqap` a todos los descendientes. A igual trabajo hacen **al menos tantas**
  evaluaciones de intercambio como `cuda_mqap`, porque pymoo también repara la población inicial y los
  descendientes que regenera para sustituir duplicados.
- **Las ejecuciones de pymoo usan un solo hilo**, y se lanzan seis a la vez, una por núcleo físico, para que un
  presupuesto de tiempo signifique lo mismo para cada una.
- **Frentes de referencia.** El óptimo publicado en KC10, [`reference/v0.4`](reference/README.md) en KC20 y KC30, y
  en las instancias de Garrett, que no tienen ninguno, la unión no dominada de todas las ejecuciones de todos los
  algoritmos y protocolos de la campaña.
- **Estadística.** Cada algoritmo de referencia contra `cuda_mqap`: Mann–Whitney bilateral con corrección de Holm
  sobre los seis de la instancia, y el tamaño del efecto A12 de Vargha–Delaney. Sobre las instancias:
  victorias/empates/derrotas con p corregido < 0,05 y prueba de Friedman sobre los rangos medios.

## Cómo reproducirlo

```
:: Una vez: las instancias de Garrett y los paquetes de Python
powershell -ExecutionPolicy Bypass -File scripts\fetch_gar60.ps1
pip install numpy scipy numba pymoo

:: Los cuatro bloques, en el orden en que se ejecutaron; una campaña interrumpida sigue donde se quedó
powershell -ExecutionPolicy Bypass -File scripts\run_comparison.ps1 -Block speedup,budget,gar60,time

:: Los resúmenes (Markdown y JSON en results\comparison)
python scripts\analyze_comparison.py results\comparison --block budget
python scripts\analyze_comparison.py results\comparison --block gar60 --protocol p64
python scripts\analyze_comparison.py results\comparison --block gar60 --protocol time
python scripts\analyze_comparison.py results\comparison --block time
```

En la RTX 2060 y el i5-11400 de la medición, los bloques tardaron 4 min (`speedup`), 1 h 12 min (`budget`),
5 h 15 min (`gar60`), y el bloque `time` se estima en unas 17 h (dominado por las siete instancias KC30, cuya ejecución por defecto tarda
104 s). Cada resultado va a `results\comparison\<bloque>\<instancia>\<algoritmo>_<protocolo>.txt`, en el formato del
fichero de resultados, con un `.json` (pymoo) o un `.log` (`cuda_mqap`) al lado.

Los algoritmos también se pueden ejecutar uno a uno:

```
build\x64\Release\cuda_mqap.exe mQAPData\KC20-2fl-1rl.dat --untuned --runs 30 --cpu --threads 12
python scripts\baselines\pymoo_mqap.py mQAPData\KC20-2fl-1rl.dat --algorithm nsga2-ls --pop 64 --gen 300 --runs 30
python scripts\baselines\pymoo_mqap.py data\gar60\Gar60-3fl-1rl.dat --algorithm moead-ls --pop 100 --seconds 16
```

## Resultados

### 1. El mismo algoritmo en la GPU y en la CPU

Tiempo por generación, GPU / CPU, y aceleración de la GPU (RTX 2060 frente a un i5-11400 con 12 hilos OpenMP):

| Instancia | P = 64 | P = 256 | P = 1 024 | P = 4 096 | P = 16 384 |
|---|---|---|---|---|---|
| KC10-2fl-1rl | 0,56 / 0,4 ms · **0,7×** | 0,80 / 0,9 ms · **1,1×** | 1,47 / 5,3 ms · **3,6×** | 3,66 / 52,0 ms · **14,2×** | 11,74 / 749,9 ms · **63,9×** |
| KC20-2fl-1rl | 0,82 / 1,4 ms · **1,7×** | 1,34 / 3,2 ms · **2,4×** | 2,95 / 12,4 ms · **4,2×** | 6,75 / 85,3 ms · **12,6×** | 27,36 / 944,8 ms · **34,5×** |
| KC30-3fl-1rl | 1,59 / 6,4 ms · **4,0×** | 2,38 / 14,2 ms · **5,9×** | 5,23 / 46,3 ms · **8,8×** | 16,71 / 217,3 ms · **13,0×** | 67,33 / 1567,1 ms · **23,3×** |

Con la población de la versión original la GPU aporta poco, y en KC10 es más lenta: una generación es una
fracción de milisegundo en ambos dispositivos. La aceleración crece con la población, hasta 23–64× con P = 16 384,
donde la CPU necesita entre 0,75 y 1,6 s por generación. Una ejecución de 300 generaciones con el tope de
población, rutinaria en la GPU, tardaría horas en la CPU.

### 2. Igual trabajo en las instancias de Knowles–Corne

P = 64, 70 o 300 generaciones, 30 ejecuciones, 23 instancias. V/E/D: victorias, empates y derrotas de `cuda_mqap`
frente a cada algoritmo (Mann–Whitney con corrección de Holm, p < 0,05):

| Algoritmo | Rango medio, cobertura | Rango medio, hipervolumen | cuda_mqap V/E/D, cobertura | cuda_mqap V/E/D, hipervolumen |
|---|---|---|---|---|
| **cuda_mqap** | **2,04** | **1,76** | — | — |
| NSGA-II+LS | 2,52 | 2,15 | 9 / 10 / 4 | 13 / 7 / 3 |
| NSGA-III+LS | 2,26 | 2,30 | 9 / 10 / 4 | 12 / 9 / 2 |
| MOEA/D+LS | 3,57 | 3,78 | 16 / 5 / 2 | 22 / 1 / 0 |
| NSGA-II | 5,65 | 5,65 | 21 / 2 / 0 | 23 / 0 / 0 |
| NSGA-III | 5,83 | 6,00 | 21 / 2 / 0 | 23 / 0 / 0 |
| MOEA/D | 6,13 | 6,35 | 21 / 2 / 0 | 23 / 0 / 0 |

Friedman p = 4,0·10⁻²¹ (cobertura) y 7,9·10⁻²³ (hipervolumen).

- `cuda_mqap` tiene el mejor rango medio en los dos indicadores.
- **La búsqueda local es lo que separa a los algoritmos.** Sin ella, los tres algoritmos de pymoo cubren menos del
  17 % del frente de referencia en todas las instancias, y el 0 % en todas las KC20 y KC30.
- **Frente a las variantes meméticas, que comparten su búsqueda local, el resultado es parejo.** Gana en las KC20
  *real-like* y en KC30-2fl-1rl (37,0 % frente a 31,8 % de cobertura en KC20-2fl-1rl, 3,7 % frente a 1,5 % en
  KC30-2fl-1rl), y pierde en KC10-2fl-3rl, -3uni y -5rl y en KC20-2fl-2uni (17,5 % frente a 34,2 %), donde el cruce
  de orden y la eliminación de duplicados de pymoo mantienen más soluciones distintas en una población de 64.
- En las KC30 de tres objetivos ningún algoritmo cubre más del 0,1 % del frente de referencia con este presupuesto,
  así que allí la ordenación descansa en el hipervolumen.

Las tablas completas por instancia están en `results\comparison\summary_budget_p64.md`.

### 3. Las instancias de Garrett (n = 60)

El frente de referencia de estas instancias es la unión de la campaña, a la que `cuda_mqap` aporta la mayoría de
los puntos, así que la cobertura no es informativa y la comparación descansa en el **hipervolumen**. El
hipervolumen es compatible con la dominancia de Pareto, así que la ordenación es válida; sus valores absolutos
dependen de esa unión.

| Protocolo | Rango medio de cuda_mqap | Mejor rival (rango medio) | cuda_mqap V/E/D frente a cada uno de los seis | Friedman p |
|---|---|---|---|---|
| Igual trabajo (P = 64, 100 generaciones) | **1,19** | NSGA-II+LS (2,81) | 15 / 1 / 0 | 6,9·10⁻¹⁶ |
| Igual tiempo (11–16 s por ejecución) | **1,00** | NSGA-II+LS (2,59) | **16 / 0 / 0** | 4,2·10⁻¹⁶ |

Fracción media del hipervolumen de referencia (%) de `cuda_mqap` y del mejor rival de cada instancia:

| Instancia | cuda_mqap, igual trabajo | mejor rival | cuda_mqap, igual tiempo | mejor rival |
|---|---|---|---|---|
| Gar60-2fl-1rl | **93,5** | 90,2 (MOEA/D+LS) | **98,5** | 89,5 (NSGA-II+LS) |
| Gar60-2fl-1uni | **85,8** | 83,3 (MOEA/D+LS) | **95,2** | 81,0 (NSGA-II+LS) |
| Gar60-2fl-2rl | **93,7** | 89,7 (NSGA-III+LS) | **98,6** | 88,6 (NSGA-III+LS) |
| Gar60-2fl-2uni | **81,8** | 79,2 (MOEA/D+LS) | **93,9** | 75,2 (MOEA/D+LS) |
| Gar60-2fl-3rl | **92,2** | 88,8 (MOEA/D+LS) | **98,3** | 87,1 (NSGA-II+LS) |
| Gar60-2fl-3uni | **68,4** | 65,3 (MOEA/D+LS) | **88,4** | 59,5 (MOEA/D+LS) |
| Gar60-2fl-4rl | **93,8** | 90,5 (NSGA-II+LS) | **98,5** | 90,4 (NSGA-II+LS) |
| Gar60-2fl-4uni | **91,1** | 89,9 (NSGA-II+LS) | **96,5** | 89,9 (NSGA-II+LS) |
| Gar60-2fl-5rl | **88,0** | 81,9 (MOEA/D+LS) | **97,8** | 80,3 (NSGA-II+LS) |
| Gar60-2fl-5uni | **0,0** | 0,0 (MOEA/D+LS) | **25,3** | 0,0 (MOEA/D+LS) |
| Gar60-3fl-1rl | **80,1** | 75,8 (NSGA-II+LS) | **95,0** | 77,1 (NSGA-II+LS) |
| Gar60-3fl-1uni | **68,3** | 63,9 (MOEA/D+LS) | **89,7** | 63,7 (NSGA-III+LS) |
| Gar60-3fl-2rl | **79,2** | 74,4 (NSGA-II+LS) | **94,7** | 76,0 (NSGA-II+LS) |
| Gar60-3fl-2uni | **77,9** | 76,2 (NSGA-II+LS) | **91,0** | 76,7 (NSGA-II+LS) |
| Gar60-3fl-3rl | **78,1** | 72,8 (NSGA-II+LS) | **94,8** | 73,4 (NSGA-III+LS) |
| Gar60-3fl-3uni | **51,7** | 45,6 (MOEA/D+LS) | **81,5** | 38,3 (MOEA/D+LS) |

- **Con n = 60, `cuda_mqap` gana incluso a igual trabajo**, cosa que no ocurría en KC10.
- **A igual tiempo la distancia se abre**, sobre todo con tres objetivos: 81,5–95,0 % frente al 38,3–77,1 % del
  mejor rival de cada instancia.
- El único empate es **Gar60-2fl-5uni** a igual trabajo, donde todos se quedan en 0 %: sus matrices de flujo tienen
  correlación 0,8 y casi ninguna ejecución domina la región que acota el punto de referencia. A igual tiempo
  `cuda_mqap` llega al 25,3 % y los demás siguen en 0. PasMoQAP también obtuvo en esta instancia sus valores más
  bajos.

### 4. Igual tiempo de reloj en las instancias de Knowles–Corne

*En curso.* Con 9 de las 23 instancias terminadas, `cuda_mqap` encuentra el 98–100 % del frente óptimo publicado
en las ocho KC10, frente al 37–90 % del mejor rival, y el 95,4 % frente al 65,4 % del frente de referencia en
KC20-2fl-1rl. La tabla se completará cuando termine la campaña.

## Qué dicen los resultados

- **La GPU no hace mucho más rápido un NSGA-II memético con población pequeña; lo que hace es volver asequibles las
  poblaciones grandes.** Con P = 64 la versión CPU es igual de rápida; con P = 16 384 la GPU es 23–64 veces más
  rápida.
- **A igual trabajo el algoritmo es competitivo, no dominante.** Tiene el mejor rango medio, pero frente a los
  NSGA-II y NSGA-III meméticos de pymoo las diferencias son sobre todo empates, y en varias instancias pequeñas o
  uniformes el cruce con eliminación de duplicados de pymoo es mejor. Lo que separa a todo algoritmo memético de su
  versión sin búsqueda local es esa búsqueda local, como ya recogía la literatura del mQAP.
- **A igual tiempo la ventaja se vuelve sistemática**, porque la GPU dedica los mismos segundos a poblaciones entre
  40 y 650 veces mayores que las 100 de los demás. Las mejoras de calidad de este proyecto vienen sobre todo de la
  población que la GPU vuelve asequible, junto con la intensidad ajustada de la búsqueda local, y no de un mejor
  diseño de operadores.

## Limitaciones

- Los algoritmos de referencia son MOEA de propósito general de una sola biblioteca, con sus operadores por
  defecto. Los algoritmos específicos del mQAP (Garrett y Dasgupta; la búsqueda local de Pareto estocástica de
  Drugan; PasMoQAP) no se pudieron ejecutar porque su código no es público. Los resultados publicados de PasMoQAP
  usan un hipervolumen normalizado contra sus propias ejecuciones y no son comparables con estos valores.
- En el protocolo de igual tiempo cada ejecución de pymoo usa un núcleo de CPU, porque pymoo no paraleliza una
  ejecución, mientras que `cuda_mqap` usa la GPU entera. El bloque 1 acota lo que podría ganar una implementación
  paralela en CPU.
- En las instancias de Garrett el frente de referencia es la unión de esta campaña, así que allí solo se usa el
  hipervolumen.
- Una GPU (RTX 2060) y una CPU (i5-11400): los resultados relativos deberían trasladarse; los tiempos absolutos, no.
- No se admiten instancias de cuatro objetivos: los kernels manejan 2 o 3, y con n = 60 cuatro matrices de flujo no
  cabrían en 64 KB de memoria compartida.
