# Neural-Nets-ATARI-project

Agente de **Deep Reinforcement Learning** que aprende a jugar **Breakout (Atari 2600)** desde cero utilizando **Deep Q-Network (DQN)**, basado en el paper fundacional de Mnih et al. (2015).

El agente no recibe reglas, estrategias ni demostraciones humanas: aprende exclusivamente por ensayo y error a partir de los píxeles crudos del juego y la señal de recompensa del entorno.

**Stack:** Python 3.10+ · PyTorch · Gymnasium (ALE) · TensorBoard · Stable-Baselines3 (solo baseline comparativo)

---

## Estructura del proyecto

```
Neural-Nets-ATARI-project/
├── main.py                  # Entry point — CLI para train / eval / watch / baseline
├── config.py                # Hiperparámetros centralizados (dataclass)
├── model.py                 # QNetwork — CNN del paper Mnih et al. 2015
├── agent.py                 # DQNAgent — selección de acción, update, target sync, save/load
├── replay_buffer.py         # ReplayBuffer FIFO con muestreo uniforme (uint8)
├── wrappers.py              # Pipeline de preprocesamiento del entorno (MaxAndSkip, Resize, etc.)
├── train.py                 # Loop de entrenamiento con Ctrl+C graceful
├── evaluate.py              # Evaluación y visualización del agente
├── baseline_comparison.py   # Baseline con Stable-Baselines3 DQN para comparación
├── utils.py                 # Utilidades (seeds, epsilon schedule, device, timer)
├── requirements.txt         # Dependencias del proyecto
├── STARTUP.md               # Guía paso a paso para configurar el entorno
├── checkpoints/             # Checkpoints del modelo (.pt)
├── runs/                    # Logs de TensorBoard
└── DocumentationTestAndLearning/  # Papers de referencia y scripts exploratorios
```

---

## 1. Problema

### 1.1 Definición

El objetivo es entrenar un agente que aprenda a jugar **Breakout (`ALE/Breakout-v5`)** a partir **únicamente de los píxeles del juego**, sin reglas codificadas a mano y sin demostraciones humanas.

En Breakout, el jugador controla una paleta horizontal en la parte inferior de la pantalla. Una pelota rebota entre la paleta y un muro de ladrillos de colores en la parte superior. Cada ladrillo destruido otorga puntos (las filas superiores valen más). El objetivo es destruir todos los ladrillos sin dejar caer la pelota; el jugador dispone de 5 vidas.

### 1.2 Formulación como MDP

El problema se formula como un **Proceso de Decisión de Markov (MDP):**

| Componente | En Breakout |
|---|---|
| **Estado** `s` | Imagen del juego (frames del emulador Atari) |
| **Acción** `a` | 4 acciones discretas: `NOOP`, `FIRE`, `RIGHT`, `LEFT` |
| **Recompensa** `r` | Puntos por destruir ladrillos (valor según fila/color). Clip estándar DeepMind a `{-1, 0, +1}` |
| **Terminación** | Perder las 5 vidas o agotar el timeout del episodio |
| **Objetivo** | Maximizar el retorno esperado descontado: `E[Σ γ^t · r_t]` con `γ = 0.99` |

A diferencia de un problema de aprendizaje supervisado, el agente nunca ve "ejemplos correctos" con etiquetas. Aprende por **ensayo y error** a partir de la señal escalar de recompensa.

### 1.3 ¿Por qué DQN para este problema?

| Razón | Explicación |
|---|---|
| **Input visual** | El estado es una imagen. DQN usa una CNN que procesa píxeles directamente. Q-learning tabular necesitaría ~10⁶⁸⁰⁰ entradas — imposible. |
| **Acciones discretas** | Breakout tiene solo 4 acciones. DQN produce un Q-value por acción y selecciona con `argmax`. Métodos policy gradient (PPO, A3C) son más adecuados para acciones continuas. |
| **Reward directo** | Hay una señal de recompensa clara (puntos por ladrillo). DQN es value-based: aprende cuánto vale cada acción, ideal cuando el reward es señal directa del entorno. |
| **Referencia fundacional** | DeepMind (Mnih et al. 2015) usó exactamente DQN para Atari. Es el algoritmo de referencia para este tipo de problema. |
| **Complejidad manejable** | DQN tiene componentes claros y bien estudiados (CNN + Replay Buffer + Target Network), sin la complejidad de métodos actor-crítico. |

> **Nota sobre variantes de DQN:** Existen extensiones como **Double DQN** (Van Hasselt et al. 2016) y **Dueling DQN** (Wang et al. 2016) que mejoran la estabilidad y el rendimiento. Sin embargo, el enfoque de este proyecto es la implementación del **DQN estándar** del paper original. Las variantes se documentan en la sección de referencias como trabajo futuro potencial.

### 1.4 ¿Por qué Breakout-v5 específicamente?

- **Benchmark estándar:** el paper original de DeepMind lo utilizó. Es el caso de estudio más documentado de DQN, lo que facilita comparar resultados.
- **Reward claro:** puntos por ladrillo destruido (más en filas superiores). El clip a `{-1, 0, +1}` (estándar DeepMind 2015) estabiliza el entrenamiento.
- **Dificultad apropiada:** suficientemente complejo para que el agente aprenda estrategia (por ejemplo, abrir un túnel lateral para que la pelota rebote por encima de los ladrillos), pero sin reward tan disperso que haga el entrenamiento impracticable.
- **Sticky actions (v5):** `ALE/Breakout-v5` incluye `repeat_action_probability=0.25` por defecto, que introduce estocasticidad intencional para evitar que el agente memorice secuencias exactas de acciones (recomendación de Machado et al. 2018).
- **Disponible nativamente:** Gymnasium lo provee vía ALE (Arcade Learning Environment); no requiere implementar el juego.

---

## 2. Datos

A diferencia de un dataset estático (ImageNet, MNIST), en reinforcement learning **los datos se generan online** mientras el agente interactúa con el entorno. Cada paso de tiempo produce una **transición** `(s, a, r, s', done)` que se almacena en el Replay Buffer para entrenamiento posterior.

### 2.1 Observaciones del entorno

| Dato | Descripción |
|---|---|
| **Observación raw** | `(210, 160, 3)` uint8 RGB — frame del emulador Atari a ~60 FPS |
| **Observación preprocesada** | `(4, 84, 84)` uint8 — 4 frames apilados en escala de grises |
| **Recompensa** | Puntos del juego (entero). Valor según la fila/color del ladrillo. Se clipea a `{-1, 0, +1}` durante el entrenamiento |
| **Terminación** | Booleano: episodio terminado (perder las 5 vidas) o truncado (timeout) |
| **Espacio de acciones** | 4 acciones discretas: `NOOP`, `FIRE`, `RIGHT`, `LEFT` |

### 2.2 Pipeline de preprocesamiento (`wrappers.py`)

Cada frame crudo del emulador pasa por la siguiente cadena de transformaciones antes de ser consumido por la red neuronal:

```
raw frame (210, 160, 3) RGB
  → MaxAndSkipEnv(skip=4)        # repite acción 4 frames + max-pool de los 2 últimos
  → RecordEpisodeStatistics()    # registra reward y longitud del episodio
  → ResizeObservation(84, 84)    # downscale espacial
  → GrayscaleObservation()       # 3 canales RGB → 1 canal
  → FrameStackObservation(4)     # apila 4 frames como canales
salida: (4, 84, 84) uint8
```

**Justificación de cada paso:**

| Wrapper | Propósito |
|---|---|
| **MaxAndSkipEnv** | El emulador Atari tiene flickering (sprites alternan entre frames pares e impares). El `max` de los 2 últimos frames lo elimina. Saltar 4 frames acelera el entrenamiento ~4× sin perder información relevante. |
| **ResizeObservation** | Reduce la dimensionalidad de `(210, 160)` a `(84, 84)` sin perder la estructura espacial del juego. |
| **GrayscaleObservation** | El color no aporta información útil en Breakout — los ladrillos importan por su posición, no por su tinte. Reduce canales de 3 a 1. |
| **FrameStackObservation** | Una imagen sola no tiene información de velocidad. Apilando 4 frames el agente puede inferir la trayectoria de la pelota y la dirección de la paleta. |

### 2.3 Almacenamiento: Replay Buffer (`replay_buffer.py`)

Las transiciones se almacenan en un **buffer FIFO circular** con muestreo uniforme:

- **Capacidad:** 100,000 transiciones.
- **Formato:** `uint8` (no `float32`) para minimizar el uso de RAM.
  - `(4, 84, 84) × 100k × 1 byte ≈ 2.8 GB` en uint8.
  - En `float32` serían ~11 GB — inviable en GPUs consumer.
- **Normalización:** se aplica **dentro de la red** (`x.float() / 255.0` en el `forward()` de `QNetwork`), no al almacenar.
- **Muestreo:** al entrenar, se muestrea un mini-batch de 32 transiciones i.i.d. del buffer. El muestreo aleatorio **rompe la correlación temporal** entre transiciones consecutivas, que de otro modo desestabilizaría al SGD.

---

## 3. Arquitectura base

El proyecto implementa **Deep Q-Network (DQN)**, el algoritmo de Mnih et al. (2015) que fue el primero en demostrar control a nivel humano en juegos Atari directamente desde píxeles. La arquitectura combina una **red neuronal convolucional (CNN)** con las técnicas de estabilización de **Experience Replay** y **Target Network**.

### 3.1 Arquitectura propuesta: CNN del paper DQN (`model.py`)

Dado que los datos de entrada son **imágenes con estructura espacial 2D** apiladas en una **dimensión temporal** (4 frames), la arquitectura natural es una **Convolutional Neural Network**. Se replica exactamente la arquitectura del paper de Mnih et al. (2015):

```
Input:    (batch, 4, 84, 84) uint8
              │  normalización: x.float() / 255.0
              ▼
Conv1:    Conv2d(in=4,  out=32, kernel=8, stride=4) + ReLU   →  (batch, 32, 20, 20)
Conv2:    Conv2d(in=32, out=64, kernel=4, stride=2) + ReLU   →  (batch, 64,  9,  9)
Conv3:    Conv2d(in=64, out=64, kernel=3, stride=1) + ReLU   →  (batch, 64,  7,  7)
Flatten:                                                      →  (batch, 3136)
FC1:      Linear(3136, 512) + ReLU
FC2:      Linear(512, 4)  ← Q-values (sin activación final)
```

**Parámetros entrenables:** 1,686,180 (~1.69M)

**Justificación de las decisiones arquitectónicas:**

| Decisión | Justificación |
|---|---|
| **3 capas convolucionales** | El receptive field crece progresivamente: bordes → texturas → formas (paleta, pelota, ladrillos). |
| **Strides grandes (4, 2, 1)** | Reducen rápidamente el tamaño espacial sin perder información. Ideal para imágenes donde los detalles relevantes están a baja frecuencia. |
| **4 frames como canales de entrada** | Captura dinámica temporal (velocidad y dirección de la pelota) sin necesidad de una RNN. |
| **Salida lineal (sin softmax)** | Los Q-values son valores reales no acotados (`ℝ`), no probabilidades. |
| **Sin pooling explícito** | Los strides de las convoluciones hacen el downsampling. Pooling adicional perdería información sobre la posición exacta de objetos pequeños como la pelota. |

#### Componentes del algoritmo DQN

La CNN es el núcleo, pero DQN requiere tres mecanismos adicionales para estabilizar el entrenamiento:

| Componente | Descripción | Implementación |
|---|---|---|
| **Target Network** | Copia "congelada" de la Q-Network que provee los targets del Bellman update. Evita la inestabilidad de bootstrap donde la red persigue su propio output cambiante. | Se sincroniza cada 10,000 pasos (`agent.py` → `update_target()`). Soporta hard copy o soft update (Polyak). |
| **Experience Replay** | Buffer FIFO de 100k transiciones muestreadas uniformemente. Rompe la correlación temporal y permite reutilizar experiencias múltiples veces. | `replay_buffer.py` → `ReplayBuffer` con almacenamiento en `uint8`. |
| **ε-greedy exploration** | Balance entre exploración (acciones aleatorias) y explotación (acción con mayor Q-value). ε decae linealmente de `1.0` a `0.1` durante 500k pasos. | `utils.py` → `linear_epsilon()`. |
| **Huber Loss + Gradient Clipping** | Smooth L1 (Huber) loss es robusta frente a outliers en el TD-error. El clipping de gradientes (norma máx. 10.0) previene actualizaciones desproporcionadas. | `agent.py` → `F.smooth_l1_loss()` + `clip_grad_norm_()`. |

#### Loop de entrenamiento (`train.py`)

```
1. Observar estado s (4 frames preprocesados)
2. Seleccionar acción a con ε-greedy sobre Q(s, ·)
3. Ejecutar a en el entorno → obtener (r, s', done)
4. Almacenar transición (s, a, clip(r), s', done) en el Replay Buffer
5. Cada 4 pasos: muestrear mini-batch de 32 del buffer y actualizar Q-Network
      loss = Huber( Q(s,a) , r + γ · max_a' Q_target(s', a') · (1 - done) )
6. Cada 10,000 pasos: sincronizar Target Network ← Q-Network
7. Repetir hasta completar los pasos configurados
```

El entrenamiento soporta **pausa y reanudación** vía `Ctrl+C` (guarda checkpoint automáticamente) y reanudación con `--resume`.

### 3.2 Arquitectura con Transfer Learning

En DQN para Atari, transfer learning desde ImageNet **no es estándar** — la CNN es pequeña (1.69M params) y los frames de pixel-art de Atari son visualmente muy distintos a fotografías naturales. Sin embargo, la forma de transfer learning que **sí tiene sentido** en este dominio es **cross-game transfer**: reutilizar el encoder convolucional aprendido en un juego Atari para acelerar el aprendizaje en otro.

**Variante propuesta (extensión):**

1. **Preentrenar** la CNN de la sección 3.1 en un juego Atari más simple — por ejemplo `ALE/Pong-v5` — hasta convergencia.
2. **Cargar** los pesos de las 3 capas convolucionales en una nueva instancia para `ALE/Breakout-v5`.
3. **Congelar Conv1** (features de bajo nivel: bordes, contrastes — universales entre juegos Atari) y **fine-tunear Conv2, Conv3 + FC** con el TD-error de Breakout.
4. **Comparar** la curva de aprendizaje vs. el agente entrenado from-scratch.

**Hipótesis:** la inicialización pretrained debería reducir las primeras ~500k transiciones de exploración aleatoria, ya que los filtros de detección de "objetos en movimiento sobre fondo oscuro" se transfieren bien entre Pong y Breakout (ambos comparten estructura visual similar: fondo negro, objetos pequeños en movimiento, paleta controlable).

---

## Quick Start

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd Neural-Nets-ATARI-project

# 2. Crear y activar entorno virtual (Python 3.10+)
python -m venv dqn-env
# Windows: dqn-env\Scripts\Activate.ps1
# Linux/Mac: source dqn-env/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar GPU (opcional)
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 5. Verificar wrappers — debe imprimir (4, 84, 84) uint8
python wrappers.py

# 6. Smoke tests por módulo
python model.py
python replay_buffer.py
python agent.py

# 7. Entrenar desde cero
python main.py train --run-name shared_run

# 8. Pausar: Ctrl+C — guarda checkpoint automáticamente

# 9. Reanudar entrenamiento
python main.py train --run-name shared_run --resume checkpoints/dqn_latest.pt

# 10. Ver métricas en TensorBoard
tensorboard --logdir runs/

# 11. Evaluar el agente
python main.py eval --checkpoint checkpoints/dqn_latest.pt --episodes 10

# 12. Ver al agente jugando
python main.py watch --checkpoint checkpoints/dqn_latest.pt

# 13. Entrenar baseline SB3 para comparación
python main.py baseline --run-name shared_run
```

### Workflow para entrenar por turnos en equipo

```bash
git pull
python main.py train --run-name shared_run --resume checkpoints/dqn_latest.pt
# Ctrl+C cuando termines tu turno
git add checkpoints/dqn_latest.pt runs/shared_run/
git commit -m "train: avancé hasta step XXX"
git push
```

> **Notas:**
> - Solo `checkpoints/dqn_latest.pt` se sube a git; los `dqn_step_N.pt` intermedios quedan locales.
> - Todos deben usar el mismo `--run-name` para tener una curva continua en TensorBoard.

---

## Sobre variantes de DQN

Existen extensiones bien documentadas al DQN estándar que mejoran su rendimiento:

| Variante | Mejora principal | Paper |
|---|---|---|
| **Double DQN** | Reduce la sobreestimación de Q-values separando la selección de acción de la evaluación | Van Hasselt et al. (2016) |
| **Dueling DQN** | Separa la estimación de valor de estado V(s) y ventaja por acción A(s,a) | Wang et al. (2016) |

**Estas variantes no son el foco del proyecto.** El objetivo es implementar y comprender el DQN estándar del paper original (Mnih et al. 2015). Las variantes quedan documentadas como posible trabajo futuro.

---

## Referencias

- Mnih et al. (2015) — *Human-level control through deep reinforcement learning*: https://www.nature.com/articles/nature14236
- Van Hasselt et al. (2016) — *Deep Reinforcement Learning with Double Q-learning*: https://arxiv.org/abs/1509.06461
- Wang et al. (2016) — *Dueling Network Architectures for Deep Reinforcement Learning*: https://arxiv.org/abs/1511.06581
- Machado et al. (2018) — *Revisiting the Arcade Learning Environment*: https://arxiv.org/abs/1709.06009
- CleanRL DQN Atari (referencia de implementación): https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py
- HuggingFace Deep RL Course — Unit 3 (DQN): https://huggingface.co/learn/deep-rl-course/unit3/from-q-to-dqn
- Gymnasium docs: https://gymnasium.farama.org
- ALE docs: https://ale.farama.org

---

![Atari Environments](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/atari-envs.gif)
