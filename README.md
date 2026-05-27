# Neural-Nets-ATARI-project

**Equipo:**
- Juan Andrés Young
- Martín Valencia
- Agustín Figueroa

---

Agente de **Deep Reinforcement Learning** que aprende a jugar **Breakout (Atari 2600)** desde cero utilizando **Deep Q-Network (DQN)**, basado en el paper fundacional de Mnih et al. (2015).

El agente no recibe reglas, estrategias ni demostraciones humanas: aprende exclusivamente por ensayo y error a partir de los píxeles crudos del juego y la señal de recompensa del entorno.

**Stack:** Python 3.10+ · PyTorch · Gymnasium (ALE) · TensorBoard

---

## Estructura del proyecto

```
Neural-Nets-ATARI-project/
├── main.py                  # CLI: train / eval / watch
├── config.py                # Hiperparámetros
├── model.py                 # QNetwork (CNN)
├── agent.py                 # DQNAgent
├── replay_buffer.py         # Replay Buffer FIFO
├── wrappers.py              # Preprocesamiento del entorno
├── train.py                 # Loop de entrenamiento
├── evaluate.py              # Evaluación del agente
├── utils.py                 # Utilidades
├── requirements.txt         # Dependencias
├── STARTUP.md               # Guía de setup
├── checkpoints/             # Modelos guardados (.pt)
├── runs/                    # Logs de TensorBoard
└── DocumentationTestAndLearning/  # Papers de referencia
```

---

## 1. Definición del Problema

### ¿Qué problema estamos resolviendo?

Entrenar un agente que aprenda a jugar **Breakout (`ALE/Breakout-v5`)** únicamente a partir de los píxeles del juego, sin reglas codificadas a mano y sin demostraciones humanas. El agente debe descubrir por sí mismo la política óptima por ensayo y error.

### ¿Por qué es relevante?

- **Benchmark fundacional de Deep RL:** el paper de Mnih et al. (2015) demostró por primera vez control a nivel humano en Atari directamente desde píxeles. Es el punto de partida moderno del Deep Reinforcement Learning.
- **Combina visión por computador y RL:** requiere tanto una CNN para procesar frames como un algoritmo de RL para aprender la política.
- **Reproducibilidad y comparabilidad:** existe extensa documentacion, implementaciones de referencia (CleanRL, SB3) y métricas estándar para validar resultados.

### Tipo de tarea: Reinforcement Learning

**Control desde píxeles con acciones discretas.** Formalmente es un **Markov Decision Process (MDP):**

| Componente | En Breakout |
|---|---|
| **Estado** `s` | Imagen del juego — frames del emulador Atari |
| **Acción** `a` | 4 acciones discretas: `NOOP`, `FIRE`, `RIGHT`, `LEFT` |
| **Recompensa** `r` | Puntos por destruir ladrillos. Clipeada a `{-1, 0, +1}` (estándar DeepMind) |
| **Terminación** | Perder las 5 vidas o agotar el timeout del episodio |
| **Objetivo** | Maximizar el retorno descontado: `E[Σ γ^t · r_t]` con `γ = 0.99` |

A diferencia del aprendizaje supervisado, el agente nunca ve "ejemplos correctos" con etiquetas — aprende por ensayo y error a partir de la señal escalar de recompensa.

### ¿Por qué DQN para este problema?

- **Input visual:** el estado es una imagen. DQN usa una CNN que procesa píxeles directamente. Q-learning tabular necesitaría ~10⁶⁸⁰⁰ entradas — imposible.
- **Acciones discretas:** Breakout tiene solo 4 acciones. DQN produce un Q-value por acción y selecciona con `argmax`. Policy gradient (PPO, A3C) es más adecuado para acciones continuas.
- **Reward directo:** hay una señal clara (puntos por ladrillo). DQN es value-based: aprende cuánto vale cada acción.
- **Referencia fundacional:** DeepMind (Mnih et al. 2015) usó exactamente DQN para Atari — algoritmo de referencia.
- **Complejidad manejable:** CNN + Replay Buffer + Target Network, sin la complejidad de métodos actor-crítico.

### ¿Por qué Breakout-v5 específicamente?

- **Benchmark estándar:** el paper original lo usó — facilita comparar resultados.
- **Reward claro:** puntos por ladrillo destruido. El clip a `{-1, 0, +1}` estabiliza el entrenamiento.
- **Dificultad apropiada:** complejo para aprender estrategia (abrir túnel lateral), pero sin reward tan disperso que sea impracticable.
- **Sticky actions** (`repeat_action_probability=0.25`): estocasticidad intencional para evitar memorización de secuencias (Machado et al. 2018).
- **Disponible nativamente:** Gymnasium lo provee vía ALE — no requiere implementar el juego.

> **Sobre variantes (Double DQN, Dueling DQN):** no son el foco. El objetivo es implementar el **DQN estándar** del paper original. Las variantes quedan documentadas como trabajo futuro.

---

## 2. Dataset

### Origen

Los datos se generan **online** desde el entorno **`ALE/Breakout-v5`** provisto por **Gymnasium**. No hay un dataset descargable: cada episodio que juega el agente produce sus propios datos en tiempo real.

### Descripción

- **Inputs (X):** frames del emulador. Raw `(210, 160, 3)` uint8 RGB; preprocesados `(4, 84, 84)` uint8 — 4 frames en escala de grises apilados.
- **"Labels" (y):** *no hay labels*. Solo una señal escalar de recompensa `r ∈ {-1, 0, +1}` y un booleano de terminación.
- **Tamaño del dataset:** dinámico — Replay Buffer FIFO de **100,000 transiciones** `(s, a, r, s', done)`.
- **Modalidad:** **imagen** (pixel-art del emulador Atari).

### Cómo se generan los datos

```
Cada timestep:
  1. El agente observa s
  2. Selecciona acción a (ε-greedy)
  3. El entorno responde con (s', r, done)
  4. La transición (s, a, r, s', done) se almacena en el Replay Buffer
  5. Si done → reset del entorno
```

**Volumen total:** la corrida final llegó a **10M pasos** ≈ **40M frames** del emulador. De estos, solo las últimas 100k transiciones coexisten en memoria.

---

## 3. Estrategia de División de Datos

> En **RL los datos se generan continuamente** por la interacción agente-entorno, así que el esquema clásico train/val/test no aplica directamente. La separación equivalente se hace **por seeds y fases**, no por particiones de un dataset.

### División efectiva

| Fase | Función | Implementación |
|---|---|---|
| **Training** | Transiciones que el agente usa para updates | **Replay Buffer FIFO 100k**, mini-batches de 32 muestreados i.i.d. |
| **Validation** | Episodios periódicos sin actualizar la red | **5 episodios cada 50k pasos**, seeds disjuntas (`SEED + 10000 + ep`), `ε=0.05` |
| **Test** | Evaluación final del modelo entrenado | **10 episodios** con `python main.py eval`, seeds distintas a validación, `ε=0.05` |

Si forzáramos la analogía con un dataset estático, los porcentajes equivalentes serían: **Training ~99%** (todas las transiciones generadas, ~10M), **Validation ~0.5%** (~1000 episodios de eval acumulados), **Test ~0.5%** (10 episodios finales).

### Por qué este "split" tiene sentido

1. **Seeds disjuntas previenen memorización trivial.** Más las sticky actions (`p=0.25`) que introducen estocasticidad.
2. **El Replay Buffer cumple la función de "shuffling".** El muestreo aleatorio rompe la correlación temporal.
3. **Validación durante entrenamiento detecta overfitting al buffer.** Si train sube pero eval se estanca → el agente sobreajusta.

**Estratificación:** no aplica — no hay clases. La "estratificación implícita" la da el muestreo i.i.d. del buffer.

---

## 4. Preprocesamiento de Datos

### Pipeline (`wrappers.py`)

```
raw frame (210, 160, 3) RGB
  → MaxAndSkipEnv(skip=4)         # repite acción 4 frames + max-pool de los 2 últimos
  → RecordEpisodeStatistics()     # registra reward y longitud
  → ResizeObservation(84, 84)     # downscale espacial
  → GrayscaleObservation()        # 3 canales → 1
  → FrameStackObservation(4)      # apila 4 frames como canales
salida: (4, 84, 84) uint8
```

**Justificación de cada paso:**

- **MaxAndSkipEnv** — el emulador produce flickering (sprites alternan). El `max` lo elimina. Saltar 4 frames acelera ~4×.
- **ResizeObservation (84×84)** — reduce dimensionalidad sin perder estructura espacial. Estándar del paper.
- **GrayscaleObservation** — el color no es informativo: la posición de los ladrillos importa, no su tinte.
- **FrameStackObservation (4)** — una imagen sola no tiene velocidad. 4 frames apilados → trayectoria de la pelota sin necesidad de una RNN.

### Normalización y almacenamiento

La normalización a `[0, 1]` se aplica **dentro de la red** (`x.float() / 255.0` en el `forward`), no al guardar. Esto permite almacenar el buffer en `uint8`:

- `(4, 84, 84) × 100k × 1 byte ≈ 2.8 GB` en uint8
- En `float32` serían ~11 GB — inviable en GPUs consumer

### "Augmentation" implícita

No se aplica data augmentation tradicional (flips, crops, noise). La estocasticidad del entorno cumple un rol similar: **sticky actions** (`p=0.25`) previenen memorización + diversidad natural del Replay Buffer.

> **Tokenization:** no aplica — no es un proyecto de NLP.

---

## 5. Arquitectura del Modelo

### Tipo: CNN (Mnih et al. 2015)

Los datos son **imágenes con estructura espacial 2D** apiladas en **dimensión temporal** (4 frames). Replicamos exactamente la arquitectura del paper:

```
Input:    (batch, 4, 84, 84) uint8
              │  normalización: x.float() / 255.0
              ▼
Conv1:    Conv2d(in=4,  out=32, kernel=8, stride=4) + ReLU   →  (batch, 32, 20, 20)
Conv2:    Conv2d(in=32, out=64, kernel=4, stride=2) + ReLU   →  (batch, 64,  9,  9)
Conv3:    Conv2d(in=64, out=64, kernel=3, stride=1) + ReLU   →  (batch, 64,  7,  7)
Flatten:                                                       →  (batch, 3136)
FC1:      Linear(3136, 512) + ReLU
FC2:      Linear(512, 4)  ← Q-values (sin activación final)
```

**Parámetros entrenables:** 1,686,180 (~1.69M)

**Justificación de las decisiones arquitectónicas:**

- **3 capas convolucionales** — el receptive field crece progresivamente: bordes → texturas → formas (paleta, pelota, ladrillos).
- **Strides grandes (4, 2, 1)** — reducen rápido el tamaño espacial sin perder información de baja frecuencia.
- **4 frames como canales** — captura dinámica temporal sin necesidad de una RNN.
- **Salida lineal (sin softmax)** — los Q-values son `ℝ` no acotados, no probabilidades.
- **Sin pooling explícito** — los strides hacen el downsampling. Pooling perdería la posición exacta de la pelota.

> **Embeddings:** no aplica — no hay tokens ni features categóricas.

### Componentes adicionales del agente DQN

| Componente | Función | Implementación |
|---|---|---|
| **Target Network** | Provee targets del Bellman update. Evita inestabilidad de bootstrap | Sync hard cada 10k pasos (`agent.py` → `update_target()`) |
| **Experience Replay** | Buffer FIFO 100k muestreado uniformemente. Rompe correlación temporal | `replay_buffer.py` en `uint8` |
| **ε-greedy exploration** | Balance exploración/explotación. ε decae linealmente 1.0 → 0.1 en 500k pasos | `utils.py` → `linear_epsilon()` |
| **Huber Loss + Grad Clip** | Smooth L1 robusta a outliers. Clip norma 10.0 previene updates desproporcionados | `agent.py` → `F.smooth_l1_loss()` + `clip_grad_norm_()` |

### Transfer Learning: NO

- **ImageNet no aplica:** pixel-art Atari ≠ fotografías naturales — los filtros no transfieren.
- **La CNN es pequeña** (1.69M params) — entrena rápido desde cero, pre-entrenarla no aporta valor.
- **Trabajo futuro:** **cross-game transfer** (preentrenar en `ALE/Pong-v5` y fine-tunear) sí tendría sentido.

---

## 6. Estrategia de Entrenamiento

### Hiperparámetros (centralizados en `config.py`)

| Hiperparámetro | Valor | | Hiperparámetro | Valor |
|---|---|---|---|---|
| **Loss** | Huber (Smooth L1) | | **Total steps** | **10,000,000** |
| **Optimizer** | Adam | | **Train frequency** | cada 4 pasos |
| **Learning rate** | `1e-4` | | **Target sync** | cada 10,000 pasos |
| **Batch size** | 32 | | **Learning starts** | 10,000 pasos |
| **γ (discount)** | 0.99 | | **Gradient clip** | 10.0 |
| **ε schedule** | `1.0 → 0.1` lineal (500k) | | **Reward clipping** | `{-1, 0, +1}` |

> **En RL no hay "epochs" tradicionales** — se mide en pasos del entorno. 10M pasos ≈ 40M frames.

### Regularización

**Sí aplicada:**
- **Gradient clipping** (norma max 10.0) — previene updates desproporcionados.
- **Reward clipping** `{-1, 0, +1}` — estabiliza gradientes entre escalas de score.
- **Target Network** (sync 10k pasos) — estabiliza el bootstrap.

**No aplicada** (justificado):
- **Dropout:** no estándar en DQN — el buffer y la estocasticidad del entorno ya inducen regularización.
- **Weight decay:** Adam sin weight decay es lo recomendado para DQN.
- **LR scheduler:** LR constante a `1e-4` — la literatura no muestra ganancias claras en DQN para Atari.

### Loop de entrenamiento (`train.py`)

```
1. Observar estado s (4 frames preprocesados)
2. Seleccionar acción a con ε-greedy sobre Q(s, ·)
3. Ejecutar a en el entorno → obtener (r, s', done)
4. Almacenar (s, a, clip(r), s', done) en el Replay Buffer
5. Cada 4 pasos: mini-batch de 32 → actualizar Q-Network
      loss = Huber( Q(s,a) , r + γ · max_a' Q_target(s', a') · (1 - done) )
6. Cada 10,000 pasos: sincronizar Target Network ← Q-Network
7. Cada 50,000 pasos: evaluar y loggear a TensorBoard
8. Cada 100,000 pasos: guardar checkpoint
9. Repetir hasta `total_steps`
```

Soporta **pausa y reanudación** vía `Ctrl+C` (guarda checkpoint automáticamente) y `--resume`.

---

## 7. Estrategia de Validación

### Monitoreo en 3 canales

1. **TensorBoard durante entrenamiento** — métricas escalares en tiempo real.
2. **Eval periódica cada 50k pasos** — 5 episodios con seeds disjuntas, `ε=0.05`.
3. **Inspección visual** — `python main.py watch` abre ventana del juego.

### Métricas

- **`episode/reward`** — reward acumulado (puntos del juego sin clipear). **Métrica principal.**
- **`episode/length`** — pasos por episodio. Indicador secundario (episodios más largos correlacionan con mejor política).
- **`train/loss`** — Huber loss del TD-error. Salud del entrenamiento (debe bajar y estabilizarse, no explotar).
- **`train/epsilon`** — ε actual. Sanity check del schedule.
- **`eval/mean_reward`** — reward medio sobre 5 episodios de eval. **Métrica de validación.**
- **`eval/std_reward`** — desviación estándar. Indicador de consistencia (alta varianza → política inestable).

> En RL las métricas tipo **Accuracy / F1 / BLEU no aplican** — son de aprendizaje supervisado. El equivalente es el **mean episode return**.

### Early stopping: NO

- El reward en RL es **no-monótono** (oscila con ε decay y rotación del buffer).
- Detener temprano puede cortar la curva justo antes de un salto de aprendizaje.
- En su lugar: **monitoreo manual** + checkpoints cada 100k pasos (100 checkpoints en total).

### Criterio de selección

**Best `eval/mean_reward` checkpoint** — se elige manualmente desde TensorBoard al final del entrenamiento. `dqn_latest.pt` apunta al más reciente (para reanudación).

---

## 8. EDA Inicial

### La distribución de datos evoluciona durante el entrenamiento

A diferencia de un dataset estático, la "distribución" en RL cambia con la política del agente:

- **Inicio (`ε=1.0`)** — política aleatoria. El agente solo ve estados tempranos del juego, casi nunca alcanza ladrillos altos.
- **Mitad (`ε≈0.5`)** — mezcla de exploración y explotación. Estados intermedios.
- **Final (`ε=0.1`)** — política aprendida. Estados avanzados (pocos ladrillos restantes, túneles).

### Espacio de acciones y recompensas

- **Acciones:** 4 discretas. `FIRE` es **necesaria** para lanzar la pelota tras perder una vida — sin esto el agente puede quedarse atascado.
- **Recompensas raw:** mayoría de pasos `r=0`. Ladrillo destruido: `r ∈ {1, 4, 7}` según la fila. Clipeado a `{0, +1}` (en Breakout no hay rewards negativos).
- **Episodio inicial (random policy):** ~50–200 pasos, reward 0–3.
- **Episodio tras 2M pasos:** ~500–2000 pasos, reward 30–80.
- **Class imbalance:** no aplica (no hay clases). Sí hay **reward sparsity** — la mayoría de transiciones tienen `r=0`.

### Visualizaciones

1. Curva de `episode/reward` vs. `step` — visualización principal del aprendizaje (TensorBoard).
2. Curva de `train/loss` vs. `step` — verificación de estabilidad.
3. Frames del estado preprocesado — `python wrappers.py` imprime forma y dtype.
4. Screenshot del agente entrenado: `latestTraining.png` (score 37 visible).

---

## 9. Pregunta de Investigación y Objetivo

### Pregunta

> **¿Puede una arquitectura DQN estándar (Mnih et al. 2015), entrenada solo con píxeles crudos y sin demostraciones humanas, aprender una política competente para Breakout en un presupuesto computacional alcanzable con hardware consumer?**

### Objetivos

1. **Aprendizaje claro:** curva de `episode/reward` sube sostenidamente, mean reward final ≥ **30 puntos** (random ~ 1-3).
2. **Generalización:** `eval/mean_reward` con seeds disjuntas comparable al entrenamiento (gap < 20%).
3. **Reproducibilidad:** seeds fijas, hiperparámetros en `config.py`, checkpoints versionados.

### Niveles de éxito

| Nivel | Mean reward (eval) | Pasos requeridos |
|---|---|---|
| **Mínimo** (sanity check) | ≥ 10 | ~500k–1M |
| **Decente** | 30–50 | ~2M–5M |
| **Cercano al paper** | 100–300 | 10M+ (DeepMind: ~317 con 200M frames) |

---

## Quick Start

```bash
# 1. Clonar e instalar
git clone https://github.com/jayounghoyos/Neural-Nets-ATARI-project.git
cd Neural-Nets-ATARI-project
python -m venv dqn-env && source dqn-env/bin/activate   # Linux/Mac
pip install -r requirements.txt

# 2. Verificar GPU + smoke tests
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python wrappers.py     # debe imprimir (4, 84, 84) uint8
python model.py
python replay_buffer.py
python agent.py

# 3. Entrenar
python main.py train --run-name shared_run

# 4. Reanudar tras Ctrl+C
python main.py train --run-name shared_run --resume checkpoints/dqn_latest.pt

# 5. Monitorear en TensorBoard
tensorboard --logdir runs/

# 6. Evaluar (sin render, métricas)
python main.py eval --checkpoint checkpoints/dqn_latest.pt --episodes 10

# 7. Ver al agente jugando (con ventana)
python main.py watch --checkpoint checkpoints/dqn_latest.pt
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

> Solo `checkpoints/dqn_latest.pt` va a git; los `dqn_step_N.pt` intermedios quedan locales. Todos usan el mismo `--run-name` para curva continua en TensorBoard.

---

## 10. Referencias

### Papers

- **Mnih, V., et al.** (2015). *Human-level control through deep reinforcement learning.* [Nature 518](https://www.nature.com/articles/nature14236) — paper original de DQN.
- **Van Hasselt, H., et al.** (2016). *Deep Reinforcement Learning with Double Q-Learning.* [arXiv:1509.06461](https://arxiv.org/abs/1509.06461) — reduce sobreestimación de Q-values (trabajo futuro).
- **Wang, Z., et al.** (2016). *Dueling Network Architectures for Deep RL.* [arXiv:1511.06581](https://arxiv.org/abs/1511.06581) — cabezas separadas V(s) y A(s,a) (trabajo futuro).
- **Machado, M., et al.** (2018). *Revisiting the Arcade Learning Environment.* [arXiv:1709.06009](https://arxiv.org/abs/1709.06009) — justifica sticky actions.

### Implementaciones de referencia

- **CleanRL — DQN Atari:** [github.com/vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py) — referencia principal para hiperparámetros.
- **Stable-Baselines3 — DQN:** [stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html) — implementación industrial.

### Documentación y material didáctico

- [Gymnasium](https://gymnasium.farama.org) · [ALE](https://ale.farama.org) · [HuggingFace Deep RL Course Unit 3](https://huggingface.co/learn/deep-rl-course/unit3/from-q-to-dqn)

---

![Atari Environments](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/atari-envs.gif)
