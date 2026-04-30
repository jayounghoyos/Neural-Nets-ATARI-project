# Neural-Nets-ATARI-project

Agente de Deep Reinforcement Learning que aprende a jugar **Breakout (Atari)** desde cero usando DQN.

**Stack:** Python 3.10+, PyTorch, Gymnasium (ALE), Stable-Baselines3 (solo para baseline comparativo)

---

## 1. Problema a resolver

Aprender una política que juegue **Breakout (`ALE/Breakout-v5`)** a partir únicamente de los píxeles del juego, sin reglas codificadas a mano y sin demostraciones humanas.

Formalmente es un **Markov Decision Process (MDP)**:

| Componente | En Breakout |
|------------|-------------|
| Estado `s` | Imagen del juego (frames del emulador) |
| Acción `a` | Una de 4 discretas: `NOOP`, `FIRE`, `RIGHT`, `LEFT` (reduced action space; el joystick Atari completo tiene 18) |
| Recompensa `r` | Puntos por destruir ladrillos — el valor depende de la fila/color (filas superiores rojas valen más). Estándar DeepMind: clip a `{-1, 0, +1}` |
| Terminación | Perder las **5 vidas** o agotar el timeout |
| Objetivo | Maximizar el reward acumulado descontado: `E[Σ γ^t r_t]` con `γ = 0.99` |

A diferencia de un problema supervisado, el agente **nunca ve "ejemplos correctos"**: aprende por ensayo y error a partir de la señal de recompensa.

### ¿Por qué DQN para este problema?

| Razón | Explicación |
|-------|-------------|
| **Input visual** | El estado es una imagen. DQN usa una CNN que procesa imágenes directamente. Q-learning tabular necesitaría ~10⁶⁸⁰⁰ entradas — imposible. |
| **Acciones discretas** | Breakout tiene 4 acciones. DQN produce un Q-value por acción y toma el `argmax`. Policy gradient (PPO/A3C) brilla más con acciones continuas. |
| **Reward directo** | Hay una señal de recompensa clara (puntos). DQN es value-based — aprende cuánto vale cada acción, ideal cuando el reward es señal directa del entorno. |
| **Referencia fundacional** | DeepMind (Mnih et al. 2015) usó exactamente DQN para Atari. Es el algoritmo de referencia para este tipo de problema. |
| **Complejidad manejable** | DQN tiene tres componentes claros (CNN + Replay Buffer + Target Network), sin la complejidad de métodos actor-crítico. |

**¿Por qué no otros algoritmos?**

- **PPO / A3C:** mejores para acciones continuas o entornos con reward complejo. Overkill para Breakout con 4 acciones discretas.
- **Q-learning tabular:** imposible — una imagen de 84×84 tiene ~10⁶⁸⁰⁰ estados posibles; no cabe en ninguna tabla.

### ¿Por qué Breakout-v5 específicamente?

- **Benchmark estándar:** el paper original de DeepMind lo usó. Es el caso de estudio más documentado de DQN, lo que facilita comparar resultados.
- **Reward claro:** puntos por ladrillo destruido (más en filas superiores rojas, menos en filas inferiores). En DQN se clipea a `{-1, 0, +1}` (estándar DeepMind 2015) para estabilizar el entrenamiento.
- **Dificultad apropiada:** suficientemente complejo para que el agente aprenda estrategia (romper un túnel para que la pelota rebote por encima), pero sin reward disperso que haría el entrenamiento impracticable.
- **Reproducible con semilla fija**, aunque `v5` incluye **sticky actions** (`repeat_action_probability=0.25`) por defecto — introducen estocasticidad intencional para evitar que el agente memorice secuencias exactas. Para reproducibilidad estricta tipo paper se pasa `repeat_action_probability=0.0` al `gymn.make`.
- **Disponible nativamente:** Gymnasium lo provee vía ALE (Arcade Learning Environment); no hay que implementar el juego.

---

## 2. Datos

A diferencia de un dataset estático (ImageNet, MNIST), **los datos se generan en línea** mientras el agente interactúa con el entorno. Cada paso produce una transición `(s, a, r, s', done)` que se almacena en el Replay Buffer.

### De Gymnasium / ALE

| Dato | Descripción |
|------|-------------|
| **Observación raw** | `(210, 160, 3)` uint8 RGB — frame del emulador a ~60 FPS |
| **Observación preprocesada** | `(4, 84, 84)` uint8 — 4 frames apilados en escala de grises (ver `wrappers.py`) |
| **Recompensa** | Entero (puntos del juego). En Breakout el valor depende de la fila/color del ladrillo — no es uniforme. En DQN se clipea a `{-1, 0, +1}` (estándar DeepMind) para estabilizar gradientes. |
| **Terminación** | Booleano: episodio terminado (perder las 5 vidas) o truncado (timeout). |
| **Espacio de acciones** | 4 acciones discretas: `NOOP`, `FIRE`, `RIGHT`, `LEFT` (reduced action space). |

### Pipeline de preprocesamiento (en `wrappers.py`)

```
raw frame (210, 160, 3) RGB
  → MaxAndSkipEnv(skip=4)        # 4 acciones repetidas + max-pool de los 2 últimos frames
  → ResizeObservation(84, 84)    # downscale espacial
  → GrayscaleObservation()       # 3 canales → 1
  → FrameStackObservation(4)     # 4 frames consecutivos como canales
final: (4, 84, 84) uint8
```


**¿Por qué cada paso?**

- `MaxAndSkipEnv`: el emulador hace flickering (sprites alternan entre frames pares e impares). Tomar el `max` de 2 frames consecutivos lo elimina. Saltar 4 frames también acelera el entrenamiento ~4× sin perder información relevante.
- `Resize 84×84`: reduce dimensionalidad sin perder estructura del juego.
- `Grayscale`: el color no es informativo en Breakout — los ladrillos importan por su posición, no por su tinte.
- `FrameStack 4`: una imagen sola no tiene velocidad. Apilando 4 frames el agente puede inferir trayectoria de la pelota y dirección de la paleta.

### Almacenamiento durante entrenamiento

Replay Buffer con capacidad de **100k transiciones** guardadas en `uint8` (no `float32`) para que la memoria no explote:

- `(4, 84, 84) × 100k × 1 byte ≈ 2.8 GB` en uint8
- En `float32` serían ~11 GB — inviable en GPU consumer.

La normalización a `[0, 1]` se aplica **dentro de la red** (`x.float() / 255.0` en `forward`), no al guardar.

---

## 3. Arquitectura base

> **Contexto histórico — no se usa en este proyecto.** Esta sección existe solo para responder a la pregunta del profe sobre "arquitectura base". Antes de la era de las CNN, el approach naive habría sido una **MLP densa** sobre los píxeles aplanados (`(4, 84, 84) → flatten 28,224 → FC layers → 4 Q-values`).

No la implementamos ni la comparamos empíricamente porque la CNN del paper de Mnih (sección 4) es **claramente superior** por razones bien establecidas: ignora estructura espacial 2D, no tiene invariancia traslacional, y explota a ~14M parámetros solo en la primera capa (~8× más que la CNN entera). Hacer una comparación empírica MLP vs CNN sería desperdiciar cómputo confirmando un resultado ya conocido desde 2015.

---

## 4. Arquitectura propuesta según la naturaleza de los datos (CNN — DQN paper)

Como los datos son **imágenes con estructura espacial 2D** apiladas en una **dimensión temporal**, la arquitectura natural es una **Convolutional Neural Network**. Replicamos exactamente la del paper de Mnih et al. 2015:

```
Input:    (4, 84, 84) uint8
            │  (normaliza /255.0 dentro del forward)
            ▼
Conv1:    Conv2d(in=4, out=32, kernel=8, stride=4) + ReLU   →  (32, 20, 20)
Conv2:    Conv2d(in=32, out=64, kernel=4, stride=2) + ReLU  →  (64, 9, 9)
Conv3:    Conv2d(in=64, out=64, kernel=3, stride=1) + ReLU  →  (64, 7, 7)
Flatten:                                                      →  3136
FC1:      Linear(3136, 512) + ReLU
FC2:      Linear(512, 4)  (Q-values, sin activación)
```

**Total:** ~1.69M parámetros entrenables.

**Por qué encaja con los datos:**

| Decisión | Justificación |
|----------|---------------|
| **3 capas conv** | Receptive field crece progresivamente: bordes → texturas → formas (paleta, pelota, ladrillos). |
| **Strides grandes (4, 2, 1)** | Reducen rápido el tamaño espacial sin perder información — ideal para imágenes con detalles relevantes a baja frecuencia. |
| **4 frames como canales de entrada** | Captura dinámica temporal (velocidad y dirección de la pelota) sin necesidad de una RNN. |
| **Output lineal (sin softmax/sigmoid)** | Los Q-values son `R` no acotados (no probabilidades). |
| **Sin pooling explícito** | Los strides de las convoluciones ya hacen el downsampling. Pooling adicional perdería información sobre la posición exacta de objetos pequeños como la pelota. |

Esta es la arquitectura que se implementará en `model.py` (Fase 3 del proyecto).

### Componentes adicionales del agente DQN (no son arquitectura, pero acompañan a la red)

- **Target Network:** copia "congelada" de la red Q que se actualiza cada 10k pasos. Estabiliza el target del Bellman update.
- **Replay Buffer:** muestreo i.i.d. de transiciones pasadas (rompe la correlación temporal que mataría a SGD).
- **ε-greedy exploration:** ε decae linealmente de `1.0` a `0.1` durante 500k pasos.
- **Huber loss + gradient clipping:** robustez frente a outliers en TD-error.

---

## 5. Arquitectura con transfer learning

En DQN para Atari, transfer learning desde ImageNet **no es estándar** (el CNN ya es pequeño y los frames de pixel-art son muy distintos a fotos naturales). La forma de TL que **sí tiene sentido** en este dominio es **cross-game transfer**: reutilizar el encoder convolucional aprendido en un juego de Atari para acelerar el aprendizaje en otro.

**Variante propuesta para este proyecto (extensión opcional):**

1. Entrenar primero la arquitectura de la sección 4 en un juego Atari más simple — por ejemplo `ALE/Pong-v5` — hasta convergencia.
2. Cargar los pesos de las **3 capas convolucionales** en una nueva instancia para `Breakout-v5`.
3. **Congelar Conv1** (low-level features: bordes, contrastes — universales entre juegos Atari) y **fine-tunear Conv2, Conv3 + las FC** con el TD-error de Breakout.
4. Comparar curva de aprendizaje vs el agente entrenado from-scratch.

**Hipótesis a verificar:** la inicialización pretrained debería reducir las primeras ~500k transiciones de exploración aleatoria, ya que los filtros de detección de "objetos en movimiento sobre fondo negro" se transfieren bien entre Pong y Breakout. Si el experimento se realiza, va en `baseline_comparison.py` como tercera curva (manual / SB3 / manual+TL).

---

## Referencias

- Mnih et al. (2015) — *Human-level control through deep reinforcement learning*: https://www.nature.com/articles/nature14236
- Van Hasselt et al. (2016) — *Deep Reinforcement Learning with Double Q-learning*: https://arxiv.org/abs/1509.06461
- Wang et al. (2016) — *Dueling Network Architectures for Deep Reinforcement Learning*: https://arxiv.org/abs/1511.06581
- CleanRL DQN Atari (referencia de implementación): https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py
- HuggingFace Deep RL Course — Unit 3 (DQN): https://huggingface.co/learn/deep-rl-course/unit3/hands-on
- Gymnasium docs: https://gymnasium.farama.org
- ALE docs: https://ale.farama.org

---

## Curso de Hugging Face

This image summarises a lot of the ideas of RL, specially for our Atari project.

![alt text](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/atari-envs.gif)
