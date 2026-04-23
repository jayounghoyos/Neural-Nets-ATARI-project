# Neural-Nets-ATARI-project

Agente de Deep Reinforcement Learning que aprende a jugar **Breakout (Atari)** desde cero usando DQN.

**Stack:** Python 3.10+, PyTorch, Gymnasium (ALE), Stable-Baselines3

---

## Investigación del Proyecto

### 1. ¿Qué datos obtenemos de la librería y el juego?

#### De Gymnasium (la librería)

| Dato | Descripción |
|------|-------------|
| **Observación (estado)** | Imagen del juego. Raw: `(210, 160, 3)` píxeles RGB. Tras preprocesamiento: `(4, 84, 84)` uint8 — 4 frames en escala de grises apilados. |
| **Recompensa** | Número entero (puntos del juego). En Breakout: `+1` por cada ladrillo destruido. |
| **Terminación** | Señal booleana — episodio terminado (perdiste todas las vidas) o truncado (límite de tiempo). |
| **Espacio de acciones** | 4 acciones discretas: `NOOP`, `FIRE`, `RIGHT`, `LEFT`. |

#### Del juego (ALE/Breakout-v5)

- Frames del emulador Atari a ~60 FPS
- Señal de reward basada en el score del juego
- Estado del entorno en cada timestep



### 2. ¿Por qué DQN?

| Razón | Explicación |
|-------|-------------|
| **Input visual** | El estado es una imagen. DQN usa una CNN que procesa imágenes directamente. Q-learning tabular necesitaría ~10³⁰ entradas — imposible. |
| **Acciones discretas** | Breakout tiene 4 acciones. DQN produce un Q-value por acción y toma el máximo. Policy gradient funciona mejor con acciones continuas (robótica, física). |
| **Reward directo** | Hay una señal de recompensa clara (puntos). DQN es value-based — aprende cuánto vale cada acción, ideal cuando el reward es señal directa del entorno. |
| **Referencia fundacional** | DeepMind (2015) usó exactamente DQN para Atari. Es el algoritmo de referencia para este tipo de problema. Paper: [Nature 2015](https://www.nature.com/articles/nature14236) |
| **Complejidad manejable** | DQN tiene tres componentes claros (CNN + Replay Buffer + Target Network) sin la complejidad de métodos actor-crítico como A3C o PPO. |

#### ¿Por qué no otros algoritmos?

- **PPO / A3C:** mejores para acciones continuas o entornos con reward complejo. Overkill para Breakout con 4 acciones discretas.
- **Q-learning tabular:** imposible — una imagen de 84×84 tiene ~10⁶⁸⁰⁰ estados posibles; no cabe en ninguna tabla.
---

### 3. ¿Por qué ALE/Breakout-v5?

- **Benchmark estándar:** el paper original de DeepMind usó Breakout. Es el caso de estudio más documentado de DQN, lo que facilita comparar resultados.
- **Reward claro:** `+1` por ladrillo destruido. Sin ambigüedad en qué significa mejorar.
- **Dificultad apropiada:** suficientemente complejo para que el agente aprenda estrategia (romper un túnel para que la pelota rebote por encima), pero sin reward disperso que haría el entrenamiento impracticable.
- **Reproducible:** determinista con semilla fija — los experimentos son comparables.
- **Disponible nativamente:** Gymnasium lo provee via ALE (Arcade Learning Environment); no hay que implementar el juego.

---

---

## Estructura del Proyecto tentativa e inspirada con lo que hay en la industria

```
Neural-Nets-ATARI-project/
├── config.py              — Hiperparámetros centralizados
├── wrappers.py            — Preprocesamiento de frame (MaxAndSkipEnv, FireResetEnv)
├── model.py               
├── agent.py     
├── train.py               
├── evaluate.py            — Evaluación de checkpoints
├── utils.py               — plot_rewards()
├── main.py            

```

---

## Referencias

- Mnih et al. (2015) — *Human-level control through deep reinforcement learning*: https://www.nature.com/articles/nature14236
- Van Hasselt et al. (2016) — *Deep Reinforcement Learning with Double Q-learning*: https://arxiv.org/abs/1509.06461
- Wang et al. (2016) — *Dueling Network Architectures for Deep Reinforcement Learning*: https://arxiv.org/abs/1511.06581
- Gymnasium docs: https://gymnasium.farama.org
- ALE docs: https://ale.farama.org
