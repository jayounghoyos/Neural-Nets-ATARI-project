# Startup

## 1. Clonar repo

```bash
git clone <url-del-repo>
cd Neural-Nets-ATARI-project
```

## 2. Crear entorno virtual

Python 3.10+. El nombre `dqn-env` ya está en `.gitignore`.

```bash
python -m venv dqn-env
source dqn-env/bin/activate
```

## 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 4. Verificar GPU (opcional pero recomendado)

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## 5. Verificar wrappers

Debe imprimir `(4, 84, 84) uint8` y un episodio random de cientos de steps.

```bash
python wrappers.py
```

## 6. Smoke tests por módulo (opcional)

```bash
python model.py
python replay_buffer.py
python agent.py
```

## 7. Entrenar desde cero

```bash
python main.py train --run-name shared_run
```

## 8. Pausar entrenamiento

Ctrl+C — guarda checkpoint automáticamente.

## 9. Reanudar desde el último checkpoint

```bash
python main.py train --run-name shared_run --resume checkpoints/dqn_latest.pt
```

## 10. Ver TensorBoard

Abrir http://localhost:6006 en el navegador.

```bash
tensorboard --logdir runs/
```

## 11. Evaluar agente

```bash
python main.py eval --checkpoint checkpoints/dqn_latest.pt --episodes 10
```

## 12. Ver al agente jugando

```bash
python main.py watch --checkpoint checkpoints/dqn_latest.pt
```

## 13. Workflow para entrenar por turnos en equipo

```bash
git pull
python main.py train --run-name shared_run --resume checkpoints/dqn_latest.pt
# Ctrl+C cuando termines tu turno
git add checkpoints/dqn_latest.pt runs/shared_run/
git commit -m "train: avancé hasta step XXX"
git push
```

## Notas

- Solo `checkpoints/dqn_latest.pt` se sube a git; los `dqn_step_N.pt` intermedios quedan **locales**.
- Todos deben usar el mismo `--run-name` para curva continua en TensorBoard.
- Hiperparámetros centralizados en `config.py` — overrides por CLI: `--total-steps`, `--lr`, `--seed`, `--device`.
