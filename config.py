import os

class Config:
    ENV_ID = "ALE/Breakout-v5"


    #Training Hyperparameters
    TOTAL_STEPS = 1000000
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    GAMMA = 0.99
    TRAIN_FREQUENCY = 4


    #Replay Buffer Hyperparameters
    BUFFER_SIZE = 100000
    LEARNING_STARTS = 10000


    #Epsilon hyperparameters
    EPSILON_START = 1.0
    EPSILON_END = 0.1
    EPSILON_DECAY = 500000

    # Target network update frequency
    TARGET_UPDATE_FREAUENCY = 10000

    # Logging and saving
    LOG_DIR = "logs/"
    CHECKPOINT_DIR = "checkpoints/"
    SAVE_FREQUENCY = 100000


    def __init__(self):
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)