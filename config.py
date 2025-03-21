# config.py

# Dataset paths
DATA_DIR = "data/"
TRAIN_DIR = DATA_DIR + "train/"
TEST_DIR = DATA_DIR + "test/"

# Model hyperparameters
INPUT_SHAPE = (1, 48, 48)  # PyTorch uses (channels, height, width)
NUM_CLASSES = 7  # 7 emotion classes
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.0001

# Data augmentation
AUGMENTATION = True
ROTATION_RANGE = 10
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
HORIZONTAL_FLIP = True

# Model saving
SAVE_MODEL_PATH = "models/"