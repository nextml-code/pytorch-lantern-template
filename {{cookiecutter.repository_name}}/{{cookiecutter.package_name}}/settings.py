from pathlib import Path


INPUT_HEIGHT = 32
INPUT_WIDTH = 32
INPUT_SIZE = (INPUT_WIDTH, INPUT_HEIGHT)
INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, 3)


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

TRAIN_CSV = Path("train.csv")
TEST_CSV = Path("test.csv")
