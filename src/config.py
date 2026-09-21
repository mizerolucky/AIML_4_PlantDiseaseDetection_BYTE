"""Central configuration for FloraLens.

Every script imports its paths and hyperparameters from here so that the
training run, the evaluation run and the ONNX export can never silently
disagree about image size, normalisation or class order.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MANIFEST = DATA_DIR / "manifest.csv"

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
SAMPLES_DIR = RESULTS_DIR / "samples"
WEB_MODEL_DIR = ROOT / "web" / "model"

CHECKPOINT = MODELS_DIR / "floralens_mobilenetv3s.pth"
ONNX_MODEL = MODELS_DIR / "floralens.onnx"
LABELS_JSON = MODELS_DIR / "labels.json"

# ImageNet-pretrained backbone weights, fetched by scripts/fetch_backbone.py.
# Not committed: they are a third-party artifact, not our output.
BACKBONE_WEIGHTS = MODELS_DIR / "mobilenetv3_small_imagenet.pth"
BACKBONE_URL = (
    "https://raw.githubusercontent.com/d-li14/mobilenetv3.pytorch/"
    "master/pretrained/mobilenetv3-small-55df8e1f.pth"
)
BACKBONE_MD5 = "5e9d2d50d7cdc023365f5ef5c6bb5918"

# Class order is fixed here and written into labels.json. The ONNX model, the
# CLI and the web demo all read that file, so the index -> name mapping cannot
# drift between them.
CLASSES = ["Early Blight", "Healthy", "Late Blight"]

# Source folder name in the PlantVillage tree -> our class name.
SOURCE_DIRS = {
    "Potato___Early_blight": "Early Blight",
    "Potato___healthy": "Healthy",
    "Potato___Late_blight": "Late Blight",
}

IMAGE_SIZE = 224

# ImageNet statistics, because the backbone is pretrained on ImageNet.
# The web demo hardcodes these same numbers; keep them in sync.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

EPOCHS = 12
BATCH_SIZE = 32
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
WEIGHT_DECAY = 1e-4
