import logging, os

logging.basicConfig(format='[%(asctime)s] %(name)s: %(message)s', level=logging.INFO)


RANDOM_SEED = None

SUMMARY_INTERVAL        = 'auto' # int(of iter) or 'auto'
IMAGE_SUMMARY_INTERVAL  = 'auto' # int(of iter) or 'auto'


ROOT_DIR = "."   # change this to the project folder


WEIGHT_ROOT_DIR       = ROOT_DIR+"/weights/"
LOG_ROOT_DIR          = ROOT_DIR+"/logs/"
DATA_ROOT_DIR         = ROOT_DIR+"/data"


CZSL_DS_ROOT = {
    'MIT': DATA_ROOT_DIR+'/mit-states-original',
    'UT':  DATA_ROOT_DIR+'/ut-zap50k-original',
}

GCZSL_DS_ROOT = {
    'MIT': DATA_ROOT_DIR+'/mit-states-natural',
    'UT':  DATA_ROOT_DIR+'/ut-zap50k-natural',
}

GRADIENT_CLIPPING = 5


if not os.path.exists(WEIGHT_ROOT_DIR):
    os.makedirs(WEIGHT_ROOT_DIR)
if not os.path.exists(LOG_ROOT_DIR):
    os.makedirs(LOG_ROOT_DIR)