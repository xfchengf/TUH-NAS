from yacs.config import CfgNode

CN = CfgNode
_C = CN()

# -----------------------------------------------------------------------------
# SEARCH
# -----------------------------------------------------------------------------
_C.SEARCH = CN()
_C.SEARCH.ARCH_START_EPOCH = 20
_C.SEARCH.SEARCH_ON = False

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.META_ARCHITECTURE = "searchnet"
_C.MODEL.NUM_LAYERS = 4
_C.MODEL.NUM_BLOCKS = 3   # cell里面基本块数量
_C.MODEL.NUM_STRIDES = 3   # ASPPModule中的膨胀率,strides不能超过layers
_C.MODEL.AFFINE = True
_C.MODEL.WEIGHT = ""
_C.MODEL.PRIMITIVES_SPEU = "HSI_SPEU"
_C.MODEL.PRIMITIVES_SPAU = "HSI_SPAU"
_C.MODEL.PRIMITIVES_FFU = "HSI_FFU"
_C.MODEL.ACTIVATION_F = "Leaky"
_C.MODEL.ASPP_RATES = (2, 4, 6)
_C.MODEL.USE_ASPP = True

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.DATA_ROOT = "./dataset"
_C.DATASET.DATA_SET = "Houston2018"
_C.DATASET.CATEGORY_NUM = 20
_C.DATASET.CROP_SIZE = 25
_C.DATASET.PATCHES_NUM = 400
_C.DATASET.MCROP_SIZE = [16, 32, 48]
_C.DATASET.OVERLAP = True
_C.DATASET.SHOW_ALL = True
_C.DATASET.DIST_MODE = 'per'
_C.DATASET.TRAIN_NUM = 30
_C.DATASET.VAL_NUM = 10
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8    # Specify the number of subprocesses used for data loading, which can speed up the data loading process.
_C.DATALOADER.BATCH_SIZE_TRAIN = 2
_C.DATALOADER.BATCH_SIZE_TEST = 2
_C.DATALOADER.DATA_LIST_DIR = "./sample"

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 200
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.WEIGHT_DECAY_BIAS = 0
_C.SOLVER.WEIGHT_DECAY = 0.00004
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.SEARCH = CN()
_C.SOLVER.SEARCH.LR_START = 0.025
_C.SOLVER.SEARCH.LR_END = 0.001
_C.SOLVER.SEARCH.MOMENTUM = 0.9
_C.SOLVER.SEARCH.WEIGHT_DECAY = 0.0003
_C.SOLVER.SEARCH.LR_A = 0.001
_C.SOLVER.SEARCH.WD_A = 0.001
_C.SOLVER.SEARCH.T_MAX = 10
_C.SOLVER.TRAIN = CN()
_C.SOLVER.TRAIN.INIT_LR = 0.1
_C.SOLVER.TRAIN.POWER = 0.9
_C.SOLVER.TRAIN.MAX_ITER = 100000
_C.SOLVER.SCHEDULER = 'poly'
_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.VALIDATE_PERIOD = 1

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
_C.RESULT_DIR = "."
