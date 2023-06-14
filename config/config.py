from yacs.config import CfgNode as CN

_C = CN()

# DATASET
_C.DATASET = CN()
_C.DATASET.TRAIN_IMAGE_DIR = ""
_C.DATASET.TRAIN_LABEL_DIR = ""
_C.DATASET.TEST_IMAGE_DIR = ""
_C.DATASET.TEST_LABEL_DIR = ""
_C.DATASET.BAND_LIST = [1, 2, 3]
_C.DATASET.WINDOW_SIZE = (256, 256)
_C.DATASET.TRAIN_DATASET_SIZE = 1000
_C.DATASET.AUGMENTATION = True
_C.DATASET.CLASS_NUM = 3
_C.DATASET.WITH_LABEL = True
_C.DATASET.LABELS = [0, 1, 2]

# TRAIN
_C.TRAIN = CN()
_C.TRAIN.DEVICE = ""
_C.TRAIN.EPOCH = 100
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.SHUFFLE = False
_C.TRAIN.INPUT_CHANNEL_NUM = 3
_C.TRAIN.OPTIMIZER = 'SGD'
_C.TRAIN.LEARNING_RATE = 0.001
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 0.0004
_C.TRAIN.LOSS_OUT_ITER = 100   # write loss every 100 batch
_C.TRAIN.LOSS_SAVE_DIR = ""
_C.TRAIN.MODEL_SAVE_ITER = 10  # save model every 10 epoch
_C.TRAIN.MODEL_SAVE_DIR = ""

# TEST
_C.TEST = CN()
_C.TEST.DEVICE = ""
_C.TEST.STRIDE = 128
_C.TEST.BATCH_SIZE = 4
_C.TEST.MODEL_LOAD_DIR = ""
_C.TEST.PREDICT_SAVE_DIR = ""
_C.TEST.CONFUSION_MATRIX_SAVE_DIR = ""

# cfg = _C
#
# def get_cfg_defaults():
#     """Get a yacs CfgNode object with default values for my_project."""
#     # Return a clone so that the defaults will not be altered
#     # This is for the "local variable" use pattern
#     return _C.clone()
