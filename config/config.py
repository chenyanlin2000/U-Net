from yacs.config import CfgNode as CN

_C = CN()

# DATASET
_C.DATASET = CN()
_C.DATASET.TRAIN_IMAGE_FOLDER = ""
_C.DATASET.TRAIN_LABEL_FOLDER = ""
_C.DATASET.TEST_IMAGE_FOLDER = ""
_C.DATASET.TEST_LABEL_FOLDER = ""
_C.DATASET.BAND_LIST = [1, 2, 3]
_C.DATASET.WINDOW_SIZE = (256, 256)
_C.DATASET.AUGMENTATION = True
_C.DATASET.CLASS_NUM = 3
_C.DATASET.TRAIN_DATASET_SIZE = 1000
_C.DATASET.WITH_LABEL = True

# TRAIN
_C.TRAIN = CN()
_C.TRAIN.DEVICE = ""
_C.TRAIN.EPOCH = 100
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.SHUFFLE = True
_C.TRAIN.INPUT_CHANNEL = 3
_C.TRAIN.OPTIMIZER = 'SGD'
_C.TRAIN.LEARNING_RATE = 0.001
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 0.0004
_C.TRAIN.LOSS_OUT_IDX = 100  # write loss every 100 idx
_C.TRAIN.LOSS_SAVE_PATH = ""
_C.TRAIN.MODEL_SAVE_ITER = 10  # save model every 10 epoch
_C.TRAIN.MODEL_SAVE_PATH = ""

# TEST
_C.TEST = CN()
_C.TEST.DEVICE = ""
_C.TEST.STRIDE = 128
_C.TEST.BATCH_SIZE = 4
_C.TEST.MODEL_LOAD_PATH = ""
_C.TEST.PREDICT_SAVE_PATH = ""
_C.TEST.CONFUSION_MATRIX_SAVE_PATH = ""

# cfg = _C
#
# def get_cfg_defaults():
#     """Get a yacs CfgNode object with default values for my_project."""
#     # Return a clone so that the defaults will not be altered
#     # This is for the "local variable" use pattern
#     return _C.clone()
