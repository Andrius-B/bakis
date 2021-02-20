
class RunParameterKey:
    def __init__(self, key: str):
        self._key = key
    
    def get(self):
        return self._key

    def __repr__(self):
        return self.get()
    

class R:

    BATCH_SIZE_TRAIN = RunParameterKey("batch_size_train")
    
    BATCH_SIZE_VALIDATION = RunParameterKey("batch_size_validation")

    SHUFFLE_TRAIN = RunParameterKey("shuffle_train")
    
    SHUFFLE_VALIDATION = RunParameterKey("shuffle_validation")

    DATASET_NAME = RunParameterKey("dataset_name")
    
    EPOCHS = RunParameterKey("epochs")

    NUM_CLASSES = RunParameterKey("num_classes")

    MEASUREMENTS = RunParameterKey("measurements")

    OPTIMIZER = RunParameterKey("optimizer")

    WEIGHT_DECAY = RunParameterKey("weight_decay")

    CRITERION = RunParameterKey("criterion")

    # learning rate
    LR = RunParameterKey("lr")

    # one of 'batch', 'epoch', 'finished', 'never'
    TRAINING_VALIDATION_MODE = RunParameterKey("training_validation_mode")

    TEST_WITH_ONE_SAMPLE = RunParameterKey("test_with_one_sample")

    #################################
    ### DISK DATASET CONFIGURATION ##
    #################################

    DISKDS_WINDOW_LENGTH = RunParameterKey("diskds_window_length")

    DISKDS_NUM_FILES = RunParameterKey("diskds_num_files")

    DISKDS_TRAIN_FEATURES = RunParameterKey("diskds_features")

    DISKDS_VALID_FEATURES = RunParameterKey("diskds_features")

    DISKDS_FORMATS = RunParameterKey("diskds_formats")

    DISKDS_WINDOW_HOP_TRAIN = RunParameterKey("diskds_window_hop_train")

    DISKDS_WINDOW_HOP_VALIDATION = RunParameterKey("diskds_window_hop_train")