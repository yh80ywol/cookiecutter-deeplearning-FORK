import abc,os ,prefect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split



class ModelBaseClass_TimeSeries(abc.ABC):
    @abc.abstractmethod  
    def __init__(self, config):
        self.config = config

    def build_model(self, input_shape, model, num_classes): 
        if num_classes == 2:
            last_act_fn = "sigmoid"
            num_classes = 1
        else:
            last_act_fn = "softmax"

        #LeNet Model from -> https://www.mdpi.com/1424-8220/20/6/1693
        if model == "lenet":
            input_layer = keras.layers.Input(input_shape)
            conv1 = keras.layers.Conv1D(filters=2, kernel_size=16)(input_layer)
            conv1 = keras.layers.BatchNormalization()(conv1)
            conv1 = keras.layers.ReLU()(conv1)
            conv1 = keras.layers.AveragePooling1D(4)(conv1)
            conv2 = keras.layers.Conv1D(filters=4, kernel_size=16)(conv1)
            conv2 = keras.layers.BatchNormalization()(conv2)
            conv2 = keras.layers.ReLU()(conv2)
            conv2 = keras.layers.AveragePooling1D(2)(conv2)
            conv3 = keras.layers.Conv1D(filters=16, kernel_size=4)(conv2)
            conv3 = keras.layers.BatchNormalization()(conv3)
            conv3 = keras.layers.ReLU()(conv3)
            conv3 = keras.layers.MaxPooling1D(2)(conv3)
            conv4 = keras.layers.Conv1D(filters=32, kernel_size=2)(conv3)
            conv4 = keras.layers.BatchNormalization()(conv4)
            conv4 = keras.layers.ReLU()(conv4)
            conv4 = keras.layers.MaxPooling1D(2)(conv4)
            conv5 = keras.layers.Conv1D(filters=32, kernel_size=1)(conv4)
            conv5 = keras.layers.BatchNormalization()(conv5)
            conv5 = keras.layers.ReLU()(conv5)
            conv5 = keras.layers.MaxPooling1D(2)(conv5)
            gap = keras.layers.Flatten()(conv5)
            fc1 = keras.layers.Dense(150, activation="relu")(gap)
            fc1 = keras.layers.Dropout(0.3)(fc1)
            fc2 = keras.layers.Dense(100, activation="relu")(fc1)
            fc2 = keras.layers.Dropout(0.3)(fc2)
            output_layer = keras.layers.Dense(num_classes, activation=last_act_fn)(fc2)
            keras_model = keras.Model(input_layer, output_layer)

        #Oberhof Model 4
        if model == "oberhof4":
            input_layer = keras.layers.Input(input_shape)
            conv1 = keras.layers.Conv1D(filters=64, kernel_size=8, activation="relu")(input_layer)
            conv1 = keras.layers.MaxPooling1D(5)(conv1)
            conv1 = keras.layers.Dropout(0.7)(conv1)
            conv2 = keras.layers.Conv1D(filters=80, kernel_size=10, activation="relu")(conv1)
            conv2 = keras.layers.MaxPooling1D(3)(conv2)
            gap = keras.layers.Flatten()(conv2)
            output_layer = keras.layers.Dense(150, activation="relu")(gap)
            output_layer = keras.layers.Dense(100, activation="relu")(output_layer)
            output_layer = keras.layers.Dropout(0.3)(output_layer)
            output_layer = keras.layers.Dense(num_classes, activation=last_act_fn)(output_layer)
            keras_model = keras.Model(input_layer, output_layer)

        #Oberhof Model 5
        if model == "oberhof5":
            input_layer = keras.layers.Input(input_shape)
            conv1 = keras.layers.Conv1D(filters=32, kernel_size=10, activation="relu")(input_layer)
            conv1 = keras.layers.MaxPooling1D(3)(conv1)
            conv2 = keras.layers.Conv1D(filters=32, kernel_size=10, activation="relu")(conv1)
            conv2 = keras.layers.MaxPooling1D(3)(conv2)
            gap = keras.layers.Flatten()(conv2)
            output_layer = keras.layers.Dense(400, activation="relu")(gap)
            output_layer = keras.layers.Dense(150, activation="relu")(output_layer)
            output_layer = keras.layers.Dropout(0.4)(output_layer)
            output_layer = keras.layers.Dense(num_classes, activation=last_act_fn)(output_layer)
            keras_model = keras.Model(input_layer, output_layer)

        #ResNet Model from -> https://github.com/hfawaz/dl-4-tsc
        if model == "resnet":
            n_feature_maps = 64
            input_layer = keras.layers.Input(input_shape)
            # BLOCK 1
            conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
            conv_x = keras.layers.BatchNormalization()(conv_x)
            conv_x = keras.layers.Activation('relu')(conv_x)
            conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
            conv_y = keras.layers.BatchNormalization()(conv_y)
            conv_y = keras.layers.Activation('relu')(conv_y)
            conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
            conv_z = keras.layers.BatchNormalization()(conv_z)
            # expand channels for the sum
            shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
            shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
            output_block_1 = keras.layers.add([shortcut_y, conv_z])
            output_block_1 = keras.layers.Activation('relu')(output_block_1)
            # BLOCK 2
            conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
            conv_x = keras.layers.BatchNormalization()(conv_x)
            conv_x = keras.layers.Activation('relu')(conv_x)
            conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
            conv_y = keras.layers.BatchNormalization()(conv_y)
            conv_y = keras.layers.Activation('relu')(conv_y)
            conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
            conv_z = keras.layers.BatchNormalization()(conv_z)
            # expand channels for the sum
            shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
            shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
            output_block_2 = keras.layers.add([shortcut_y, conv_z])
            output_block_2 = keras.layers.Activation('relu')(output_block_2)
            # BLOCK 3
            conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
            conv_x = keras.layers.BatchNormalization()(conv_x)
            conv_x = keras.layers.Activation('relu')(conv_x)
            conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
            conv_y = keras.layers.BatchNormalization()(conv_y)
            conv_y = keras.layers.Activation('relu')(conv_y)
            conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
            conv_z = keras.layers.BatchNormalization()(conv_z)
            # no need to expand channels because they are equal
            shortcut_y = keras.layers.BatchNormalization()(output_block_2)
            output_block_3 = keras.layers.add([shortcut_y, conv_z])
            output_block_3 = keras.layers.Activation('relu')(output_block_3)
            # FINAL
            gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
            output_layer = keras.layers.Dense(num_classes, activation='softmax')(gap_layer)
            keras_model = keras.Model(input_layer, output_layer)


        keras_model.summary() #if necessary  
        return keras_model

    def compile_model(self, model, opt, loss, metrics, lr):
        if opt.lower() == "sgd":
            optimizer = keras.optimizers.SGD(learning_rate=lr)
        elif opt.lower() == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=lr)
        elif opt.lower() == "rmsprop":
            optimizer = keras.optimizers.RMSprop(learning_rate=lr)
        else:
            raise ValueError("Wrong Input in config.yaml model_structureddata optimizier!")
        model.compile(optimizer=optimizer, loss=loss, metrics=[metrics]) 
    
    def callbacks_model(self, dataset):
        #for more Callbacks see https://keras.io/api/callbacks/ 
        callbacks=[]
        #ModelCheckpoint 
        if self.callbacks.model_checkpoint.usage is True:
            #setup ModelCheckpoint
            ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint(
                            filepath = self.callbacks.model_checkpoint.filepath,
                            monitor = self.callbacks.model_checkpoint.monitor,
                            verbose = self.callbacks.model_checkpoint.verbose,
                            save_best_only = self.callbacks.model_checkpoint.save_best_only,
                            save_weights_only = self.callbacks.model_checkpoint.save_weights_only,
                            mode = self.callbacks.model_checkpoint.mode,
                            save_freq = self.callbacks.model_checkpoint.save_freq,)
            #add to callbacks list
            callbacks.append(ModelCheckpoint)
        #EarlyStopping
        if self.callbacks.early_stopping.usage is True:
            #setup EarlyStopping
            EarlyStopping = tf.keras.callbacks.EarlyStopping(
                                monitor = self.callbacks.early_stopping.monitor,
                                min_delta = self.callbacks.early_stopping.min_delta,
                                patience = self.callbacks.early_stopping.patience,
                                verbose = self.callbacks.early_stopping.verbose,
                                mode = self.callbacks.early_stopping.mode,
                                baseline = self.callbacks.early_stopping.baseline,
                                restore_best_weights = self.callbacks.early_stopping.restore_best_weights,)
            #add to callbacks list
            callbacks.append(EarlyStopping)
        #SuperConvergence (for more details see below Classes CosineAnnealer/OneCycleScheduler)
        if self.callbacks.super_convergence.usage is True:
            #setup SuperConvergence
            steps = np.ceil(len(dataset) / self.batch_size) * self.epochs
            SuperConvergence = OneCycleScheduler(self.learning_rate, steps)
            #add to callbacks list
            callbacks.append(SuperConvergence)
        print("Using following Callbacks:", callbacks)
        return callbacks


class ModelBaseClass_StructuredData(abc.ABC):
    @abc.abstractmethod  
    def __init__(self, config):
        self.config = config

    def build_model(self, all_inputs, encoded_features, model, num_classes): 
        #set params
        if num_classes == 2:
            last_act_fn = "sigmoid"
            num_classes = 1
        else:
            last_act_fn = "softmax"
        
        #Model MLP
        if model == "mlp":
            all_features = keras.layers.concatenate(encoded_features) 
            x = keras.layers.Dense(32, activation="relu")(all_features)
            x = keras.layers.Dropout(0.5)(x)
            output = keras.layers.Dense(num_classes, activation=last_act_fn)(x)
            keras_model = keras.Model(all_inputs, output)

        #keras_model.summary() #if necessary
        return keras_model

    def compile_model(self, model, opt, loss, metrics, lr):
        if opt.lower() == "sgd":
            optimizer = keras.optimizers.SGD(learning_rate=lr)
        elif opt.lower() == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=lr)
        elif opt.lower() == "rmsprop":
            optimizer = keras.optimizers.RMSprop(learning_rate=lr)
        else:
            raise ValueError("Wrong Input in config.yaml model_structureddata optimizier!")
        model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])        

    def callbacks_model(self, dataset):
        #for more Callbacks see https://keras.io/api/callbacks/ 
        callbacks=[]
        #ModelCheckpoint 
        if self.callbacks.model_checkpoint.usage is True:
            #setup ModelCheckpoint
            ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint(
                            filepath = self.callbacks.model_checkpoint.filepath,
                            monitor = self.callbacks.model_checkpoint.monitor,
                            verbose = self.callbacks.model_checkpoint.verbose,
                            save_best_only = self.callbacks.model_checkpoint.save_best_only,
                            save_weights_only = self.callbacks.model_checkpoint.save_weights_only,
                            mode = self.callbacks.model_checkpoint.mode,
                            save_freq = self.callbacks.model_checkpoint.save_freq,)
            #add to callbacks list
            callbacks.append(ModelCheckpoint)
        #EarlyStopping
        if self.callbacks.early_stopping.usage is True:
            #setup EarlyStopping
            EarlyStopping = tf.keras.callbacks.EarlyStopping(
                                monitor = self.callbacks.early_stopping.monitor,
                                min_delta = self.callbacks.early_stopping.min_delta,
                                patience = self.callbacks.early_stopping.patience,
                                verbose = self.callbacks.early_stopping.verbose,
                                mode = self.callbacks.early_stopping.mode,
                                baseline = self.callbacks.early_stopping.baseline,
                                restore_best_weights = self.callbacks.early_stopping.restore_best_weights,)
            #add to callbacks list
            callbacks.append(EarlyStopping)
        #SuperConvergence (for more details see below Classes CosineAnnealer/OneCycleScheduler)
        if self.callbacks.super_convergence.usage is True:
            #setup SuperConvergence
            steps = np.ceil(len(dataset)) * self.epochs
            SuperConvergence = OneCycleScheduler(self.learning_rate, steps)
            #add to callbacks list
            callbacks.append(SuperConvergence)

        print("Using following Callbacks:", callbacks) 
        return callbacks


class ModelBaseClass_Images(abc.ABC):
    @abc.abstractmethod
    def __init__(self, config):
        self.config = config

    def build_model(self, augmentation, rescaling, model, weights, num_classes, input_shape, train_params):
        #set params
        if num_classes == 2:
            last_act_fn = "sigmoid"
            num_classes = 1
        else:
            last_act_fn = "softmax"
        if weights == "none":
            weights = None

        #Model from scratch
        if model == "scratch": #LeNet model trained from "scratch"
            keras_model = keras.Sequential(
                [
                    keras.Input(shape=input_shape),
                    augmentation,
                    rescaling,
                    keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
                    keras.layers.MaxPooling2D(),
                    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
                    keras.layers.MaxPooling2D(),
                    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
                    keras.layers.MaxPooling2D(),
                    keras.layers.Dropout(0.2),
                    keras.layers.Flatten(),
                    keras.layers.Dense(128, activation='relu'),
                    keras.layers.Dense(num_classes, activation = last_act_fn) #sigmoid -> dense(1) classes=2, softmax -> dense(num_classes) classes>2
                ])

        #Transfer learning (for model pool see -> https://keras.io/api/applications/)
        else:
            if model == "xception": #Xception
                base_model = keras.applications.Xception(weights=weights,input_shape=input_shape,include_top=False)
                input_layer = keras.applications.xception.preprocess_input #rescaling
            if model == "resnet50": #ResNet50
                base_model = keras.applications.ResNet50(weights=weights,input_shape=input_shape,include_top=False)
                input_layer = keras.applications.resnet.preprocess_input #zero-center and change RGB to BGR
            if model == "resnet152v2": #ResNet152V2
                base_model = keras.applications.ResNet152V2(weights=weights,input_shape=input_shape,include_top=False)
                input_layer = keras.applications.resnet_v2.preprocess_input #rescaling
            if model == "inceptionv3": #InceptionV3
                base_model = keras.applications.InceptionV3(weights=weights,input_shape=input_shape,include_top=False)
                input_layer = keras.applications.inception_v3.preprocess_input #rescaling
            if model == "mobilenet": #MobileNet
                base_model = keras.applications.MobileNet(weights=weights,input_shape=input_shape,include_top=False)
                input_layer= keras.applications.mobilenet.preprocess_input #rescaling

            if train_params == 1: #retrain ALL params (top layer AND pretrained weights)
                base_model.trainable = True
                inputs = keras.Input(shape=input_shape)
                x = augmentation(inputs)
                x = input_layer(x)
                x = base_model(x)
                x = keras.layers.GlobalAveragePooling2D()(x)
                x = keras.layers.Dropout(0.2)(x)
                outputs = keras.layers.Dense(num_classes, activation = last_act_fn)(x)
                keras_model = keras.Model(inputs, outputs)
            elif train_params == 0: #just train top layer
                base_model.trainable = False
                inputs = keras.Input(shape=input_shape)
                x = augmentation(inputs)
                x = input_layer(x)
                x = base_model(x, training=False)
                x = keras.layers.GlobalAveragePooling2D()(x)
                x = keras.layers.Dropout(0.2)(x)
                outputs = keras.layers.Dense(num_classes, activation = last_act_fn)(x)
                keras_model = keras.Model(inputs, outputs)
            else:
                raise ValueError("Wrong Input in config.yaml train_all_params!")

        keras_model.summary() #if necessary
        return keras_model

    def compile_model(self, model, opt, loss, metrics, lr):
        if opt.lower() == "sgd":
            optimizer = keras.optimizers.SGD(learning_rate=lr)
        elif opt.lower() == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=lr)
        elif opt.lower() == "rmsprop":
            optimizer = keras.optimizers.RMSprop(learning_rate=lr)
        else:
            raise ValueError("Wrong Input in config.yaml model_images optimizier!")
        model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    def callbacks_model(self, dataset):
        #for more Callbacks see https://keras.io/api/callbacks/ 
        callbacks=[]

        #ModelCheckpoint 
        if self.callbacks.model_checkpoint.usage is True:
            #setup ModelCheckpoint
            ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint(
                            filepath = self.callbacks.model_checkpoint.filepath,
                            monitor = self.callbacks.model_checkpoint.monitor,
                            verbose = self.callbacks.model_checkpoint.verbose,
                            save_best_only = self.callbacks.model_checkpoint.save_best_only,
                            save_weights_only = self.callbacks.model_checkpoint.save_weights_only,
                            mode = self.callbacks.model_checkpoint.mode,
                            save_freq = self.callbacks.model_checkpoint.save_freq,)
            #add to callbacks list
            callbacks.append(ModelCheckpoint)
        
        #EarlyStopping
        if self.callbacks.early_stopping.usage is True:
            #setup EarlyStopping
            EarlyStopping = tf.keras.callbacks.EarlyStopping(
                                monitor = self.callbacks.early_stopping.monitor,
                                min_delta = self.callbacks.early_stopping.min_delta,
                                patience = self.callbacks.early_stopping.patience,
                                verbose = self.callbacks.early_stopping.verbose,
                                mode = self.callbacks.early_stopping.mode,
                                baseline = self.callbacks.early_stopping.baseline,
                                restore_best_weights = self.callbacks.early_stopping.restore_best_weights,)
            #add to callbacks list
            callbacks.append(EarlyStopping)

        #SuperConvergence (for more details see below Classes CosineAnnealer/OneCycleScheduler)
        if self.callbacks.super_convergence.usage is True:
            #setup SuperConvergence
            steps = np.ceil(len(dataset)) * self.epochs
            SuperConvergence = OneCycleScheduler(self.learning_rate, steps)
            #add to callbacks list
            callbacks.append(SuperConvergence)

        print("Using following Callbacks:", callbacks)  
        return callbacks


#Super Convergence (-> see https://www.avanwyk.com/tensorflow-2-super-convergence-with-the-1cycle-policy/ or https://arxiv.org/abs/1708.07120)
class CosineAnnealer:
    
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.n = 0
        
    def step(self):
        self.n += 1
        cos = np.cos(np.pi * (self.n / self.steps)) + 1
        return self.end + (self.start - self.end) / 2. * cos

#Super Convergence (-> see https://www.avanwyk.com/tensorflow-2-super-convergence-with-the-1cycle-policy/ or https://arxiv.org/abs/1708.07120)
class OneCycleScheduler(Callback):
    """`Callback` that schedules the learning rate on a 1cycle policy as per Leslie Smith's paper(https://arxiv.org/pdf/1803.09820.pdf).
    If the model supports a momentum parameter, it will also be adapted by the schedule.
    The implementation adopts additional improvements as per the fastai library: https://docs.fast.ai/callbacks.one_cycle.html, where
    only two phases are used and the adaptation is done using cosine annealing.
    In phase 1 the LR increases from `lr_max / div_factor` to `lr_max` and momentum decreases from `mom_max` to `mom_min`.
    In the second phase the LR decreases from `lr_max` to `lr_max / (div_factor * 1e4)` and momemtum from `mom_max` to `mom_min`.
    By default the phases are not of equal length, with the phase 1 percentage controlled by the parameter `phase_1_pct`.
    """

    def __init__(self, lr_max, steps, mom_min=0.85, mom_max=0.95, phase_1_pct=0.3, div_factor=25.):
        super(OneCycleScheduler, self).__init__()
        lr_min = lr_max / div_factor
        final_lr = lr_max / (div_factor * 1e4)
        phase_1_steps = steps * phase_1_pct
        phase_2_steps = steps - phase_1_steps
        
        self.phase_1_steps = phase_1_steps
        self.phase_2_steps = phase_2_steps
        self.phase = 0
        self.step = 0
        
        self.phases = [[CosineAnnealer(lr_min, lr_max, phase_1_steps), CosineAnnealer(mom_max, mom_min, phase_1_steps)], 
                [CosineAnnealer(lr_max, final_lr, phase_2_steps), CosineAnnealer(mom_min, mom_max, phase_2_steps)]]
        
        self.lrs = []
        self.moms = []

    def on_train_begin(self, logs=None):
        self.phase = 0
        self.step = 0

        self.set_lr(self.lr_schedule().start)
        self.set_momentum(self.mom_schedule().start)
        
    def on_train_batch_begin(self, batch, logs=None):
        self.lrs.append(self.get_lr())
        self.moms.append(self.get_momentum())

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step >= self.phase_1_steps:
            self.phase = 1
            
        self.set_lr(self.lr_schedule().step())
        self.set_momentum(self.mom_schedule().step())
        
    def get_lr(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.lr)
        except AttributeError:
            return None
        
    def get_momentum(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.momentum)
        except AttributeError:
            return None
        
    def set_lr(self, lr):
        try:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
            mlflow.log_metric("learning_rate",lr,step=self.step)
        except AttributeError:
            pass # ignore
        
    def set_momentum(self, mom):
        try:
            tf.keras.backend.set_value(self.model.optimizer.momentum, mom)
            mlflow.log_metric("momentum",mom,step=self.step)
        except AttributeError:
            pass # ignore

    def lr_schedule(self):
        return self.phases[self.phase][0]
    
    def mom_schedule(self):
        return self.phases[self.phase][1]
    
    def plot(self):
        ax = plt.subplot(1, 2, 1)
        ax.plot(self.lrs)
        ax.set_title('Learning Rate')
        ax = plt.subplot(1, 2, 2)
        ax.plot(self.moms)
        ax.set_title('Momentum')
        plt.show()

