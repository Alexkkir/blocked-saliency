from kuti import model_helper as mh
from kuti import applications as apps
from kuti import tensor_ops as ops
from kuti import generic as gen
from kuti import image_utils as iu

import pandas as pd, numpy as np, os
from matplotlib import pyplot as plt
from munch import Munch

from tensorflow.keras.models import Model

class MyModel:
    def __init__(self):
        self.data_root = './koniq_api/'
        self.drive_root = './koniq_api/'


        self.ids = pd.read_csv(self.data_root + 'metadata/koniq10k_distributions_sets.csv')

        # Build scoring model
        self.base_model, self.preprocess_fn = apps.get_model_imagenet(apps.InceptionResNetV2)
        head = apps.fc_layers(self.base_model.output, name='fc', 
                            fc_sizes      = [2048, 1024, 256, 1], 
                            dropout_rates = [0.25, 0.25, 0.5, 0], 
                            batch_norm    = 2)    

        self.model = Model(inputs = self.base_model.input, outputs = head)

        # Parameters of the generator
        pre = lambda im: self.preprocess_fn(
                iu.ImageAugmenter(im, remap=False).fliplr().result)
        gen_params = dict(batch_size  = 16,
                        data_path   = 'images/',
                        process_fn  = pre, 
                        input_shape = (384,512,3),
                        inputs      = ['image_name'],
                        outputs     = ['MOS'])

        # Wrapper for the model, helps with training and testing
        self.helper = mh.ModelHelper(self.model, 'KonCept512', self.ids, 
                            loss='MSE', metrics=["MAE", ops.plcc_tf],
                            monitor_metric = 'val_loss', 
                            monitor_mode   = 'min', 
                            multiproc   = True, workers = 5,
                            logs_root   = self.drive_root + 'logs/koniq',
                            models_root = self.drive_root + 'models/koniq',
                            gen_params  = gen_params)

        self.helper.model.load_weights(self.drive_root + 'koncep512-model.h5')

    def __call__(self, im):
        # print("Perfomring")
        # Load an image
        im = self.preprocess_fn(im)

        # Create a batch, of 1 image
        batch = np.expand_dims(im, 0)

        # Predict quality score
        y_pred = self.helper.model.predict(batch).squeeze()
        # print(f'Predicted score: {y_pred:.{2}f}, ground-truth score: {self.ids.MOS.values[0]:.{2}f}')
        return y_pred

# m = MyModel()


