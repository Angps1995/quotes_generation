import tensorflow as tf
from model import model_fn


def model_estimator(feature_columns, input_shape,
                          lr_decay_steps=3000,
                          lr=0.001, lr_decay=0.5, n_epochs=1,  #epoch originally 10
                          run_config=None, model_dir=None):
    model_estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        config=run_config,
        params={
            'feature_columns': feature_columns,
            'input_shape': input_shape,
            'lr': lr,
            'lr_decay_rate': lr_decay,
            'lr_decay_steps': lr_decay_steps,
            'n_epochs': n_epochs
        })
    
    return model_estimator