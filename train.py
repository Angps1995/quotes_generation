import argparse
import os
from glob import glob
import tensorflow as tf
from model import train_input_fn, eval_input_fn
from model_estimator import ppf_foldnet_estimator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.5)
    parser.add_argument('--learning_rate_decay_steps', type=int, default=6000)
    parser.add_argument('--input_shape', nargs='+', required=True,
                        type=int, help='Input shape')
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--train', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str,
                        default=os.environ.get('SM_CHANNEL_TEST'))

    args, _ = parser.parse_known_args()
    tf.logging.set_verbosity(tf.logging.INFO)
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    feature_columns = []
    feature_columns.append(
        tf.feature_column.numeric_column(
            'features', args.input_shape))

    # gpu_options = tf.GPUOptions(allow_growth=True)
    # session_config = tf.ConfigProto(gpu_options=gpu_options)
    session_config = tf.ConfigProto(
        allow_soft_placement=False, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    # strategy = tf.contrib.distribute.MirroredStrategy()
    run_config = tf.estimator.RunConfig(
        session_config=session_config,
        save_checkpoints_steps=300
        # train_distribute=strategy,
        # eval_distribute=strategy
    )
    estimator = ppf_foldnet_estimator(
        feature_columns=feature_columns,
        input_shape=args.input_shape,
        model_dir=args.model_dir,
        lr=args.learning_rate,
        lr_decay_steps=args.learning_rate_decay_steps,
        n_epochs=args.epochs,
        run_config=run_config)

    train_data_files = glob('{}/{}'.format(args.train, '*.h5'))
    val_data_files = glob('{}/{}'.format(args.test, '*.h5'))

    train_hyperparameters = {
        'train_data_filenames': train_data_files,
        'input_shape': args.input_shape,
        'batch_size': args.batch_size,
        'epochs': args.epochs
    }
    train_input_function = lambda: train_input_fn(
        args.train, train_hyperparameters)
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_function,
    )

    eval_val_hyperparameters = {
        'val_data_filenames': val_data_files,
        'batch_size': args.batch_size,
        'input_shape': args.input_shape
    }
    eval_val_input_function = lambda: eval_input_fn(args.test, eval_val_hyperparameters)
    eval_spec = tf.estimator.EvalSpec(
        eval_val_input_function,
        steps=300, throttle_secs=2
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
