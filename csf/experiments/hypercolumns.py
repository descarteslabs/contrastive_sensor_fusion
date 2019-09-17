import random
import subprocess as sp
import tensorflow as tf
from tensorflow.keras.layers import Activation, Add, Concatenate, Conv2D, Input, Lambda, UpSampling2D
from tensorflow.keras import Model
import tensorflow.keras.backend as K

from csf.experiments.utils import encoder_head, LRMultiplierAdam


def hypercolumn_model(size, bands=None, batchsize=8, checkpoint_file=None, checkpoint_dir=None):
    """Create a model based on the trained encoder (see encoder.py)
    for semantic segmentation. Can operate on any subset of bands. """
    model_inputs, scaled_inputs, encoded = encoder_head(
        size,
        bands=bands,
        batchsize=batchsize,
        checkpoint_file=checkpoint_file,
        checkpoint_dir=checkpoint_dir,
        trainable=True
    )

    stack1 = encoded['conv2_block2_out']
    stack2 = encoded['conv3_block3_out']
    stack3 = encoded['conv4_block5_out']
    stack4 = encoded['conv5_block3_out']

    up0 = scaled_inputs
    up1 = UpSampling2D(size=(128 // 32), interpolation='bilinear')(stack1)
    up2 = UpSampling2D(size=(128 // 16), interpolation='bilinear')(stack2)
    up3 = UpSampling2D(size=(128 // 8), interpolation='bilinear')(stack3)
    up4 = UpSampling2D(size=(128 // 4), interpolation='bilinear')(stack4)

    cat = Concatenate(axis=-1)([up0, up1, up2, up3, up4])
    #conv = Conv2D(filters=1000, kernel_size=1, activation='relu')(cat)
    conv = cat
    out = Conv2D(filters=1, kernel_size=1, activation='sigmoid', name='dense1')(conv)

    return Model(inputs=model_inputs, outputs=[out])


def boundary_loss(y_true, y_pred):
    true_dilation = tf.nn.max_pool2d(y_true, ksize=3, strides=1, padding='SAME')
    true_interior = 1.0 - tf.nn.max_pool2d(1.0 - y_true, ksize=3, strides=1, padding='SAME')
    true_edge = true_dilation - true_interior
    true_extended = tf.nn.max_pool2d(true_edge, ksize=3, strides=1, padding='SAME')

    pred_dilation = tf.nn.max_pool2d(y_pred, ksize=3, strides=1, padding='SAME')
    pred_interior = 1.0 - tf.nn.max_pool2d(1.0 - y_pred, ksize=3, strides=1, padding='SAME')
    pred_edge = true_dilation - true_interior
    pred_extended = tf.nn.max_pool2d(true_edge, ksize=3, strides=1, padding='SAME')

    y_pred_edge = y_pred * pred_edge
    y_true_edge = y_true * true_edge
    precision = tf.reduce_sum(y_true * true_extended * y_pred * pred_edge) / (tf.reduce_sum(y_pred_edge) + K.epsilon())
    recall = tf.reduce_sum(y_true_edge * y_pred * pred_extended) / (tf.reduce_sum(y_true_edge) + K.epsilon())

    return 1.0 - 2.0 * precision * recall / (K.epsilon() + precision + recall)


def jaccard_loss(y_true, y_pred, smooth=1.0):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1.0 - jac) * smooth


def weighted_boundary_loss(y_true, y_pred, weight=0.5):
    """Weighted average of boundary loss and jaccard (IoU) loss"""
    return (
        weight * boundary_loss(y_true, y_pred) +
        (1.0 - weight) * jaccard_loss(y_true, y_pred)
    )


def recall_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_metric(y_true, y_pred):
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2*(precision * recall) / (precision + recall + K.epsilon())


def get_dataset(remote_prefix):
    features = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/channels': tf.io.FixedLenFeature([], tf.int64),
        'image/colorspace': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/image_data': tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
        'target/height': tf.io.FixedLenFeature([], tf.int64),
        'target/width': tf.io.FixedLenFeature([], tf.int64),
        'target/channels': tf.io.FixedLenFeature([], tf.int64),
        'target/target_data': tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True)
    }

    initial_size = 512
    input_shape = (initial_size, initial_size, 4)
    target_shape = (initial_size, initial_size, 1)

    # We need to upsample NAIP to the target resolution of 0.5m from 1.0m
    upsample_size = 2*initial_size

    # We need to crop our images to the target_size seen by the neural net
    target_size = 128

    def _parse_image_function(example_proto):
        example_features = tf.io.parse_single_example(example_proto, features)
        image = tf.reshape(example_features['image/image_data'], input_shape)
        target = tf.reshape(example_features['target/target_data'], target_shape)
        image /= 128.0
        image -= 1.0
        image = tf.image.resize(image, size=(upsample_size, upsample_size), method='bilinear')
        target = tf.image.resize(target, size=(upsample_size, upsample_size), method='bilinear')
        images = list()
        targets = list()
        for j in range(upsample_size // target_size):
            for i in range(upsample_size // target_size):
                images.append(image[j*target_size: (j+1)*target_size, i*target_size: (i+1)*target_size, :])
                targets.append(target[j*target_size: (j+1)*target_size, i*target_size: (i+1)*target_size, :])
        images = tf.data.Dataset.from_tensors(images)
        targets = tf.data.Dataset.from_tensors(targets)
        return tf.data.Dataset.zip((images, targets))

    tfrecord_paths = sp.check_output(('gsutil', '-m', 'ls', remote_prefix)).decode("ascii").split('\n')
    dataset = tf.data.TFRecordDataset(tfrecord_paths)
    return dataset.interleave(
        _parse_image_function,
        cycle_length=tf.data.experimental.AUTOTUNE,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).unbatch()


def sensor_fusion_experiment():
    batchsize = 8

    #checkpoint_file = 'gs://dl-appsci/basenets/outputs/basenets_fusion_tpu_deploy_1/ckpt-80'
    checkpoint_dir = 'gs://dl-appsci/basenets/outputs/basenets_fusion_tpu_deploy_2' # later?
    model = hypercolumn_model(
        size=128,
        bands=('NAIP_nir', 'NAIP_red', 'NAIP_green', 'NAIP_blue'),
        batchsize=batchsize,
        #checkpoint_file=checkpoint_file,
        checkpoint_dir=checkpoint_dir
    )

    model.compile(
        optimizer=LRMultiplierAdam(
            learning_rate=1e-6,
            clipnorm=1.0,
            multipliers={"dense1": 10.0}
        ),
        loss=weighted_boundary_loss,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            f1_metric,
        ]
    )

    # Provides 4-band NAIP images and targets:
    train_dataset = get_dataset('gs://dl-appsci/cv-sandbox/data/semantic_segmentation/buildings/tfrecords/train-*')
    val_dataset = get_dataset('gs://dl-appsci/cv-sandbox/data/semantic_segmentation/buildings/tfrecords/eval-*')
    test_dataset = get_dataset('gs://dl-appsci/cv-sandbox/data/semantic_segmentation/buildings/tfrecords/test-*')

    train_dataset = train_dataset.shuffle(buffer_size=750).batch(batchsize)
    val_dataset = val_dataset.batch(batchsize)
    test_dataset = test_dataset.batch(batchsize)

    n_train_samples = 12000
    n_val_samples = 1500
    n_test_samples = 1500

    model.fit(
        train_dataset,
        epochs=64,
        steps_per_epoch=n_train_samples // batchsize,
        validation_data=val_dataset,
        validation_steps=n_val_samples // batchsize,
        callbacks=[tf.keras.callbacks.ModelCheckpoint(
            'hypercolumns-{epoch:02d}-{val_f1_metric:.4f}.h5',
            verbose=1,
            mode='max',
            save_weights_only=True,
            save_best_only=True
        )])
    model.evaluate(test_dataset, n_test_samples // batchsize)


def imagenet_comparison_experiment():
    pass # TODO


if __name__ == '__main__':
    sensor_fusion_experiment()
