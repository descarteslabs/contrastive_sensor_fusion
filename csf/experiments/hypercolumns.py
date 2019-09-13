import random
import subprocess as sp
import tensorflow as tf
from tensorflow.keras.layers import Activation, Add, Concatenate, Conv2D, Input, Lambda, UpSampling2D
from tensorflow.keras import Model
import tensorflow.keras.backend as K

from csf.experiments.utils import encoder_head


def hypercolumn_model(size, bands=None, batchsize=8, checkpoint_dir=None):
    """Create a model based on the trained encoder (see encoder.py)
    for semantic segmentation. Can operate on any subset of bands. """
    model_inputs, scaled_inputs, encoded = encoder_head(
        size,
        bands=bands,
        batchsize=batchsize,
        checkpoint_dir=checkpoint_dir
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
    conv = Conv2D(filters=1000, kernel_size=1, activation='relu')(cat)
    out = Conv2D(filters=1, kernel_size=1, activation='sigmoid')(conv)

    return Model(inputs=model_inputs, outputs=[out])


def edge_weighted_binary_crossentropy(
        y_true,
        y_pred,
        edge_weight=2.0,
        interior_weight=1.0,
        exterior_weight=0.5
    ):
    dilation = tf.nn.max_pool2d(y_true, ksize=3, strides=1, padding='SAME')
    interior = 1.0 - tf.nn.max_pool2d(1.0 - y_true, ksize=3, strides=1, padding='SAME')
    edge = dilation - interior
    exterior = 1.0 - dilation

    true_at_edge = y_true * edge
    true_at_interior = y_true * interior
    true_at_exterior = y_true * exterior

    edge_loss = tf.keras.losses.binary_crossentropy(true_at_edge, y_pred * edge)
    interior_loss = tf.keras.losses.binary_crossentropy(true_at_interior, y_pred * interior)
    exterior_loss = tf.keras.losses.binary_crossentropy(true_at_exterior, y_pred * exterior)

    return (
        edge_weight * edge_loss +
        interior_weight * interior_loss +
        exterior_weight * exterior_loss
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


def sensior_fusion_experiment()
    batchsize = 16

    checkpoint_dir = 'gs://dl-appsci/basenets/outputs/basenets_fusion_test_tf2/'
    model = hypercolumn_model(
        size=128,
        bands=('NAIP_nir', 'NAIP_red', 'NAIP_green', 'NAIP_blue'),
        batchsize=batchsize,
        checkpoint_dir=checkpoint_dir
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss=edge_weighted_binary_crossentropy,
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
        )])
    model.evaluate(test_dataset, n_test_samples // batchsize)


def imagenet_comparison_experiment():
    pass # TODO


if __name__ == '__main__':
    sensor_fusion_experiment()
