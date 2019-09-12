import subprocess as sp
import tensorflow as tf
from tensorflow.keras.layers import Activation, Add, Concatenate, Conv2D, Input, Lambda, UpSampling2D
from tensorflow.keras import Model
import tensorflow.keras.backend as K

from csf.encoder import resnet_encoder


def hypercolumn_model(size, products=None, batchsize=8, checkpoint_dir=None):
    """Create a model based on the trained encoder (see encoder.py)
    for semantic segmentation. Can operate on any subset of products. """

    default_products = ('SPOT', 'NAIP', 'PHR')
    if products is None:
        products = default_products

    encoder = resnet_encoder((size, size, 12))
    if checkpoint_dir is not None:
        weights_path = tf.train.latest_checkpoint(checkpoint_dir)
        checkpoint = tf.train.Checkpoint(encoder=encoder)
        checkpoint.restore(weights_path).expect_partial()
    encoder.trainable = False

    model_inputs = list()
    inputs = list()
    for default_product in default_products:
        if default_product in products:
            model_input = Input(batch_shape=(batchsize, size, size, 4))
            inputs.append(model_input)
            model_inputs.append(model_input)
        else:
            inputs.append(K.zeros(shape=(batchsize, size, size, 4)))

    # Zero pad the band dimensions
    padded_inputs = Concatenate(axis=-1)(inputs)

    # Multiply inputs according to missing products
    encoder_input = Lambda(lambda x: len(default_products) / len(products) * x)(padded_inputs)
    encoded = encoder(encoder_input)

    stack1 = encoded['conv2_block2_out']
    stack2 = encoded['conv3_block3_out']
    stack3 = encoded['conv4_block5_out']
    stack4 = encoded['conv5_block3_out']

    conv0 = Conv2D(
        filters=1,
        kernel_size=1,
        bias_initializer='zeros',
        kernel_initializer='VarianceScaling'
    )(encoder_input)

    up0 = conv0
    up1 = UpSampling2D(size=(128 // 32), interpolation='bilinear')(stack1)
    up2 = UpSampling2D(size=(128 // 16), interpolation='bilinear')(stack2)
    up3 = UpSampling2D(size=(128 // 8), interpolation='bilinear')(stack3)
    up4 = UpSampling2D(size=(128 // 4), interpolation='bilinear')(stack4)

    cat = Concatenate(axis=-1)([up0, up1, up2, up3, up4])
    out = Conv2D(filters=1, kernel_size=1, activation='sigmoid')(cat)
    return Model(inputs=model_inputs, outputs=[out])


def dice_loss(y_true, y_pred, smooth=1.0):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return 1.0 - (2. * intersection + smooth) / (
        K.sum(K.square(y_true), axis=-1) + K.sum(K.square(y_pred), axis=-1) + smooth
    )


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

    input_shape = (128, 128, 4)
    target_shape = (128, 128, 1)

    def _parse_image_function(example_proto):
        example_features = tf.io.parse_single_example(example_proto, features)
        image = tf.reshape(example_features['image/image_data'], input_shape)
        target = tf.reshape(example_features['target/target_data'], target_shape)
        return image, target

    tfrecord_paths = sp.check_output(('gsutil', '-m', 'ls', remote_prefix)).decode("ascii").split('\n')
    dataset = tf.data.TFRecordDataset(tfrecord_paths)
    return dataset.map(_parse_image_function)


if __name__ == '__main__':
    batchsize = 16

    checkpoint_dir = '***REMOVED***outputs/basenets_fusion_test_tf2/'
    model = hypercolumn_model(
        size=128,
        products=('NAIP',),
        batchsize=batchsize,
        checkpoint_dir=checkpoint_dir
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=dice_loss,
        metrics=[
            tf.keras.metrics.MeanIoU(num_classes=2),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.BinaryAccuracy()
        ]
    )

    # Provides 4-band NAIP images and targets:
    train_dataset = get_dataset('***REMOVED******REMOVED***data/semantic_segmentation/buildings_x4/tfrecords/train-*')
    eval_dataset = get_dataset('***REMOVED******REMOVED***data/semantic_segmentation/buildings_x4/tfrecords/eval-*')
    test_dataset = get_dataset('***REMOVED******REMOVED***data/semantic_segmentation/buildings_x4/tfrecords/test-*')

    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batchsize)
    eval_dataset = eval_dataset.batch(batchsize)
    test_dataset = test_dataset.batch(batchsize)

    model.fit(train_dataset, epochs=64, steps_per_epoch=8192 // batchsize,
        validation_data=eval_dataset, validation_steps=4)
    model.evaluate(test_dataset)
