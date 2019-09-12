import subprocess as sp
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Conv2D, Dense, Input, Lambda, GlobalMaxPooling2D
from tensorflow.keras import Model
import tensorflow.keras.backend as K

from csf.encoder import resnet_encoder


def classification_model(size, n_labels, products=None, batchsize=8):
    """Create a model based on the trained encoder (see encoder.py)
    for classification. Can operate on any subset of products or bands. """

    default_products = ('SPOT', 'NAIP', 'PHR') # TODO: import band names
    if products is None:
        products = default_products

    encoder = resnet_encoder((size, size, 12))
    encoder.trainable = False

    model_inputs = Input(batch_shape=(batchsize, size, size, 12))

    encoded = encoder(model_inputs)

    stack3 = encoded['conv4_block5_out']
    stack4 = encoded['conv5_block3_out']

    conv3 = Conv2D(filters=64, kernel_size=1)(stack3)
    conv4 = Conv2D(filters=128, kernel_size=1)(stack4)

    pooled3 = GlobalMaxPooling2D()(conv3)
    pooled4 = GlobalMaxPooling2D()(conv4)

    cat = Concatenate(axis=-1)([pooled3, pooled4])
    out = Dense(units=n_labels, activation='sigmoid')(cat)
    return Model(inputs=model_inputs, outputs=[out])


def get_dataset(remote_prefix, n_labels):
    features = {
        'spot_naip_phr': tf.io.FixedLenSequenceFeature([], dtype=tf.string, allow_missing=True),
        'label': tf.io.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True)
    }

    input_shape = (128, 128, 12)
    target_shape = (n_labels,)

    def _parse_image_function(example_proto):
        example_features = tf.io.parse_single_example(example_proto, features)
        image = tf.reshape(tf.io.decode_raw(example_features['spot_naip_phr'], tf.uint8), input_shape)
        target = tf.reshape(tf.one_hot(example_features['label'], depth=n_labels), target_shape)
        return image, target

    tfrecord_paths = sp.check_output(('gsutil', '-m', 'ls', remote_prefix)).decode("ascii").split('\n')
    dataset = tf.data.TFRecordDataset(tfrecord_paths)
    return dataset.map(_parse_image_function)


if __name__ == '__main__':
    batchsize = 8
    n_labels = 12
    model = classification_model(size=128, n_labels=n_labels, products=('SPOT', 'NAIP', 'PHR'), batchsize=batchsize)

    model.compile(
        optimizer=tf.keras.optimizers.Ftrl(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(),
                 tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
    )

    # Provides 4-band NAIP images and OSM labels:
    dataset = get_dataset('gs://dl-appsci/basenets/osm_data/osm_*.tfrecord', n_labels=n_labels)

    train_dataset = dataset.shuffle(buffer_size=1024).batch(batchsize)
    #eval_dataset = eval_dataset.batch(batchsize) # TODO: split dataset
    #test_dataset = test_dataset.batch(batchsize)

    model.fit(train_dataset, epochs=4)
    #model.evaluate(test_dataset)
