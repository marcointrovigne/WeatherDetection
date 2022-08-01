import tensorflow as tf
from tensorflow.keras import layers, Sequential

def build_convolutional_layer(
        kernel_size,
        strides,
        filters,
        batch_normalization,
        dropout
):
    layer = Sequential()
    layer.add(
        layers.Conv2D(
            kernel_size=kernel_size,
            strides=strides,
            filters=filters,
            padding='SAME'
        )
    )
    if batch_normalization: layer.add(layers.BatchNormalization())
    layer.add(layers.ReLU())
    if dropout > 0.: layer.add(layers.Dropout(dropout))

    return layer


class Patches(layers.Layer):
    def __init__(self, patch_size, num_patches=None):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

    def call(self, images, **kwargs):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims]) if self.num_patches is None else tf.reshape(patches, [-1, self.num_patches, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch, **kwargs):
        projected_patches = self.projection(patch)
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8, num_patches=None):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_patches = num_patches
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        ) if self.num_patches is None else tf.reshape(
            attention, (-1, self.num_patches, self.embed_dim)
        )
        output = self.combine_heads(concat_attention)
        return output