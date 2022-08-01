from tensorflow.keras import layers, Sequential, Model, Input, losses, optimizers, metrics
from tensorflow.keras.layers import Add, LayerNormalization, Dropout, Dense
import build


def _get_heads_fn(
        classes,
        inputs
):
    inputs = layers.Flatten()(inputs)
    return [
        layers.Dense(units=class_cardinality, activation='softmax', name=class_name)(inputs)
        for class_name, class_cardinality in classes.items()
    ]


def get_heads(
        classes,
        inputs,
        heads_parameters
):
    return _get_heads_fn(classes, inputs, **heads_parameters)


def _get_vit(
        inputs,
        patch_size=64,
        transformer_layers=1,
        num_heads=8,
        transformer_mlp_depth=1,
        dropout=0.1,
        embed_dim=64
):
    transformer_mlp_units = embed_dim
    num_patches = (inputs.shape[-3] // patch_size) * (inputs.shape[-2] // patch_size)

    patches = build.Patches(patch_size, num_patches=num_patches)(inputs)
    encoded_patches = build.PatchEncoder(num_patches, embed_dim)(patches)

    for _ in range(transformer_layers):
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = build.MultiHeadSelfAttention(
            num_heads=num_heads,
            embed_dim=embed_dim,
            num_patches=num_patches
        )(x1)
        x2 = Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = Sequential(
            [
                Sequential(
                    [
                        Dense(
                            units=transformer_mlp_units,
                            activation='relu'
                        ),
                        Dropout(dropout)
                    ]
                )
                for _ in range(transformer_mlp_depth)
            ]
        )(x3)
        # mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)

    model = Model(inputs=inputs, outputs=representation, name='ViT')
    return model(inputs)


def get_body(inputs, model_parameters):
    return _get_vit(inputs, **model_parameters)


def get_model(
        input_shape,
        classes={
            'daytime_output': 2,
            'precipitation_output': 4,
            'fog_output': 3,
            'roadState_output': 4,
            'sidewalkState_output': 3,
            'infrastructure_output': 3
        },
        head_type='fc',  # either fc or conv or conv_deep
        heads_parameters={},
        model_name='custom',
        model_parameters={},
        optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                                  epsilon=1e-08),
        loss=(losses.CategoricalCrossentropy(),
              losses.CategoricalCrossentropy(),
              losses.CategoricalCrossentropy(),
              losses.CategoricalCrossentropy(),
              losses.CategoricalCrossentropy(),
              losses.CategoricalCrossentropy()),
        metrics_list=(
                metrics.AUC(curve='PR'),
                metrics.Precision(),
                metrics.Recall(),
                metrics.CategoricalAccuracy()
        )
):
    inputs = Input(shape=input_shape, name='image')

    hidden_layers = get_body(inputs, model_parameters)

    outputs = get_heads(classes, hidden_layers, heads_parameters)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=optimizer,
        loss=list(loss),
        metrics=list(metrics_list)
    )

    return model
