from utils import data_generator, get_df, manage_df, prediction, focal_loss
from net import efficientnet
from net.transformer import vit_model
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import os
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input


def get_args():
    parser = argparse.ArgumentParser()

    # Root_path
    parser.add_argument('--root_path', '-r', help='Path to Dataset root directory',
                        default='/media/mintrov/mintrov8TB/Algolux_allv3')

    # Run the script for training or prediction
    parser.add_argument('--training', '-t', help='Run the script for training (default) or for prediction',
                        dest='training', action='store_true')
    parser.add_argument('--prediction', '-p', dest='training', action='store_false')
    parser.set_defaults(training=True)

    # Use single images or concatenated images
    parser.add_argument('--architecture',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=['single', 'concatenated', 'vit'],
                        help='Architectures available: single image, concatenated image or ViT (default: %(default)s)')

    # All parameters to set for the CNN
    parser.add_argument('--height', '-he', help='Size of resized image', default=224, type=int)
    parser.add_argument('--width', '-wi', help='Size of resized image', default=224, type=int)

    # Prediction
    parser.add_argument('--weights_path', '-wp', help='Path of the weights to load for the model architecture',
                        default='/home/mintrov/PycharmProjects/Weather_Classification/weights/'
                                'finalmodel.h5')
    parser.add_argument('--path_image', '-pi', help='Path of the image to see the predicted labels',
                        default='/media/mintrov/mintrov8TB/Algolux_allv3/cam_stereo_left_lut/'
                                '2018-02-07_07-35-52_00100.png')
    parser.add_argument('--type_prediction', '-tp', help='Choose what kind of prediction want to use: report of'
                                                         'the model, print one image, print misclassified labels for'
                                                         'all the images', choices=('report', 'visualization',
                                                                                    'all_images'))
    return parser.parse_args()


def main():
    args = get_args()

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        print("Name:", gpu.name, "  Type:", gpu.device_type)
    # Set GPU preferences
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    tf.device('/gpu:0')

    wandb.init(entity="marcointrovigne", project="weather_classification")
    config = wandb.config

    config.epochs = 20
    config.patience = 4
    config.batch_size = 8
    config.height = args.height
    config.width = args.width
    # EfficientNet
    config.lr = 0.00003143305714230761
    config.dropout = 0.4126860202905518
    config.conv_filters = 256
    config.model_name = 'efficientnet-b2'
    config.width_coefficient = 1.1
    config.depth_coefficient = 1.2
    config.add_layer = False
    config.division = 4
    config.weights = 'imagenet'
    # Concatenated
    config.temporal_length = 5
    # ViT
    config.patch_size = 128
    config.transformer_layers = 12
    config.num_heads = 16
    config.transformer_mlp_depth = 2
    config.dropout = 0.4
    config.embed_dim = 128

    seed = 1234
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Get txt path
    cwd = os.getcwd()
    train_path_txt = os.path.join(cwd, 'train.txt')
    valid_path_txt = os.path.join(cwd, 'valid.txt')
    test_path_txt = os.path.join(cwd, 'test.txt')

    # Create split dataframe from txt files
    train_frame = manage_df.create_split_df(train_path_txt)
    valid_frame = manage_df.create_split_df(valid_path_txt)
    test_frame = manage_df.create_split_df(test_path_txt)

    data = get_df.DataFrame(args.root_path)

    train_df, dict_df = data.build_df(df=train_frame)
    valid_df, dict_df = data.build_df(df=valid_frame)
    test_df, dict_df = data.build_df(df=test_frame)

    dict_df['daytime_alias'] = dict((g, i) for i, g in dict_df['daytime'].items())
    dict_df['precipitation_alias'] = dict((g, i) for i, g in dict_df['precipitation'].items())
    dict_df['fog_alias'] = dict((g, i) for i, g in dict_df['fog'].items())
    dict_df['roadState_alias'] = dict((g, i) for i, g in dict_df['roadState'].items())
    dict_df['sidewalkState_alias'] = dict((g, i) for i, g in dict_df['sidewalkState'].items())
    dict_df['infrastructure_alias'] = dict((g, i) for i, g in dict_df['infrastructure'].items())

    train_df_alias = manage_df.df_to_alias(train_df, dict_df)
    valid_df_alias = manage_df.df_to_alias(valid_df, dict_df)
    test_df_alias = manage_df.df_to_alias(test_df, dict_df)

    train_df_shuffle = train_df_alias.sample(frac=1, random_state=seed).reset_index(drop=True)
    valid_df_shuffle = valid_df_alias.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Different data generator if 'concatenated' architecture is selected
    if args.architecture != 'concatenated':
        img_data_gen = ImageDataGenerator(width_shift_range=0.1,
                                          fill_mode='nearest',
                                          horizontal_flip=True,
                                          brightness_range=[0.5, 1.5])
        train_data_generator = data_generator.DataGenerator(df=train_df_shuffle, img_generator=img_data_gen,
                                                            root_path=args.root_path, height=config.height,
                                                            width=config.width, batch_size=config.batch_size)
        valid_data_generator = data_generator.DataGenerator(df=valid_df_shuffle, root_path=args.root_path,
                                                            height=config.height,
                                                            width=config.width, batch_size=config.batch_size)
        test_data_generator = data_generator.DataGenerator(df=test_df_alias, root_path=args.root_path,
                                                           height=config.height,
                                                           width=config.width, batch_size=1)

        train_gen = train_data_generator.generate_images(for_training=True)
        valid_gen = valid_data_generator.generate_images(for_training=True)
        test_gen = test_data_generator.generate_images(for_training=False)
    else:
        train_data_generator = data_generator.DataGenerator(df=train_df_shuffle,
                                                            root_path=args.root_path, height=config.height,
                                                            width=config.width, batch_size=config.batch_size)
        valid_data_generator = data_generator.DataGenerator(df=valid_df_shuffle, root_path=args.root_path,
                                                            height=config.height,
                                                            width=config.width, batch_size=config.batch_size)
        test_data_generator = data_generator.DataGenerator(df=test_df_alias, root_path=args.root_path,
                                                           height=config.height,
                                                           width=config.width, batch_size=1)

        train_gen = train_data_generator.generate_images_concatenated(for_training=True,
                                                                      temporal_length=config.temporal_length)
        valid_gen = valid_data_generator.generate_images_concatenated(for_training=True,
                                                                      temporal_length=config.temporal_length)
        test_gen = test_data_generator.generate_images_concatenated(for_training=False,
                                                                    temporal_length=config.temporal_length)

    def metrics(num_classes):
        return [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Recall(),
                tf.keras.metrics.Precision(), tfa.metrics.F1Score(num_classes=num_classes, average='weighted'),
                tf.keras.metrics.AUC(curve='pr', num_thresholds=50)]

    losses = {'daytime_output': [focal_loss.categorical_focal_loss(alpha=[[.25, .25]], gamma=2)],
              'precipitation_output': [focal_loss.categorical_focal_loss(alpha=[[.25, .25, .25, .25]], gamma=2)],
              'fog_output': [focal_loss.categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)],
              'roadState_output': [focal_loss.categorical_focal_loss(alpha=[[.25, .25, .25, .25]], gamma=2)],
              'sidewalkState_output': [focal_loss.categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)],
              'infrastructure_output': [focal_loss.categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)]}

    metrics = {'daytime_output': metrics(2),
               'precipitation_output': metrics(4),
               'fog_output': metrics(3),
               'roadState_output': metrics(4),
               'sidewalkState_output': metrics(3),
               'infrastructure_output': metrics(3)}

    if args.architecture == 'single':
        model = efficientnet.EfficientNet(base_efficientnet=True,
                                          width_coefficient=config.width_coefficient,
                                          depth_coefficient=config.depth_coefficient,
                                          conv_filters=config.conv_filters,
                                          dropout=config.dropout,
                                          add_layer=config.add_layer,
                                          division_layer=config.division,
                                          loss=losses,
                                          metrics=metrics,
                                          lr=config.lr,
                                          drop_connect_rate=0.2,
                                          depth_divisor=8,
                                          model_name=config.model_name,
                                          weights=config.weights,
                                          input_tensor=Input(shape=(config.height, config.width, 3)))
        model.summary()
    elif args.architecture == 'concatenated':
        old_model = efficientnet.EfficientNet(base_efficientnet=True,
                                              width_coefficient=config.width_coefficient,
                                              depth_coefficient=config.depth_coefficient,
                                              conv_filters=config.conv_filters,
                                              dropout=config.dropout,
                                              add_layer=config.add_layer,
                                              division_layer=config.division,
                                              loss=losses,
                                              metrics=metrics,
                                              lr=config.lr,
                                              drop_connect_rate=0.2,
                                              depth_divisor=8,
                                              model_name=config.model_name,
                                              weights=config.weights,
                                              input_tensor=Input(shape=(config.height, config.width, 3)))
        configs = old_model.get_config()
        layers_to_modify = ['stem_conv']

        def multify_weights(kernel, out_channels):
            mean_1d = np.mean(kernel, axis=-2).reshape(kernel[:, :, -1:, :].shape)
            tiled = np.tile(mean_1d, (out_channels, 1))
            return tiled

        model = efficientnet.EfficientNet(base_efficientnet=True,
                                          width_coefficient=config.width_coefficient,
                                          depth_coefficient=config.depth_coefficient,
                                          conv_filters=config.conv_filters,
                                          dropout=config.dropout,
                                          add_layer=config.add_layer,
                                          division_layer=config.division,
                                          loss=losses,
                                          metrics=metrics,
                                          lr=config.lr,
                                          drop_connect_rate=0.2,
                                          depth_divisor=8,
                                          model_name=config.model_name,
                                          weights=config.weights,
                                          input_tensor=Input(
                                              shape=(config.height, config.width, 3 * config.temporal_length)))

        for layer in old_model.layers:
            if 'dense' in layer.name:
                break
            if layer.get_weights() != []:
                target_layer = model.get_layer(name=layer.name)
                if layer.name in layers_to_modify:
                    kernels = layer.get_weights()[0]

                    kernels_extra_channel = np.concatenate((kernels,
                                                            multify_weights(kernels, (3 * config.temporal_length) - 3)),
                                                           axis=-2)  # For channels_last
                    target_layer.set_weights([kernels_extra_channel])
                #             target_layer.trainable = False

                else:
                    target_layer.set_weights(layer.get_weights())
        #             target_layer.trainable = False
        model.summary()

    elif args.architecture == 'vit':
        model = vit_model.get_model((config.height, config.width, 3),
                                    head_type='fc',
                                    model_parameters={
                                        'patch_size': config.patch_size,
                                        'transformer_layers': config.transformer_layers,
                                        'num_heads': config.num_heads,
                                        'transformer_mlp_depth': config.transformer_mlp_depth,
                                        'dropout': config.dropout,
                                        'embed_dim': config.embed_dim}
                                    )

    if args.training:
        n_train = len(train_df_shuffle)
        n_valid = len(valid_df_shuffle)
        trained_model = model.fit(x=train_gen,
                                  steps_per_epoch=n_train // config.batch_size,
                                  epochs=config.epochs,
                                  callbacks=[WandbCallback(),
                                             tf.keras.callbacks.EarlyStopping(patience=config.patience,
                                                                              restore_best_weights=True)],
                                  validation_data=valid_gen,
                                  validation_steps=n_valid // config.batch_size)
    else:
        pred = prediction.Pred(args.root_path, model, args.weights_path, test_gen, test_df_alias,
                               dict_df, config.height, config.width)
        cm_daytime, cm_precipitation, cm_fog, cm_roadstate, cm_sidewalk, cm_infrastructure = pred.test_prediction()


if __name__ == "__main__":
    main()
