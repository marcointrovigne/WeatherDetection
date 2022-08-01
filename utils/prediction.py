from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt


class Pred:

    def __init__(self, root_path, model, weights_path, test_gen, test_df, dict_df, height, width):
        """
        Prediction
        :param root_path: Path to Algolux_allv3
        :param model: model you want to use
        :param weights_path: Path of the weights you want to use for your model -> Check also if the model is correct
        :param test_gen: test generator
        :param test_df: test df after df_to_alias
        :param dict_df: dictionary
        :param width: height of image
        :param height: width of image
        """
        self.root_path = root_path
        self.model = model
        self.weights_path = weights_path
        self.test_gen = test_gen
        self.test_df = test_df
        self.dict_df = dict_df
        self.height = height
        self.width = width

    def test_prediction(self):
        self.model.load_weights(self.weights_path)

        daytime_pred, precipitation_pred, fog_pred, roadstate_pred, sidewalk_pred, infra_pred = self.model.predict(
            self.test_gen)

        daytime_pred = daytime_pred.argmax(axis=-1)
        precipitation_pred = precipitation_pred.argmax(axis=-1)
        fog_pred = fog_pred.argmax(axis=-1)
        roadstate_pred = roadstate_pred.argmax(axis=-1)
        sidewalk_pred = sidewalk_pred.argmax(axis=-1)
        infra_pred = infra_pred.argmax(axis=-1)

        daytime_true = self.test_df.loc[:, "daytime"]
        daytime_true = np.array(daytime_true)
        precipitation_true = self.test_df.loc[:, "precipitation"]
        precipitation_true = np.array(precipitation_true)
        fog_true = self.test_df.loc[:, "fog"]
        fog_true = np.array(fog_true)
        roadstate_true = self.test_df.loc[:, "roadState"]
        roadstate_true = np.array(roadstate_true)
        sidewalk_true = self.test_df.loc[:, "sidewalkState"]
        sidewalk_true = np.array(sidewalk_true)
        infra_true = self.test_df.loc[:, "infrastructure"]
        infra_true = np.array(infra_true)

        cr_daytime = classification_report(daytime_true, daytime_pred,
                                           target_names=self.dict_df['daytime_alias'].keys())
        print(cr_daytime)

        cr_precipitation = classification_report(precipitation_true, precipitation_pred,
                                                 target_names=self.dict_df['precipitation_alias'].keys())
        print(cr_precipitation)

        cr_fog = classification_report(fog_true, fog_pred,
                                       target_names=self.dict_df['fog_alias'].keys())
        print(cr_fog)

        cr_roadState = classification_report(roadstate_true, roadstate_pred,
                                             target_names=self.dict_df['roadState_alias'].keys())
        print(cr_roadState)

        cr_sidewalkState = classification_report(sidewalk_true, sidewalk_pred,
                                                 target_names=self.dict_df['sidewalkState_alias'].keys())
        print(cr_sidewalkState)

        cr_infrastructure = classification_report(infra_true, infra_pred,
                                                  target_names=self.dict_df['infrastructure_alias'].keys())
        print(cr_infrastructure)

        # Design confusion matrix
        cm_daytime = confusion_matrix(daytime_pred, daytime_true)
        cm_precipitation = confusion_matrix(precipitation_pred, precipitation_true)
        cm_fog = confusion_matrix(fog_pred, fog_true)
        cm_roadstate = confusion_matrix(roadstate_pred, roadstate_true)
        cm_sidewalk = confusion_matrix(sidewalk_pred, sidewalk_true)
        cm_infrastructure = confusion_matrix(infra_pred, infra_true)

        return cm_daytime, cm_precipitation, cm_fog, cm_roadstate, cm_sidewalk, cm_infrastructure

    def confusion_matrix(self, cm, name):
        """
        :param cm: confusion matrix
        :param name: name of what you want to predict, must be in [Daytime, Environment, Weather, Infrastructure]
        """

        if name not in ["daytime", "precipitation", "fog", "roadState", "infrastructure", "sidewalkState"]:
            raise ValueError("name must be in [daytime, precipitation, fog, roadState, infrastructure, sidewalkState]")
        fig = plt.figure(figsize=(16, 14))
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax, fmt='g', cmap="YlGnBu")
        # labels, title and ticks
        ax.set_xlabel('Predicted', fontsize=20)
        ax.xaxis.set_label_position('bottom')
        plt.xticks(rotation=90)
        ax.xaxis.set_ticklabels(self.dict_df["{}_alias".format(name)], fontsize=10)
        ax.xaxis.tick_bottom()

        ax.set_ylabel('True', fontsize=20)
        ax.yaxis.set_ticklabels(self.dict_df["{}_alias".format(name)], fontsize=10)
        plt.yticks(rotation=0)

        plt.title('Refined Confusion Matrix {}'.format(name), fontsize=20)
