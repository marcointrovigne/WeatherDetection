import pandas as pd
import os

"""
Create Dataframe containing all information for each image
"""


class DataFrame:

    def __init__(self, root_path):

        """
        Get the root path and dictionary of all the outputs
        :param root_path: Path to Algolux_allv3
        """

        self.root_path = root_path
        self.json_path = os.path.join(self.root_path, 'labels_updated')
        self.dict_df = {"daytime": {0: "day", 1: "night"},
                        "roadState": {0: "dry", 1: "wet", 2: "partialSnow", 3: "fullSnow"},
                        "sidewalkState": {0: "clean", 1: "partialSnow", 2: "snowCovered"},
                        "infrastructure": {0: "highway", 1: "inCity", 2: "suburban"},
                        "precipitation": {0: "no_precipitation", 1: "lightSnow", 2: "heavySnow", 3: "rain"},
                        "fog": {0: "no_fog", 1: "lightFog", 2: "denseFog"}}

    def build_df(self, df):

        """
        Build the dataframe from the .json file of each image
        :param df: df from split txt files
        :return: - df according to new dict_df
                 - dictionary
        """

        frame = pd.DataFrame(columns={"file_path",
                                      "Filename",
                                      "daytime",
                                      "precipitation",
                                      "fog",
                                      "roadState",
                                      "sidewalkState",
                                      "infrastructure",
                                      "twilight",
                                      "tunnel",
                                      "point_removed"})

        for i in (-6, 0, 4):
            if i == 0:
                folder_path = os.path.join(self.root_path, 'cam_stereo_left_lut')
            else:
                folder_path = os.path.join(self.root_path, 'cam_stereo_left_lut_history_{}'.format(i))

            for x in range(len(df)):

                file_path = os.path.join(self.json_path, df["filename"][x])
                img_df = pd.read_json(file_path)

                # Get Daytime
                for daytime, index in zip(img_df.loc[:, "daytime"], range(len(img_df.loc[:, "daytime"]))):
                    if daytime == 1.0:
                        daytime_value = img_df.iloc[[index]].index.tolist()[0]

                # Get Precipitation
                if img_df.loc["no", "precipitation"]:
                    precipitation_value = self.dict_df["precipitation"][0]
                elif img_df.loc["yes", "precipitation"]["snow"]["lightSnow"]:
                    precipitation_value = self.dict_df["precipitation"][1]
                elif img_df.loc["yes", "precipitation"]["snow"]["heavySnow"]:
                    precipitation_value = self.dict_df["precipitation"][2]
                elif img_df.loc["yes", "precipitation"]["rain"]:
                    precipitation_value = self.dict_df["precipitation"][3]

                # Get Fog
                if img_df.loc["no", "fog"]:
                    fog_value = self.dict_df["fog"][0]
                elif img_df.loc["yes", "fog"]["lightFog"]:
                    fog_value = self.dict_df["fog"][1]
                elif img_df.loc["yes", "fog"]["denseFog"]:
                    fog_value = self.dict_df["fog"][2]

                # Get roadState
                for roadState, index in zip(img_df.loc[:, "roadState"], range(len(img_df.loc[:, "roadState"]))):
                    if roadState == 1.0:
                        roadState_value = img_df.iloc[[index]].index.tolist()[0]

                # Get sidewalkState
                for sidewalkState, index in zip(img_df.loc[:, "sidewalkState"],
                                                range(len(img_df.loc[:, "sidewalkState"]))):
                    if sidewalkState == 1.0:
                        sidewalkState_value = img_df.iloc[[index]].index.tolist()[0]

                # Get Infrastructure
                for infrastructure, index in zip(img_df.loc[:, "infrastructure"],
                                                 range(len(img_df.loc[:, "infrastructure"]))):
                    if infrastructure == 1.0:
                        infrastructure_value = img_df.iloc[[index]].index.tolist()[0]

                # Get tunnel
                tunnel_value = img_df.loc["day", "tunnel"]

                # Get twilight
                twilight_value = img_df.loc["day", "twilight"]

                # Get point_removed
                point_removed_value = img_df.loc["day", "point_removed"]

                frame = frame.append({'file_path': folder_path,
                                      'Filename': df["filename"][x].replace('.json', '.png'),
                                      'daytime': daytime_value,
                                      'precipitation': precipitation_value,
                                      'fog': fog_value,
                                      'roadState': roadState_value,
                                      'sidewalkState': sidewalkState_value,
                                      'infrastructure': infrastructure_value,
                                      'twilight': twilight_value,
                                      'tunnel': tunnel_value,
                                      'point_removed': point_removed_value}, ignore_index=True)

        return frame, self.dict_df
