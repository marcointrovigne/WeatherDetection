Adverse Weather Conditions Detection
====================================
This project proposes a novel deep learning framework to categorize weather conditions, road conditions and environment 
understanding for autonomous vehicles in adverse or regular cases. Weather significantly impacts driver behaviour,
vehicle performance, surface friction, and road-way infrastructure, raising the chance of an accident.
The contributions of this project are as follows. First, a model capable of accurately identifying weather conditions,
road and surrounding environment conditions was constructed. Specifically, using the EfficientNet, it was feasible 
to create an architecture that can be efficiently applied in real-time settings to provide choices for autonomous
cars with rapid, precise detection capacity. This network’s output comprises six categories: 

- Daytime;
- Precipitation;
- Fog;
- Road condition;
- Roadside condition;
- Scene-setting. 

Then, a dataset containing 12997 images collected in Northern Europe for a total of 10,000km travelled was suitably 
relabeled to account for the categories described above, allowing the possibility of simultaneously classifying
the external weather conditions, the road and the surrounding environment through a single architecture.

<img src="./images/hierarchy.png" width="500">

##### Diagram of new label hierarchy. 

## Getting Started

Clone the benchmark code.
```
git clone ""
cd weather_classification
```

For running the evaluation and visualization code you need Python with the following packages:
- tensorflow
- numpy
- cv2
- argparse
- wandb
- pandas
- matplotlib

We provide a conda environment to run our code.
```
conda env create -f environment.yml
```

Activate the conda environment.
```
conda activate WeatherClassificattion
```