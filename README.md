# Motion-Prediction
A motion prediction model trained on the **Lyft's Prediction Dataset** capable of predicting the motion of all agents visible to the ego vehicle at a given point in time(A frame).    

## Model 
The base model is a ***GoogLeNet*** model that is not pretrained. It is customized with multiple Fully Connected convolution layers at the output to meet the requirements to plot the predicted trajectories of the agents. The input layer is also customized to match the output size of the rasterizer. 

## Result
![img](./Assets/Googlenet2_output.gif)

## Setup

1. Download the Lyft's Prediction Dataset and the example codes from the Lyft's [website](https://level-5.global/data/prediction/)
2. Create a new virtual python environment using conda or python env.
**Note:** (Only Part 1 of training dataset was used to train the mote due to RAM limitations)
2. Download the Lyft Python SDK in the new environment. 
3. Reorganize the folders and files downloaded as mentioned below (Refer Directory Structure)
4. The directories titled trained_models is createded to save and load trained prediction models.
5. Paste the files in the /src sub directory of this repository inside the examples directory.
6. Navigate to `/l5kit-1.5.0/examples/agent_motion_precition` and create a directory called *output* to store the results of the model.
6. Change parameters in the config file as fit for your system specifications. 
7. Change the PATH in the prediction python/jupyter notebook scripts, 
```python
os.environ["L5KIT_DATA_FOLDER"] = "/home/akhil/lyft_predict/l5kit-1.5.0"
...
cfg = load_config_data("./agent_motion_config.yaml")
```
## System Specifications

- 16 GB RAM 
- Nvidia RTX 3070 8 GB VRAM
- Ubuntu 20.04 LTS
- 200 GB SSD

***NOTE:***

These are the specifications of the system this model was trained on and is **not the minimum specification** 


## Directory Structure

```bash
Motion_Prediction(any_name)
├── aerial_map
│   ├── aerial_map.png
│   ├── feedback.txt
│   ├── LICENSE
│   └── nearmap_images
├── l5kit-1.5.0
│   ├── dataset_metadata
│   ├── docs
│   ├── examples
│   ├── l5kit
│   ├── meta.json
│   ├── README.md
│   ├── scenes
│   ├── scripts
│   ├── semantic_map
│   └── trained_models
├── sample
│   ├── feedback.txt
│   ├── LICENSE
│   └── sample.zarr
└── semantic_map
    ├── feedback.txt
    ├── LICENSE
    ├── meta.json
    └── semantic_map.pb

```



