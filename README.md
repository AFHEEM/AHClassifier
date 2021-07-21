<p align="center">
  <p align="center">
    <a href="https://www.aholddelhaize.com/" target="_blank">
      <img src="https://upload.wikimedia.org/wikipedia/en/thumb/c/c8/Ahold_Delhaize_logo.svg/1200px-Ahold_Delhaize_logo.svg.png" alt="JustDjango" height="72">
    </a>
  </p>
  <p align="center">
    Ahold Delhaize Image Classifier
  </p>
</p>

# Ahold DelHaize Image Classifier

Pytorch project using python


## Project Summary

This project processes images and applies deep learning to check whether the image is a food item or not.

---
## Project Structure:
├───.idea
│   ├───azure
│   └───sonarlint
│       └───issuestore
├───data
│   ├───test
│   │   ├───food
│   │   └───nonfood
│   └───train
│       ├───food
│       └───nonfood
├───env_setup
├───pipelines
└───src
├───evaluate
│   └───__pycache__
├───model
│   ├───metrics
│   ├───pickle
│   └───__pycache__
├───register
├───scoring
│   ├───score_test
│   └───__pycache__
├───training
│   └───__pycache__
└───util
└───__pycache__


## Running this project

To get this project up and running you should start by having Python installed on your computer. It's advised you create a virtual environment to store your projects dependencies separately. You can install virtualenv with

```
pip install virtualenv
```

Clone or download this repository and open it in your editor of choice. In a terminal (mac/linux) or windows terminal, run the following command in the base directory of this project

```
virtualenv env
```

That will create a new folder `env` in your project directory. Next activate it with this command on mac/linux:

```
source env/bin/active
```

Then install the project dependencies with

```
pip install -r requirements.txt
```

Now you can run the project with this command by navigating to the AHImgClassifier/src folder

```
cd AHImgClassifier\src\
python main.py
```

---

## Running Project using Docker
Pytorch + fastai + python

### Requirements

In order to use the Dockerfile in the project, you must have Docker Engine installed. Instructions
for setting up Docker Engine are
[available on the Docker website](https://docs.docker.com/engine/installation/).

#### CUDA requirements

If you have a CUDA-compatible NVIDIA graphics card, you can use a CUDA-enabled
version of the PyTorch image to enable hardware acceleration. To enable gpu, do the following in your project root.
```
cd AHImgClassifier\src\
```
Open the file parameters.json and modify the "device" value from cpu to gpu.
```aidl
"system":
    {
        "device": "gpu"
    }
```

## Support

If you'd like to support this project and all the other open source work on this organization, you can use the following option

### Option: Email

Contact me on titanrulez@gmail.com