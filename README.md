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
├───data<br />
│   ├───test<br />
│   └───train<br />
├───env_setup<br />
├───pipelines<br />
└───src<br />
│   ├───evaluate<br />
│   ├───model<br />
│   ├───register<br />
│   ├───scoring<br />
│   ├───training<br />
│   └───util<br />
└───main.py<br />
└───parameters.json<br />

<b>data</b>: This folder contains the train and test split data. Each folder contains a food and a non food image.</br>
<b>env_setup</b>: This folder contains the yaml pipelines needed to provision Azure resources</br>
<b>pipelines</b>: This folder contains the training CI, CD and model registry pipelines</br>
<b>src</b>: This folder contains the source code for the application. It is divided in the following packages</br>
- evaluate: Evaluates model performance
- model: Contains model and model metrics
- register: Used for model registry to Azure Machine Learning workspace
- scoring: Used for scoring the model. 
- training: Used for training the model. This package also contains a [test_train.py](src/training/test_train.py)  file used for unit testing of the training module
- util: Utility classes to help model building, packaging and deployment.
  
<b>main.py</b>: Entry point for the program. Used for running the ML pipeline i.e. training, evaluation and scoring </br>
  <b>parameters.json</b>: This file contains global program parameters for use within the application</br>

<b>Note</b>: The .yml piplines in this repository are just for reference purposes. They are not fully integrated in the source code.
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

Now you can run the project with this command by opening a terminal and navigating to the AHImgClassifier/src folder

```
cd AHImgClassifier/src/
python main.py
```

---

## Running Project using Docker
Pytorch + fastai + python

### Requirements

In order to use the Dockerfile in the project, you must have Docker Engine installed. Instructions
for setting up Docker Engine are
[available on the Docker website](https://docs.docker.com/engine/installation/).

Once docker is installed, open a terminal and run the following command:
```
cd AHImgClassifier

docker build --tag ahclassifier .
docker run -p 8084:8084 ahclassifier
```

The Dockerfile is built using a standard python environment using python==3.8
```
from python:3.8

WORKDIR /AHImageClassifier

# Add requirements file to avoid reinstalling libraries again. Docker will pick these libs from cache next time
ADD ./requirements.txt /AHImageClassifier/requirements.txt
RUN pip install --trusted-host pypi.python.org --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Add all files to working directory
ADD . /AHImageClassifier

# Change working directory to source code
WORKDIR /AHImageClassifier/src

# Run program
CMD ["python3", "-m", "main"]

```
If your network is behind a proxy, then you can add values in the http_proxy and https_proxy variables for e.g.
```
ENV http_proxy <proxyurl:port>
ENV https_proxy <proxyurl:port>
```

#### CUDA requirements

If you have a CUDA-compatible NVIDIA graphics card, you can use a CUDA-enabled
version of the PyTorch image to enable hardware acceleration. To enable gpu, do the following in your project root.
```
cd AHImgClassifier\src\
```
Open the file parameters.json and modify the "device" value from cpu to gpu.
```
"system":
    {
        "device": "gpu"
    }
```

## Support

If you'd like to support this project and all the other open source work on this organization, you can use the following option

### Option: Email

Contact me on titanrulez@gmail.com