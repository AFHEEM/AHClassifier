from python:3.8

# Add proxy if network is behind a firewall
#ENV http_proxy http://abc.com:8080
#ENV https_proxy http://abc.com:8080

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
