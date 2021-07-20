from python:3.8

ENV http_proxy http://nl-userproxy-access.net.abnamro.com:8080
ENV https_proxy http://nl-userproxy-access.net.abnamro.com:8080

WORKDIR /AHImageClassifier
COPY . /AHImageClassifier
RUN pip install --trusted-host pypi.python.org --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
CMD ["python", "main2.py"]
