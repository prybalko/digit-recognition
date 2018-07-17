FROM python:2.7.12

# Setup app dirs
RUN mkdir -p /opt/repos/digit-recognition
WORKDIR /opt/repos/digit-recognition

# Copy repo
ADD . .

# Create a virtualenv and install dependencies
RUN virtualenv ve

# Install requirements
RUN ve/bin/pip install -r requirements.txt

EXPOSE 8080
CMD ["ve/bin/python2.7", "run.py"]
