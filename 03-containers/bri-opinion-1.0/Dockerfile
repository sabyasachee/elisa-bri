FROM python:3.7

# install necessary python packages
COPY requirements.txt /
RUN pip install --upgrade pip
RUN pip install -r /requirements.txt

# copy services
COPY services /services
COPY data /data
COPY run.sh /

# command to run
# CMD ["bash", "./run.sh"]

# service & web port
EXPOSE 6001