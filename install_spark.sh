#!/bin/bash

sudo apt-get update && sudo apt-get -y dist-upgrade

sudo apt-get install -y openjdk-8-jdk-headless

wget https://dlcdn.apache.org/spark/spark-3.2.4/spark-3.2.4-bin-hadoop3.2.tgz
tar -xvzf spark-3.2.4-bin-hadoop3.2.tgz
sudo mv spark-3.2.4-bin-hadoop3.2 /usr/local/spark

echo 'export SPARK_HOME=/usr/local/spark' >> ~/.bashrc
echo 'export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin' >> ~/.bashrc

source ~/.bashrc

rm spark-3.2.4-bin-hadoop3.2.tgz

echo "Apache Spark installation and configuration completed."
