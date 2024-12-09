#!/bin/bash

sudo cp /usr/local/spark/conf/spark-env.sh.template /usr/local/spark/conf/spark-env.sh
sudo tee -a /usr/local/spark/conf/spark-env.sh > /dev/null <<EOL
export SPARK_MASTER_HOST=master
export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64
EOL


sudo cp /usr/local/spark/conf/spark-defaults.conf.template /usr/local/spark/conf/spark-defaults.conf
sudo tee -a /usr/local/spark/conf/spark-defaults.conf > /dev/null <<EOL
spark.master spark://master:7077
EOL


echo "Spark configuration completed."
