# The Below content is from a project conducted during my Masters at CCT College Dublin

To make things easier I've written a series of scripts to get the EC2 instances up and running smoothly. (after much experimentation)

**Note:** Replace YOURKEY and IP with your ssh key and public ip of your AWS ssh key and public IP address of your instance:

## Installing Spark on each instance

1) scp -i YOURKEY .\install_spark.sh ubuntu@IP:~/

2) chmod +x install_spark.sh

3) sudo apt-get update

4) sudo apt-get install dos2unix

5) dos2unix install_spark.sh

6) ./install_spark.sh

If successful you will get the message: "Apache Spark installation and configuration completed"

Note: Your server may need to restart

## Configuring Spark

To configure spark we need to add the **private** IPs of each node to /etc/hosts on each machine

Use nano to add IP Name to the end of the file

e.g. 172.1.24.3 master

Do this for master, worker1, worker2 etc.

Follow same instructions as the install_spark.sh to scp the configure_spark.sh script onto the server, convert to unix format and run it.

Once it runs you should see the message "Spark configuration completed."

## AWS Security Group Settings

There are a whole bunch of ports used for Spark to communicate with your cluster

The best way to ensure this is to allow ALL TRAFFIC for your private IP range in the inbound rules.

For example if I have a master node with a private IP of 172.31.44.56 , I will add a rule allowing all traffic within 172.31.0.0/16 which will allow all IPs on that VPC to communicate.

Alternatively you would have to check the logs of the spark cluster and enable each port but this is quite time consuming and likely to change each time you run the cluster.

Also ensure to add port 8080 and 4040 for your IP so that you can view the Web UI of Spark on your master node 

## Configuring SSH from Master to Workers

1) ssh-keygen -t rsa -P ""

2) SSH into each worker and set a password using: sudo passwd ubuntu

3) Change the ssh config file on each worker to set these variables:

   * PubkeyAuthentication yes
   * PasswordAuthentication yes
  
4) Restart ssh process with: sudo systemctl restart ssh

5) Now SSH into your master node and copy your newly created ssh public key to each worker using: ssh-copy-id -i ~/.ssh/id_rsa.pub ubuntu@worker1

6) Finally while your in the master node try to ssh into each worker using: ssh worker1 etc.

   * Use the password you gave each node
   * Note first time it might ask you to accept the fingerprint, enter yes

## Starting up your cluster

1) Enter the command start-master.sh on your master node
2) On each worker node enter the command: start-worker.sh spark://master:7077

Once this is done connect to the Web UI on your master node by using it's public IP address and port 8080

* http://54.167.213.195:8080/

You should see the Spark UI and your worker nodes

## Running the test cluster script

I have written a simple script that sums all numbers from 1 to 1000.

**Note** : You need to convert the jupyter notebook to a py file to get it to work with Spark

1) Run sudo apt-get update
2) Install pip using sudo apt install python3-pip
3) Install nbconvert using pip install nbconvert
4) Install jupyter using sudo apt install jupyter

Now to convert and run it
1) scp the script onto the master node
2) jupyter nbconvert --to=python testCluster.ipynb
3) Enter the command: spark-submit testCluster.py
4) You should see the job running live on the logs in the master node, and also in the Web UI
5) Eventually you should get the answer back within the log: 500500

## Copying dataset to master (REDUNDANT)

1) scp -i "CCT_CA1.pem" .\Dataset.zip ubuntu@IP.compute-1.amazonaws.com:~/

**Note:** The data is about 1GB, might take a while to copy

## S3 Storage

While attempting to get the cluster to train the neural network it became clear that I needed a common data storage.
In hindsight implemenmting a HDFS would make sense but due to time constraints S3 was chosen.

In summary:
1) Create an S3 Bucket
2) Create a group allowing access to S3 Bucket
3) Create a user in that group and generate their Key and Secret
4) On each instance run the following:

   * sudo apt-get update
   * sudo apt-get install awscli
   * aws configure (Enter your Key and Secret)
   * Enter command aws s3 ls to test you can see the S3 Bucket
   
## Running the train neural network script

1) scp the requirements.txt to the master node and all worker nodes
2) run: pip install -r requirements.txt
3) scp the train_neural_network.ipynb onto the master server
4) jupyter nbconvert --to=python train_neural_network.ipynb
5) spark-submit train_neural_network.py

## Resizing EBS Volume

I quickly found that the filesystem space was taking up 90+ percent of the storage

Resize your instance to 12GB in AWS

Then on your master node, run:

1) sudo lsblk

You should see the partitions of your file system:

e.g. 
ubuntu@ip-172-31-37-0:~$ sudo lsblk
NAME     MAJ:MIN RM   SIZE RO TYPE MOUNTPOINTS
loop0      7:0    0  24.6M  1 loop /snap/amazon-ssm-agent/7528
loop1      7:1    0  24.4M  1 loop /snap/amazon-ssm-agent/6312
loop2      7:2    0  55.6M  1 loop /snap/core18/2745
loop3      7:3    0  55.7M  1 loop /snap/core18/2790
loop4      7:4    0  63.3M  1 loop /snap/core20/1879
loop5      7:5    0  63.5M  1 loop /snap/core20/2015
loop6      7:6    0 111.9M  1 loop /snap/lxd/24322
loop7      7:7    0  53.2M  1 loop /snap/snapd/19122
loop8      7:8    0  40.8M  1 loop /snap/snapd/20092
xvda     202:0    0    12G  0 disk
├─xvda1  202:1    0   7.9G  0 part /
├─xvda14 202:14   0     4M  0 part
└─xvda15 202:15   0   106M  0 part /boot/efi

2) sudo growpart /dev/xvda 1
3) sudo e2fsck -f /dev/xvda1
4) sudo resize2fs /dev/xvda1
5) df -h

You should see you have more space now:

ubuntu@ip-172-31-37-0:~$ df -h
Filesystem      Size  Used Avail Use% Mounted on
/dev/root        12G  7.4G  4.2G  64% /
tmpfs           2.0G     0  2.0G   0% /dev/shm
tmpfs           781M  840K  781M   1% /run
tmpfs           5.0M     0  5.0M   0% /run/lock
/dev/xvda15     105M  6.1M   99M   6% /boot/efi
tmpfs           391M  4.0K  391M   1% /run/user/1000


## Troubleshoot

ImportError: libGL.so.1: cannot open shared object file - sudo apt-get install libgl1-mesa-glx on each node

  
  

   

   
