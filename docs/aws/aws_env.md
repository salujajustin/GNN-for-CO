# Custom Python Environment on AWS 
--------------------------------------------------------------------------

This is a short walk though  on how to manage a custom python environment on a remote server. The remote server in this instance is AWS and the virtual environment manager is pipenv. 

## Amazon Web Services (AWS) Setup 

1. Launch Instance and choose an AMI: Deep Learning AMI (Ubuntu 18.04)

 2. Select and Launch: g4dn.xlarge

 3. ssh -i **/path-to/my-key-pair**.pem **Instance User Name**@**Public IPv4 DNS**
 ```bash
 ssh -i 18786_cloud.pem ubuntu@ec2-3-142-252-170.us-east-2.compute.amazonaws.com
 ```

## Create a Virtual Environment Locally or on the Cloud

This step can be used to create a *Pipfile* on your local machine that can be transferred over to the cloud or can be done on AWS directly. 

1. Install pipenv 
```bash
pip install pipenv 
```
> If you run into `pipenv: command not found` run the following:  
```bash
sudo -H pip install -U pipenv
```

2. Install Python 3.8 
> this is necessary when trying to run the *Attention, Learn to Solve Routing Problems!* [paper](https://arxiv.org/pdf/1803.08475.pdf) [code](https://github.com/wouterkool/attention-learn-to-route). 
```bash
sudo apt install python3.8
```
Check the where it was installed
```bash
which python3.8
```

3. Start a pipenv environment with Python 3.8: pipenv --python **path/to/python**
```bash
pipenv --python /usr/bin/python3.8
```

4. Enter environment
```bash
pipenv shell
```

5. Install dependencies via pipenv: [video tutorial](https://www.youtube.com/watch?v=6Qmnh5C4Pmo) [cheat sheet](https://gist.github.com/bradtraversy/c70a93d6536ed63786c434707b898d55)
```bash
pipenv install torch==1.8.0
```
**Note** if a *Pipfile* is already present just run `pipenv install` and it will install all of the dependencies specified in the file.
	
## Preparing Local Code for the Server 

1.  Zip up relevant files: zip -r **compressed.zip** **/path/to/dir**
```bash
zip -r my-project.zip my-project
``` 
2. Make sure the aws instance is running 

3. Copy them to  the running instance
 ```bash
scp -i {{/path/my-key-pair}}.pem {{/path/my-file}}  {{Instance User Name}}@{{Public IPv4 DNS}}:{{path/}}
```




## Switching Cuda versions
[Using the Deep Learning Base AMI](https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-base.html)

## Miscellaneous 
**TODO Format Later for below**

https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-attaching-volume.html
https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-using-volumes.html
https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/add-instance-store-volumes.html

For the attention paper: 
 env shows CUDA11.0, which is not shipped in the 1.8.0 binaries (10.2 and 11.1 are used)
 so pytorch version 1.7 will be installed - it works fine 
 
Transfering a file from EC2 --> local machine [tutorial](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html)
scp -i /path/my-key-pair.pem ec2-user@my-instance-public-dns-name:path/my-file.txt path/my-file2.txt
 
 ## Jupyter Notebook on AWS via Port Forwarding 
 
jupyter notebook can run over remote server
Run on server:
pipenv install environment_kernels
jupyter notebook --no-browser --port=8080

On local machine:
Replace <PORT> with the port number you selected in the above step
Replace <REMOTE_USER> with the remote server username
Replace <REMOTE_HOST> with your remote server address
ssh -L 8080:localhost:<PORT> <REMOTE_USER>@<REMOTE_HOST>

In the browser go to:
http://localhost:8080/

On server:
search for **token=** for runnning server and paste it if the browser asks for token. 

