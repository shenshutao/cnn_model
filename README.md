# step 1: clone git
git clone https://github.com/shenshutao/deeplearning_sample.git

# step 2: run model
## If have PBS
qsub submit.pbs
## Otherwise
python custom_resNet.py


