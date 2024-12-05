# ABBG Attack
[NeurIPS 2024 in AdvML Workshop] Adversarial Bounding Boxes Generation (ABBG) Attack against Visual Object Trackers


[OpenReview] coming soon!

[ArXiv](https://arxiv.org/abs/2411.17468)


## Object Bounding Box Evaluation (GOT10k dataset)
## Step 1: Download the trackers packages
Please download the trackers from the VOT challenge (VOT2022) website and GitHub, as follows:

1- ROMTrack: https://github.com/dawnyc/ROMTrack/archive/refs/heads/master.zip 

2- TransTM: https://data.votchallenge.net/vot2022/trackers/TransT_M-code-2022-05-03T05_05_59.924051.zip 

## Step 2: Create the environment
For each tracker follow the instructions to build the suitable environment as stated in their README.md file. 

## Step 3: Download the networks 
For ABBG experiments, we used the following networks:

+ ROMTrack: 
    - Tracker network(ROMTrack-Base/ROMTrack/GOT-10k/ROMTrack_epoch0050.pth.tar) from https://drive.google.com/drive/folders/1Q7CpNIhWX05VU7gECnhePu3dKzTV_VoK?usp=drive_link

       Download the network and place the file into the following directory:
        "ROMTrack/lib/models"

+ TransTM:
    - Tracker network is on the directory of (TransT_M/models/TransTiouhsegm_ep0090.pth.tar)


## Step 4: Run the setup files 
Follow the instructions of each tracker to correct the paths and run the setup files. 

## Step 5.a: Set the paths for ROMTrack
1- Copy the ABBG folder from the ABBG/ROMTrack directory and paste it into the tracker folder at ROMTrack/lib/test/tracker.

2- Copy the tracker_ABBG.py from the ABBG/ROMTrack/ directory and paste it into the evaluation folder at ROMTrack/lib/test/evaluation. 

3- Copy the test_ABBG.py from the ABBG/ROMTrack/ directory and paste it into the tracking folder located at ROMTrack/tracking

## Step 5.b: Set the paths for TransTM
1- Copy the ABBG folder from the ABBG/TransTM directory and paste it into the trackers folder located at TransT_M/pysot_toolkit/trackers.

2- Copy the test_ABBG.py from the ABBG/TransTM directory and paste it into the pysot_toolkit folder located at TransT_M/pysot_toolkit.

3- Set the paths for the dataset (dataset_root) in line 46 and model (net_path) in line 53 of the test_ABBG.py file.

## Step 6.a: Run the ROMTrack tracker attacked by ABBG 

1- Download the GOT10k dataset (http://got-10k.aitestunion.com/downloads) and set the dataset paths in ROMTrack/lib/test/evaluation/local.py.

2- Activate the ROMTrack environment. 

3- From the ROMTrack directory, run:

```
python tracking/test_ABBG.py ROMTrack got_stage2 --dataset got10k_val --params__model ROMTrack_epoch0050.pth.tar
```

## Step 6.b: Run the TransTM tracker attacked by ABBG
1- Download the GOT10k dataset (http://got-10k.aitestunion.com/downloads) and set the dataset path in TransT_M/pysot_toolkit/test_ABBG.py.

2- Activate the TransTM environment. 

3- From the TransT_M directory, run:

```
python pysot_toolkit/test_ABBG.py --dataset GOT-10k
```

## Contact:


[Fatemeh Nokabadi](mailto:nourifatemeh1@gmail.com)
