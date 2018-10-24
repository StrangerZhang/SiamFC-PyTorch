# Pytorch implementation of SiamFC

## Run demo
```bash
cd SiamFC-Pytorch

mkdir models

# for color model
wget http://www.robots.ox.ac.uk/%7Eluca/stuff/siam-fc_nets/2016-08-17.net.mat -P models/
# for color+gray model
wget http://www.robots.ox.ac.uk/%7Eluca/stuff/siam-fc_nets/2016-08-17_gray025.net.mat -P models/

python bin/convert_pretrained_model.py

# video dir should conatin groundtruth_rect.txt which the same format like otb
python bin/demo_siamfc --gpu-id [gpu_id] --video-dir path/to/video
```

## Training
Download ILSVRC2015-VID 

```bash
cd SiamFC-Pytorch

mkdir models

# using 12 threads should take an hour
python bin/create_dataset.py --data-dir path/to/data/ILSVRC2015 \
			     --output-dir path/to/data/ILSVRC_VID_CURATION \
			     --num-threads 8

# ILSVRC2015_VID_CURATION and ILSVRC2015_VID_CURATION.lmdb should be in the same directory
# the ILSVRC2015_VID_CURATION.lmdb should be about 34G or so
python bin/create_lmdb.py --data-dir path/to/data/ILSVRC_VID_CURATION \
			  --output-dir path/to/data/ILSVRC2015_VID_CURATION.lmdb \
		          --num-threads 8

# training should take about 1.5~2hrs on a Titan Xp GPU with 30 epochs
python bin/train_siamfc.py --gpu-id [gpu_id] --data-dir path/to/data/ILSVRC2015_VID_CURATION
```
## Benchmark results
#### OTB100

| Tracker 			    		 | AUC             |
| ---------------------------------------------  | --------------- |
| SiamFC-color(converted from matconvnet)        | 0.5544          |
| SiamFC-color+gray(converted from matconvnet)   | 0.5818(vs 0.582)|
| SiamFC(trained from scratch)      		 | 0.5820(vs 0.582)|

## Note
We use SGD without momentum, weight decay setting 0, detailed setting can be found in config.py
Training is unstable, In order to reproduce the result, you should evaluate all epoches between 
10 to 30 on OTB100, and choose the best one.
below is one of my experiment result.
```bash
Epoch 11 AUC: 0.5522
Epoch 12 AUC: 0.5670
Epoch 13 AUC: 0.5604
Epoch 14 AUC: 0.5559
Epoch 15 AUC: 0.5790
Epoch 16 AUC: 0.5687
Epoch 17 AUC: 0.5534
Epoch 18 AUC: 0.5745
Epoch 19 AUC: 0.5619
Epoch 20 AUC: 0.5749
Epoch 21 AUC: 0.5648
Epoch 22 AUC: 0.5775
Epoch 23 AUC: 0.5784
Epoch 24 AUC: 0.5812
Epoch 25 AUC: 0.5785
Epoch 26 AUC: 0.5637
Epoch 27 AUC: 0.5764
Epoch 28 AUC: 0.5675
Epoch 29 AUC: 0.5787
Epoch 30 AUC: 0.5820
```
## Reference
[1] Bertinetto, Luca and Valmadre, Jack and Henriques, Joo F and Vedaldi, Andrea and Torr, Philip H S
		Fully-Convolutional Siamese Networks for Object Tracking
		In ECCV 2016 workshops
