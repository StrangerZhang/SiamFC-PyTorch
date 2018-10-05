# Pytorch implementation of SiamFC

## Run demo
cd SiamFC-Pytorch/models

wget http://www.robots.ox.ac.uk/%7Eluca/stuff/siam-fc_nets/2016-08-17.net.mat

cd SiamFC-Pytorch

python bin/convert_pretrained_model.py

python bin/demo_siamfc --gpu_id [gpu_id] --video_dir path/to/video

## Training
Download imagenet vid data

python bin/create_dataset.py --data_dir path/to/ILSVRC2015 --output_dir path/to/output_dir

python bin/train_siamfc.py --gpu_id [gpu_id] --data_dir path/to/output_dir

## Benchmark results
#### OTB100

| Tracker 			    | AUC            |
| --------------------------------- | -------------- |
| SiamFC(converted from matconvnet) | 55.54(vs 0.582)|
| SiamFC(trained from scratch)      | 56.16(vs 0.582)|

work in progress



## Reference
[1] Bertinetto, Luca and Valmadre, Jack and Henriques, Joo F and Vedaldi, Andrea and Torr, Philip H S
		Fully-Convolutional Siamese Networks for Object Tracking
		In ECCV 2016 workshops
