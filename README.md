# Pytorch implementation of SiamFC

## Run demo
export PYTHONPATH=/path/to/your/SiamFC-Pytorch:$PYTHONPATH

cd SiamFC-Pytorch

mkdir models

wget http://www.robots.ox.ac.uk/%7Eluca/stuff/siam-fc_nets/2016-08-17.net.mat -P models/

python bin/convert_pretrained_model.py

python bin/demo_siamfc --gpu_id [gpu_id] --video_dir path/to/video

## Training
Download imagenet vid data

export PYTHONPATH=/path/to/your/SiamFC-Pytorch:$PYTHONPATH

cd SiamFC-Pytorch

mkdir models

python bin/create_dataset.py --data_dir path/to/ILSVRC2015 --output_dir path/to/output_dir --num-threads 8

python bin/create_lmdb.py --data_dir path/to/your/previous/output_dir --output_dir path/to/your/lmdb_dir  --num-threads 8

python bin/train_siamfc.py --gpu_id [gpu_id] --data_dir path/to/your/lmdb_dir

## Benchmark results
#### OTB100

| Tracker 			    | AUC            |
| --------------------------------- | -------------- |
| SiamFC(converted from matconvnet) | 55.54(vs 0.582)|
| SiamFC(trained from scratch)      | 57.85(vs 0.582)|


## Reference
[1] Bertinetto, Luca and Valmadre, Jack and Henriques, Joo F and Vedaldi, Andrea and Torr, Philip H S
		Fully-Convolutional Siamese Networks for Object Tracking
		In ECCV 2016 workshops
