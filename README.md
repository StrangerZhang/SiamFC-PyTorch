# Pytorch implementation of SiamFC

## Run demo
```bash
export PYTHONPATH=/path/to/your/SiamFC-Pytorch:$PYTHONPATH

cd SiamFC-Pytorch

mkdir models

wget http://www.robots.ox.ac.uk/%7Eluca/stuff/siam-fc_nets/2016-08-17.net.mat -P models/

python bin/convert_pretrained_model.py

# video dir should conatin groundtruth_rect.txt which the same format like otb
python bin/demo_siamfc --gpu-id [gpu_id] --video-dir path/to/video
```

## Training
Download imagenet vid data

```bash
export PYTHONPATH=/path/to/your/SiamFC-Pytorch:$PYTHONPATH

cd SiamFC-Pytorch

mkdir models

python bin/create_dataset.py --data-dir path/to/ILSVRC2015 --output-dir path/to/ILSVRC2015_VID_CURATION --num-threads 8

# ILSVRC2015_VID_CURATION and ILSVRC2015_VID_CURATION should be in the same directory
# the ILSVRC2015_VID_CURATION.lmdb should be about 34G or so
python bin/create_lmdb.py --data-dir path/to/your/previous/ILSVRC2015_VID_CURATION --output-dir path/to/your/ILSVRC2015_VID_CURATION.lmdb  --num-threads 8

python bin/train_siamfc.py --gpu-id [gpu_id] --data-dir path/to/your/ILSVRC2015_VID_CURATION 
```

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
