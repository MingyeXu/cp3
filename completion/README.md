


### MVP Completion Dataset
<!-- Download the MVP completion dataset by the following commands:
```
cd data; sh download_data.sh
``` -->
Download the MVP completion dataset(https://www.dropbox.com/sh/la0kwlqx4n2s5e3/AACjoTzt-_vlX6OF9mfSpFMra?dl=0&lst=)


### Requirements
+ pip -r requirements.txt
+ source setup.sh (refer to https://github.com/paul007pl/VRCNet )

### Usage

## Generation stage:
The pre-trained model path in generation stage: ./log/vrcnet_cd_debug_2022-02-22T17:07:04/pretrain.pth
You can also retrain the generation stage yourself. During the training process, you need to modify the code to comment out the refinement network.

## Refinement stage:
+ To train a model: run `python train.py -c ./cfgs/*.yaml`, e.g. `python train.py -c ./cfgs/cp3.yaml`
+ To test a model: run `python test.py -c ./cfgs/*.yaml`, e.g. `python test.py -c ./cfgs/cp3.yaml`






