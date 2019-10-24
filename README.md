# Segmentation Training Pipeline

![Build status](https://travis-ci.com/musket-ml/segmentation_training_pipeline.svg?branch=master)

  * [Motivation](#motivation)
  * [Installation](#installation)
  * [Usage guide](#usage-guide)
    + [Training a model](#training-a-model)
      - [Image/Mask Augmentations](#image-and-mask-augmentations)
      - [Freezing/Unfreezing encoder](#freezing-and-unfreezing-encoder)
      - [Custom datasets](#custom-datasets)      
      - [Balancing your data](#balancing-your-data)
      - [Multistage training](#multistage-training)
      - [Composite losses](#composite-losses)
      - [Cyclical learning rates](#cyclical-learning-rates)
      - [LR Finder](#lr-finder)      
      - [Background Augmenter](#background-augmenter)
      - [Training on crops](#training-on-crops)
    + [Using trained model](#using-trained-model)
      - [Ensembling predictions and test time augmentation](#ensembling-predictions)
    + [Custom evaluation code](#custom-evaluation-code)
    + [Accessing model](#accessing-model)
  * [Analyzing Experiments Results](#analyzing-experiments-results)
  * [What is supported?](#what-is-supported-)    
  * [Custom architectures, callbacks, metrics](#custom-architectures--callbacks--metrics)
  * [Examples](#examples)
  * [Faq](#faq)


## Motivation

Idea for this project came from my first attempts to participate in Kaggle competitions. My programmers heart was painfully damaged by looking on my own code as well as on other people kernels. Code was highly repetitive, suffering from numerous reimplementations of same or almost same things through the kernels, model/experiment configuration was often mixed with models code, in other words - from programmer perspective it all looked horrible. 

So I decided to extract repetitive things into framework that will work at least for me and will follow these statements: 
 - experiment configurations should be cleanly separated from model definitions;
 - experiment configuration files should be easy to compare and should fully describe experiment that is being performed except for the dataset;
- common blocks like an architecture, callbacks, storing model metrics, visualizing network predictions, should be written once and be a part of common library


## Installation

```
pip install segmentation_pipeline
```
*Note: this package requires python 3.6*

## Usage guide

### Training a model

Let's start from the absolutely minimalistic example. Let's say that you have two folders, one of them contains
jpeg images, and another one - png files with segmentation masks for these images. And you need to train a neural network
that will do segmentation for you. In this extremely simple setup all that you need is to type following 5
lines of python code:
```python
from segmentation_pipeline.impl.datasets import SimplePNGMaskDataSet
from segmentation_pipeline import  segmentation
ds=SimplePNGMaskDataSet("./pics/train","./pics/train_mask")
cfg = segmentation.parse("config.yaml")
cfg.fit(ds)
```

Looks simple, but there is a `config.yaml` file in the code, and probably it is the place where everything actually happens.

```yaml
backbone: mobilenetv2 #let's select classifier backbone for our network 
architecture: DeepLabV3 #let's select segmentation architecture that we would like to use
augmentation:
 Fliplr: 0.5 #let's define some minimal augmentations on images
 Flipud: 0.5 
classes: 1 #we have just one class (mask or no mask)
activation: sigmoid #one class means that our last layer should use sigmoid activation
encoder_weights: pascal_voc #we would like to start from network pretrained on pascal_voc dataset
shape: [320, 320, 3] #This is our desired input image and mask size, everything will be resized to fit.
optimizer: Adam #Adam optimizer is a good default choice
batch: 16 #Our batch size will be 16
metrics: #We would like to track some metrics
  - binary_accuracy 
  - iou
primary_metric: val_binary_accuracy #and the most interesting metric is val_binary_accuracy
callbacks: #Let's configure some minimal callbacks
  EarlyStopping:
    patience: 15
    monitor: val_binary_accuracy
    verbose: 1
  ReduceLROnPlateau:
    patience: 4
    factor: 0.5
    monitor: val_binary_accuracy
    mode: auto
    cooldown: 5
    verbose: 1
loss: binary_crossentropy #We use simple binary_crossentropy loss
stages:
  - epochs: 100 #Let's go for 100 epochs
```

So as you see, we have decomposed our task in two parts, *code that actually trains the model* and *experiment configuration*,
which determines the model and how it should be trained from the set of predefined building blocks.
 
What does this code actually do behind the scenes?

-  it splits your data into 5 folds, and trains one model per fold;
-  it takes care of model checkpointing, generates example image/mask/segmentation triples, collects training metrics. All this data will
   be stored in the folders just near your `config.yaml`;
-  All your folds are initialized from fixed default seed, so different experiments will use exactly the same train/validation splits

Also, datasets can be specified directly in your config file in more generic way, see examples ds_1, ds_2, ds_3 in "segmentation_training_pipeline/examples/people" folder. In this case you can just call cfg.fit() without providing dataset programmatically.

#### Image and Mask Augmentations

Framework uses awesome [imgaug](https://github.com/aleju/imgaug) library for augmentation, so you only need to configure your augmentation process in declarative way like in the following example:
 
```yaml
augmentation:  
  Fliplr: 0.5
  Flipud: 0.5
  Affine:
    scale: [0.8, 1.5] #random scalings
    translate_percent:
      x: [-0.2,0.2] #random shifts
      y: [-0.2,0.2]
    rotate: [-16, 16] #random rotations on -16,16 degrees
    shear: [-16, 16] #random shears on -16,16 degrees
```

#### Freezing and Unfreezing encoder

Freezing encoder is often used with transfer learning. If you want to start with frozen encoder just add

```yaml
freeze_encoder: true
stages:
  - epochs: 10 #Let's go for 10 epochs with frozen encoder
  
  - epochs: 100 #Now let's go for 100 epochs with trainable encoder
    unfreeze_encoder: true  
```

in your experiments configuration, then on some stage configuration just add

```yaml
unfreeze_encoder: true
```
to stage settings.


*Note: This option is not supported for DeeplabV3 architecture.*

#### Custom datasets

Training data and masks are not necessarily stored in files, so sometimes you need to declare your own dataset class,
for example, the following code was used in my experiments with [Airbus ship detection challenge](https://www.kaggle.com/c/airbus-ship-detection/overview)
to decode segmentation masks from rle encoded strings stored in csv file 

```python
from segmentation_pipeline.impl.datasets import PredictionItem
import os
from segmentation_pipeline.impl import rle
import imageio
import pandas as pd

class SegmentationRLE:

    def __init__(self,path,imgPath):
        self.data=pd.read_csv(path);
        self.values=self.data.values;
        self.imgPath=imgPath;
        self.ship_groups=self.data.groupby('ImageId');
        self.masks=self.ship_groups['ImageId'];
        self.ids=list(self.ship_groups.groups.keys())
        pass
    
    def __len__(self):
        return len(self.masks)


    def __getitem__(self, item):
        pixels=self.ship_groups.get_group(self.ids[item])["EncodedPixels"]
        return PredictionItem(self.ids[item] + str(), imageio.imread(os.path.join(self.imgPath,self.ids[item])),
                              rle.masks_as_image(pixels) > 0.5)


    
```         

#### Balancing your data

One common case is the situation when part of your images does not contain any objects of interest, like in 
[Airbus ship detection challenge](https://www.kaggle.com/c/airbus-ship-detection/overview). More over your data may
be to heavily inbalanced, so you may want to rebalance it. Alternatively you may want to inject some additional
images that do not contain objects of interest to decrease amount of false positives that will be produced by the framework.
    
These scenarios are supported by `negatives` and `validation_negatives` settings of training stage configuration,
these settings accept following values:

- none - exclude negative examples from the data
- real - include all negative examples 
- integer number(1 or 2 or anything), how many negative examples should be included per one positive example   

if you are using this setting your dataset class must support `isPositive` method which returns true for indexes
which contain positive examples: 

```python        
    def isPositive(self, item):
        pixels=self.ddd.get_group(self.ids[item])["EncodedPixels"]
        for mask in pixels:
            if isinstance(mask, str):
                return True;
        return False
```        

#### Multistage training

Sometimes you need to split your training into several stages. You can easily do it by adding several stage entries
in your experiment configuration file like in the following example:

```yaml
stages:
  - epochs: 6 #Train for 6 epochs
    negatives: none #do not include negative examples in your training set 
    validation_negatives: real #validation should contain all negative examples    

  - lr: 0.0001 #let's use different starting learning rate
    epochs: 6
    negatives: real
    validation_negatives: real

  - loss: lovasz_loss #let's override loss function
    lr: 0.00001
    epochs: 6
    initial_weights: ./fpn-resnext2/weights/best-0.1.weights #let's load weights from this file    
```

Stage entries allow you to configure custom learning rate, balance of negative examples, callbacks, loss function
and even initial weights which should be used on a particular stage.

#### Composite losses

Framework supports composing loss as a weighted sum of predefined loss functions. For example, following construction
```yaml
loss: binary_crossentropy+0.1*dice_loss
```
will result in loss function which is composed from `binary_crossentropy` and  `dice_loss` functions.

#### Cyclical learning rates

![Example](https://github.com/bckenstler/CLR/blob/master/images/triangularDiag.png?raw=true)

As told in [Cyclical learning rates for training neural networks](https://arxiv.org/abs/1506.01186) CLR policies can provide quicker converge for some neural network tasks and architectures. 

![Example2](https://github.com/bckenstler/CLR/raw/master/images/cifar.png)

We support them by adopting Brad Kenstler [CLR callback](https://github.com/bckenstler/CLR) for Keras.

If you want to use them, just add `CyclicLR` in your experiment configuration file as shown below: 

```yaml
callbacks:
  EarlyStopping:
    patience: 40
    monitor: val_binary_accuracy
    verbose: 1
  CyclicLR:
     base_lr: 0.0001
     max_lr: 0.01
     mode: triangular2
     step_size: 300
```

#### LR Finder

[Estimating optimal learning rate for your model](https://arxiv.org/abs/1506.01186) is an important thing, we support this by using slightly changed 
version of [Pavel Surmenok - Keras LR Finder](https://github.com/surmenok/keras_lr_finder)

```python
cfg= segmentation.parse(people-1.yaml)
ds=SimplePNGMaskDataSet("./train","./train_mask")
finder=cfg.lr_find(ds,start_lr=0.00001,end_lr=1,epochs=5)
finder.plot_loss(n_skip_beginning=20, n_skip_end=5)
plt.show()
finder.plot_loss_change(sma=20, n_skip_beginning=20, n_skip_end=5, y_lim=(-0.01, 0.01))
plt.show()
```
will result in this couple of helpful images: 

![image](https://camo.githubusercontent.com/b41aeaff00fb7b214b5eb2e5c151e7e353a7263e/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313630302f312a48566a5f344c57656d6a764f57762d63514f397939672e706e67)

![image](https://camo.githubusercontent.com/834996d32bbd2edf7435c5e105b53a6b447ef083/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313630302f312a38376d4b715f586f6d59794a4532396c39314b3064772e706e67)
#### Background Augmenter

One interesting augentation option when doing background removal task is replacing backgrounds with random 
images. We support this with `BackgroundReplacer` augmenter:

```yaml
augmentation:
  BackgroundReplacer:
    path: ./bg #path to folder with backgrounds
    rate: 0.5 #fraction of original backgrounds to preserve

```

#### Training on crops

Your images can be too large to train model on them. In this case you probably want to train model on crops. All
that you need to do is to specify number of splits per axis. For example, following lines in config 

```yaml
shape: [768, 768, 3]
crops: 3
``` 
will lead to splitting each image/mask into 9 cells (3 horizontal splits and 3 vertical splits) and training model on these splits.
Augmentations will be run separately on each cell.


During prediction time, your images will be split into these cells, prediction will be executed on each cell, and then results
will be assembled in single final mask. Thus the whole process of cropping will be invisible from a consumer perspective.   

### Using trained model

Okey, our model is trained, now we need to actually do image segmentation. Let's say, we need to run image segmentation on
images in the directory and store results in csv file:

```python
from segmentation_pipeline import  segmentation
from segmentation_pipeline.impl.rle import rle_encode
from skimage.morphology import remove_small_objects, remove_small_holes
import pandas as pd

#this is our callback which is called for every image
def onPredict(file_name, img, data):
    threshold = 0.25
    predictions = data["pred"]
    imgs = data["images"]
    post_img = remove_small_holes(remove_small_objects(img.arr > threshold))
    rle = rle_encode(post_img)
    predictions.append(rle)
    imgs.append(file_name[:file_name.index(".")])
    pass

cfg= segmentation.parse("config.yaml")

predictions = []
images = []
#Now let's use best model from fold 0 to do image segmentation on images from images_to_segment
cfg.predict_in_directory("./images_to_segment", 0, 0, onPredict, {"pred": predictions, "images": images})

#Let's store results in csv
df = pd.DataFrame.from_dict({'image': images, 'rle_mask': predictions})
df.to_csv('baseline_submission.csv', index=False)
``` 
#### Ensembling predictions

And what if you want to ensemble models from several folds? Just pass a list of fold numbers to
`predict_in_directory` like in the following example:

```python
cfg.predict_in_directory("./images_to_segment", [0,1,2,3,4], onPredict, {"pred": predictions, "images": images})
```
Another supported option is to ensemble results from extra test time augmentation (flips) by adding keyword arg `ttflips=True`.
  
### Custom evaluation code

Sometimes you need to run custom evaluation code. In such case you may use: `evaluateAll` method, which provides an iterator
on the batches containing original images, training masks and predicted masks

```python
for batch in cfg.evaluateAll(ds,2):
    for i in range(len(batch.predicted_maps_aug)):
        masks = ds.get_masks(batch.data[i])
        for d in range(1,20):
            cur_seg = binary_opening(batch.predicted_maps_aug[i].arr > d/20, np.expand_dims(disk(2), -1))
            cm = rle.masks_as_images(rle.multi_rle_encode(cur_seg))
            pr = f2(masks, cm);
            total[d]=total[d]+pr
```

### Accessing model
You may get trained keras model by calling: ```cfg.load_model(fold, stage)```.

## Analyzing experiments results

Okey, we have done a lot of experiments and now we need to compare the results and understand what works better. This repository
contains [script](segmentation_pipeline/analize.py) which may be used to analyze folder containing sub folders
with experiment configurations and results. This script gathers all configurations, diffs them by doing structural diff, then 
for each configuration it averages metrics for all folds and  generates csv file containing metrics and parameters that
was actually changed in your experiment like in the following [example](report.csv)

This script accepts following arguments:

 - inputFolder - root folder to search for experiments configurations and results
 - output - file to store aggregated metrics
 - onlyMetric - if you specify this option all other metrics will not be written in the report file
 - sortBy - metric that should be used to sort results 

Example: 
```commandline
python analize.py --inputFolder ./experiments --output ./result.py
``` 
 
## What is supported?

At this moment segmentation pipeline supports following architectures:

- [Unet](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- [Linknet](https://codeac29.github.io/projects/linknet/)
- [PSP](https://arxiv.org/abs/1612.01105)
- [FPN](https://arxiv.org/abs/1612.03144)
- [DeeplabV3](https://arxiv.org/abs/1706.05587)

`FPN`, `PSP`, `Linkenet`, `UNet` architectures support following backbones: 

  - [VGGNet](https://arxiv.org/abs/1409.1556)
    - vgg16
    - vgg19
  - [ResNet](https://arxiv.org/abs/1512.03385)
    - resnet18
    - resnet34
    - resnet50 
    - resnet101
    - resnet152
  - [ResNext](https://arxiv.org/abs/1611.05431)
    - resnext50
    - resnext101
  - [DenseNet](https://arxiv.org/abs/1608.06993)
    - densenet121
    - densenet169
    - densenet201
  - [Inception-v3](https://arxiv.org/abs/1512.00567)
  - [Inception-ResNet-v2](https://arxiv.org/abs/1602.07261)

All them support the weights pretrained on [ImageNet](http://www.image-net.org/):
```yaml
encoder_weights: imagenet
```

At this moment `DeeplabV3` architecture supports following backbones:
 - [MobileNetV2](https://arxiv.org/abs/1801.04381)
 - [Xception](https://arxiv.org/abs/1610.02357)

Deeplab supports weights pretrained on [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/):

```yaml
encoder_weights: pascal_voc
``` 

Each architecture also supports some specific options, list of options is documented in [segmentation RAML library](segmentation_pipeline/schemas/segmentation.raml#L166).

Supported augmentations are documented in [augmentation RAML library](segmentation_pipeline/schemas/augmenters.raml).

Callbacks are documented in [callbacks RAML library](segmentation_pipeline/schemas/callbacks.raml).  

## Custom architectures, callbacks, metrics

Segmentation pipeline uses keras custom objects registry to find entities, so if you need to use
custom loss function, activation or metric all that you need to do is to register it in Keras as: 

```python
keras.utils.get_custom_objects()["my_loss"]= my_loss
```

If you want to inject new architecture, you should register it in `segmentation.custom_models` dictionary.

For example:
```python
segmentation.custom.models['MyUnet']=MyUnet 
```
where `MyUnet` is a function that accepts architecture parameters as arguments and returns an instance
of keras model.

## Examples

[Training background removal task(Pics Art Hackaton) in google collab](https://colab.research.google.com/drive/1HtJLwoI_93m8pnRkK4u8JiFwv33L9Pil)

# FAQ

#### How to continue training after crash?

If you would like to continue training after crash, call `setAllowResume` method before calling `fit`

```python
cfg= segmentation.parse("./people-1.yaml")
cfg.setAllowResume(True)
ds=SimplePNGMaskDataSet("./pics/train","./pics/train_mask")
cfg.fit(ds)
```


#### My notebooks constantly run out of memory, what can I do to reduce memory usage?

One way to reduce memory usage is to limit augmentation queue limit which is 50 by default, 
like in the following example: 

```python
segmentation_pipeline.impl.datasets.AUGMENTER_QUEUE_LIMIT = 3
```

#### How can I run sepate set of augmenters on initial image/mask when replacing backgrounds with Background Augmenter?
```yaml
  BackgroundReplacer:
    rate: 0.5
    path: ./bg
    augmenters: #this augmenters will run on original image before replacing background
      Affine:
        scale: [0.8, 1.5]
        translate_percent:
              x: [-0.2,0.2]
              y: [-0.2,0.2]
        rotate: [-16, 16]
        shear: [-16, 16]
    erosion: [0,5]   
```


#### How can I visualize images that are used for training (after augmentations)?

You should set `showDataExamples` to True like in the following sample
```python
cfg= segmentation.parse("./no_erosion_aug_on_masks/people-1.yaml")
cfg.showDataExamples=True
```
if will lead to generation of training images samples and storing them in examples folder at the end of each epoch

#### What I can do if i have some extra training data, that should not be included into validation, but should be used during the training?

```python
extra_data=NotzeroSimplePNGMaskDataSet("./phaces/all","./phaces/masks") #My dataset that should be added to training
segmentation.extra_train["people"] = extra_data
```   

and in the config file:

```yaml
extra_train_data: people
```

#### How to get basic statistics across my folds/stages


This code sample will return primary metric stats over folds/stages
```
cfg= segmentation.parse("./no_erosion_aug_on_masks/people-1.yaml")
metrics = cfg.info()
```


#### I have some callbacks that are configured globally, but I need some extra callbacks for my last training stage?

There are two possible ways how you may configure callbacks on stage level:

- override all global callbacks with `callbacks` setting.
- add your own custom callbacks with `extra_callbacks` setting.

In the following sample CyclingRL callback is only appended to the sexond stage of training:

```yaml
loss: binary_crossentropy
stages:
  - epochs: 20
    negatives: real
  - epochs: 200
    extra_callbacks:
      CyclicLR:
        base_lr: 0.000001
        max_lr: 0.0001
        mode: triangular
        step_size: 800
    negatives: real
```

#### What if I would like to build a really large ansemble of models?

One option to do this, is to store predictions for each file and model in numpy array, and then sum these predictions
like in the following sample:

```python
cfg.predict_to_directory("./pics/test","./pics/arr1", [0, 1, 4, 2], 1, ttflips=True,binaryArray=True)
cfg.predict_to_directory("./pics/test", "./pics/arr", [0, 1, 4, 2], 2, ttflips=True, binaryArray=True)
segmentation.ansemblePredictions("./pics/test",["./pics/arr/","./pics/arr1/"],onPredict,d)
``` 

#### How to train on multiple gpus?

```python
cfg.gpus=4 #or another number matching to the count of gpus that you have
``` 
