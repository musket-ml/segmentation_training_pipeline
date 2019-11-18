# Segmentation Training Pipeline
![Build status](https://travis-ci.com/musket-ml/segmentation_training_pipeline.svg?branch=master)

This package is a part of [Musket ML](https://musket-ml.com/) framework.

## Reasons to use Segmentation Pipeline
Segmentation Pipeline was developed with a focus of enabling to make fast and 
simply-declared experiments, which can be easily stored, 
reproduced and compared to each other.

Segmentation Pipeline has a lot of common parts with [Generic pipeline](https://musket-ml.github.io/webdocs/generic/), but it is easier to define an architecture of the network.
Also there are a number of segmentation-specific features.

The pipeline provides the following features:

* Allows to describe experiments in a compact and expressive way
* Provides a way to store and compare experiments in order to methodically find the best deap learning solution
* Easy to share experiments and their results to work in a team
* Experiment configurations are separated from model definitions
* It is easy to configure network architecture
* Provides great flexibility and extensibility via support of custom substances
* Common blocks like an architecture, callbacks, model metrics, predictions vizualizers and others should be written once and be a part of a common library

## Installation

```
pip install segmentation_pipeline
```
*Note: this package requires python 3.6*

This package is a part of [Musket ML](https://musket-ml.com/) framework,
 it is recommended to install the whole collection of the framework
 packages at once using instructions [here](https://musket-ml.github.io/webdocs/generic/#installation).

## Launching

### Launching experiments

`fit.py` script is designed to launch experiment training.

In order to run the experiment or a number of experiments,   

A typical command line may look like this:

`musket fit --project "path/to/project" --name "experiment_name" --num_gpus=1 --gpus_per_net=1 --num_workers=1 --cache "path/to/cache/folder"`

[--project](https://musket-ml.github.io/webdocs/segmentation/reference/#fitpy-project) points to the root of the [project](#project-structure)

[--name](https://musket-ml.github.io/webdocs/segmentation/reference/#fitpy-name) is the name of the project sub-folder containing experiment yaml file.

[--num_gpus](https://musket-ml.github.io/webdocs/segmentation/reference/#fitpy-num_gpus) sets number of GPUs to use during experiment launch.

[--gpus_per_net](https://musket-ml.github.io/webdocs/segmentation/reference/#fitpy-gpus_per_net) is a maximum number of GPUs to use per single experiment.

[--num_workers](https://musket-ml.github.io/webdocs/segmentation/reference/#fitpy-num_workers) sets number of workers to use.

[--cache](https://musket-ml.github.io/webdocs/segmentation/reference/#fitpy-cache) points to a cache folder to store the temporary data.

Other parameters can be found in the [fit script reference](https://musket-ml.github.io/webdocs/segmentation/reference/#fit-script-arguments)

### Launching tasks

`task.py` script is designed to launch experiment training.
 
Tasks must be defined in the project python scope and marked by an 
annotation like this:

```python
from musket_core import tasks, model
@tasks.task
def measure2(m: model.ConnectedModel):
    return result
```

Working directory *must* point to the `musket_core` root folder.

In order to run the experiment or a number of experiments,   

A typical command line may look like this:

`python -m musket_core.task --project "path/to/project" --name "experiment_name" --task "task_name" --num_gpus=1 --gpus_per_net=1 --num_workers=1 --cache "path/to/cache/folder"`

[--project](https://musket-ml.github.io/webdocs/segmentation/reference/#taskpy-project) points to the root of the [project](#project-structure)

[--name](https://musket-ml.github.io/webdocs/segmentation/reference/#taskpy-name) is the name of the project sub-folder containing experiment yaml file.

[--task](https://musket-ml.github.io/webdocs/segmentation/reference/#taskpy-name) is the name of the task function.

[--num_gpus](https://musket-ml.github.io/webdocs/segmentation/reference/#taskpy-num_gpus) sets number of GPUs to use during experiment launch.

[--gpus_per_net](https://musket-ml.github.io/webdocs/segmentation/reference/#taskpy-gpus_per_net) is a maximum number of GPUs to use per single experiment.

[--num_workers](https://musket-ml.github.io/webdocs/segmentation/reference/#taskpy-num_workers) sets number of workers to use.

[--cache](https://musket-ml.github.io/webdocs/segmentation/reference/#taskpy-cache) points to a cache folder to store the temporary data.

Other parameters can be found in the [task script reference](https://musket-ml.github.io/webdocs/segmentation/reference/#task-script-arguments)

### Launching project analysis

`analize.py` script is designed to launch project-scope analysis.

Note that only experiments, which training is already finished will be covered.

`musket analize --inputFolder "path/to/project"`

[--inputFolder](https://musket-ml.github.io/webdocs/segmentation/reference/#analyzepy-inputfolder) points to a folder to search for finished experiments in. Typically, project root.

Other parameters can be found in the [analyze script reference](https://musket-ml.github.io/webdocs/segmentation/reference/#analyze-script-arguments)


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
testSplit: 0.4
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

Moreover, the whole fitting and prediction process can be launched with built-in script, 
the only really required python code is dataset definition to let the system know, which data to load.

What does this code actually do behind the scenes?

-  it splits your data into 5 folds, and trains one model per fold;
-  it takes care of model checkpointing, generates example image/mask/segmentation triples, collects training metrics. All this data will
   be stored in the folders just near your `config.yaml`;
-  All your folds are initialized from fixed default seed, so different experiments will use exactly the same train/validation splits

Also, datasets can be specified directly in your config file in more generic way, see examples ds_1, ds_2, ds_3 in "segmentation_training_pipeline/examples/people" folder. In this case you can just call cfg.fit() without providing dataset programmatically.

Lets discover what's going on in more details:

#### General train properties

Lets take our standard example and check the following set of instructions:

```yaml
testSplit: 0.4
optimizer: Adam #Adam optimizer is a good default choice
batch: 16 #Our batch size will be 16
metrics: #We would like to track some metrics
  - binary_accuracy 
  - iou
primary_metric: val_binary_accuracy #and the most interesting metric is val_binary_accuracy
loss: binary_crossentropy #We use simple binary_crossentropy loss
```

[testSplit](https://musket-ml.github.io/webdocs/segmentation/reference/#testsplit) Splits the train set into two parts, using one part for train and leaving the other untouched for a later testing.
The split is shuffled. 

[optimizer](https://musket-ml.github.io/webdocs/segmentation/reference/#optimizer) sets the optimizer.

[batch](https://musket-ml.github.io/webdocs/segmentation/reference/#batch) sets the training batch size.

[metrics](https://musket-ml.github.io/webdocs/segmentation/reference/#metrics) sets the metrics to track during the training process. Metric calculation results will be printed in the console and to `metrics` folder of the experiment.

[primary_metric](https://musket-ml.github.io/webdocs/segmentation/reference/#primary_metric) Metric to track during the training process. Metric calculation results will be printed in the console and to `metrics` folder of the experiment.
Besides tracking, this metric will be also used by default for metric-related activity, in example, for decision regarding which epoch results are better.

[loss](https://musket-ml.github.io/webdocs/segmentation/reference/#loss) sets the loss function. if your network has multiple outputs, you also may pass a list of loss functions (one per output) 

Framework supports composing loss as a weighted sum of predefined loss functions. For example, following construction
```yaml
loss: binary_crossentropy+0.1*dice_loss
```
will result in loss function which is composed from `binary_crossentropy` and `dice_loss` functions.

There are many more properties to check in [Reference of root properties](https://musket-ml.github.io/webdocs/segmentation/reference/#pipeline-root-properties)

#### Defining architecture

Lets take a look at the following part of our example:

```yaml
backbone: mobilenetv2 #let's select classifier backbone for our network 
architecture: DeepLabV3 #let's select segmentation architecture that we would like to use
classes: 1 #we have just one class (mask or no mask)
activation: sigmoid #one class means that our last layer should use sigmoid activation
encoder_weights: pascal_voc #we would like to start from network pretrained on pascal_voc dataset
shape: [320, 320, 3] #This is our desired input image and mask size, everything will be resized to fit.
```

The following three properties are required to set:

[backbone](https://musket-ml.github.io/webdocs/segmentation/reference/#backbone) This property configures encoder that should be used. Different kinds of `FPN`, `PSP`, `Linkenet`, `UNet` and more are supported.

[architecture](https://musket-ml.github.io/webdocs/segmentation/reference/#architecture) This property configures decoder architecture that should be used. `net`, `Linknet`, `PSP`, `FPN` and more are supported.

[classes](https://musket-ml.github.io/webdocs/segmentation/reference/#classes) sets the number of classes that should be used. 

The following ones are optional, but commonly used:

[activation](https://musket-ml.github.io/webdocs/segmentation/reference/#activation) sets activation function that should be used in last layer.

[shape](https://musket-ml.github.io/webdocs/segmentation/reference/#shape) set the desired shape of the input picture and mask, in the form heigth, width, number of channels. Input will be resized to fit.

[encoder_weights](https://musket-ml.github.io/webdocs/segmentation/reference/#encoder_weights) configures initial weights of the encoder.

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
[augmentation](https://musket-ml.github.io/webdocs/segmentation/reference/#augmentation) property defines [IMGAUG](https://imgaug.readthedocs.io) transformations sequence.
Each object is mapped on [IMGAUG](https://imgaug.readthedocs.io) transformer by name, parameters are mapped too.

In this example, `Fliplr` and `Flipud` keys are automatically mapped on [Flip agugmenters](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_flip.html),
their `0.5` parameter is mapped on the first `p` parameter of the augmenter.
Named parameters are also mapped, in example `scale` key of `Affine` is mapped on `scale` parameter of [Affine augmenter](https://imgaug.readthedocs.io/en/latest/source/augmenters.html?highlight=affine#affine).

One interesting augementation option when doing background removal task is replacing backgrounds with random 
images. We support this with `BackgroundReplacer` augmenter:

```yaml
augmentation:
  BackgroundReplacer:
    path: ./bg #path to folder with backgrounds
    rate: 0.5 #fraction of original backgrounds to preserve

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

Both [freeze_encoder](https://musket-ml.github.io/webdocs/segmentation/reference/#freeze_encoder) and [unfreeze_encoder](https://musket-ml.github.io/webdocs/segmentation/reference/#unfreeze_encoder)
can be put into the root section and inside the stage.

*Note: This option is not supported for DeeplabV3 architecture.*

#### Custom datasets

Training data and masks are not necessarily stored in files, so sometimes you need to declare your own dataset class,
for example, the following code was used to support [Airbus ship detection challenge](https://www.kaggle.com/c/airbus-ship-detection/overview)
to decode segmentation masks from rle encoded strings stored in csv file 

```python
from segmentation_pipeline.impl.datasets import PredictionItem
import os
from segmentation_pipeline.impl import rle
import imageio
import pandas as pd

class SegmentationRLE(datasets.DataSet):

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

def getTrain()->datasets.DataSet:
    return SegmentationRLE("train.csv","images/")
```         

Now, if this python code sits somewhere in python files located in `modules` folder of the project, and that file is referred by [imports](https://musket-ml.github.io/webdocs/segmentation/reference/#imports) instruction, following YAML can refer it:
```yaml
dataset:
  getTrain: []
```

[dataset](https://musket-ml.github.io/webdocs/segmentation/reference/#dataset) sets the main training dataset.

[datasets](https://musket-ml.github.io/webdocs/segmentation/reference/#datasets) sets up a list of available data sets to be referred by other entities.

#### Multistage training

Sometimes you need to split your training into several stages. You can easily do it by adding several stage entries
in your experiment configuration file.

[stages](https://musket-ml.github.io/webdocs/segmentation/reference/#stages) instruction allows to set up stages of the train process, where for each stage it is possible to set some specific training options like the number of epochs, learning rate, loss, callbacks, etc.
Full list of stage properties can be found [here](https://musket-ml.github.io/webdocs/segmentation/reference/#stage-properties).

```yaml
stages:
  - epochs: 100 #Let's go for 100 epochs
  - epochs: 100 #Let's go for 100 epochs
  - epochs: 100 #Let's go for 100 epochs
```

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

#### Balancing your data

One common case is the situation when part of your images does not contain any objects of interest, like in 
[Airbus ship detection challenge](https://www.kaggle.com/c/airbus-ship-detection/overview). More over your data may
be to heavily inbalanced, so you may want to rebalance it. Alternatively you may want to inject some additional
images that do not contain objects of interest to decrease amount of false positives that will be produced by the framework.
    
These scenarios are supported by [negatives](https://musket-ml.github.io/webdocs/segmentation/reference/#negatives) and 
[validation_negatives](https://musket-ml.github.io/webdocs/segmentation/reference/#validation_negatives) settings of training stage configuration,
these settings accept following values:

- none - exclude negative examples from the data
- real - include all negative examples 
- integer number(1 or 2 or anything), how many negative examples should be included per one positive example   

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
#### Advanced learning rates
##### Dynamic learning rates

![Example](https://github.com/bckenstler/CLR/blob/master/images/triangularDiag.png?raw=true)

As told in [Cyclical learning rates for training neural networks](https://arxiv.org/abs/1506.01186) CLR policies can provide quicker converge for some neural network tasks and architectures. 

![Example2](https://github.com/bckenstler/CLR/raw/master/images/cifar.png)

We support them by adopting Brad Kenstler [CLR callback](https://github.com/bckenstler/CLR) for Keras.

If you want to use them, just add [CyclicLR](https://musket-ml.github.io/webdocs/segmentation/reference/#cycliclr) in your experiment configuration file as shown below: 

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

There are also [ReduceLROnPlateau](https://musket-ml.github.io/webdocs/segmentation/reference/#reducelronplateau) and [LRVariator](https://musket-ml.github.io/webdocs/segmentation/reference/#lrvariator) options to modify learning rate on the fly.

##### LR Finder

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

#### Training on crops

Your images can be too large to train model on them. In this case you probably want to train model on crops. All
that you need to do is to specify number of splits per axis. For example, following lines in config 

```yaml
shape: [768, 768, 3]
crops: 3
``` 
will lead to splitting each image/mask into 9 cells (3 horizontal splits and 3 vertical splits) and training model on these splits.
Augmentations will be run separately on each cell. 
[crops](https://musket-ml.github.io/webdocs/segmentation/reference/#crops) property sets the number of single dimension cells.


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
