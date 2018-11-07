# Segmentation Training Pipeline
Research Pipeline for image masking/segmentation in Keras

Idea for this project came from my first attempts to participate in Kaggle competitions. My programmers heart was painfully damaged by looking on my own code as well as on other people kernels. Code was highly repetitive, suffering from numerous reimplementation of same or almost same things through the kernels, model/experiment configuration was oftenly mixed with models code, in other words from programmer perspective it all looked horrible. 

So I decided to extract repetitive things into framework that will work at least for me, and that will follow to this statements: 
 - experiment configurations should be cleanly separated from model definitions.
 - experiment configuration files should be easy to compare, and should fully describe experiment that is being performed except of the dataset
- common blocks like an architecture, callbacks, storing model metrics, visualizing network predictions, should be written once and should be a part of common library

## Installation

```
pip install segmentation_pipeline
```

## Usage guide

### Training a model

Let's start from the absolutely minimalistic example. Let's say that you have two folders, one of them contains
jpeg images, and another one - png files containing segmentation masks for them. And you need to train a neural network
that will do segmentation for you. In this extremly simple setup all that you need to do is to do is to type following 5

lines of python code:
```python
from segmentation_pipeline.impl.datasets import SimplePNGMaskDataSet
from segmentation_pipeline import  segmentation
ds=SimplePNGMaskDataSet("D:/pics/train","D:/pics/train_mask")
cfg = segmentation.parse("config.yaml")
cfg.fit(ds)
```

Looks simple, but there is a `config.yaml` file in the code, and probably it is the place where everything actually happens.

```yaml
backbone: mobilenetv2 #lets select classifier backbone for our network 
architecture: DeepLabV3 #lets select segmentation architecture that we would like to use
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
primary_metric: val_binary_accuracy #and most interesting metric is val_binary_accuracy
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

So as you see, we have decomposed our task in two parts, *code that actually trains model* and *experiment configuration*,
which determines model and how it should be trained from the set of predefined building blocks.
 
What this code actually does behind of the scenes?

-  it splits you data in 5 folds, and trains one model per fold
-  it takes care about model checkpointing, generates example image/mask/segmentation triples, collects training metrics. All this data will
   be stored in the folders just near your `config.yaml`
-  All you folds are initialized from fixed default seed, so different experiments will use exactly same train/validation splits     

### Using trained model

Okey, our model is trained, now we need to actually do image segmenation, let's say that we need to run image segmentation on
images in the directory and store results in csv file:

```python
from segmentation_pipeline import  segmentation
from segmentation_pipeline.impl.rle import rle_encode
from skimage.morphology import remove_small_objects, remove_small_holes
import pandas as pd

#this is our callback that is called for every image
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

predictions = [],images = []
d = {"pred": predictions, "images": images}

#Now lets use best model from fold 0 to do image segmentation on images from images_to_segment
cfg.predict_in_directory("D:/images_to_segment", 0, onPredict, d)

#Let, store results in csv
df = pd.DataFrame.from_dict({'image': images, 'rle_mask': predictions})
df.to_csv('baseline_submission.csv', index=False)
``` 

 
## What is supported?

### Multistage training

### Composite losses

### Test Time augmentation

### Negative Examples

### Ansembling predictions from different folds

## Custom architectures, callbacks, metrics

## Analyzing Results

## Examples