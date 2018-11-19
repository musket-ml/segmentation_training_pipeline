#FAQ

####How to continue training after crash?

If you would like to continue training after crash, call `setAllowResume` method before calling `fit`

```python
cfg= segmentation.parse("./people-1.yaml")
cfg.setAllowResume(True)
ds=SimplePNGMaskDataSet("D:/pics/train","D:/pics/train_mask")
cfg.fit(ds)
```


####My notebooks constantly run out of memory, what can I do to reduce memory usage?

One way to reduce memory usage is to limit augmentation queue limit which is 50 by default, 
like in the following example: 

```python
segmentation_pipeline.impl.datasets.AUGMENTER_QUEUE_LIMIT = 3
```

####How can I run sepate set of augmenters on initial image/mask when replacing backgrounds with Background Augmenter?
```yaml
  BackgroundReplacer:
    rate: 0.5
    path: D:/bg
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


###How can I visualize images that are used for training (after augmentations)?

You should set `showDataExamples` to True like in the following sample
```python
cfg= segmentation.parse("./no_erosion_aug_on_masks/people-1.yaml")
cfg.showDataExamples=True
```
if will lead to generation of training images samples and storing them in examples folder at the end of each epoch

###What I can do if i have some extra training data, that should not be included into validation, but should be used during the training?

```python
extra_data=NotzeroSimplePNGMaskDataSet("D:/phaces/all","D:/phaces/masks") #My dataset that should be added to training
segmentation.extra_train["people"] = extra_data
```   

and in the config file:

```yaml
extra_train_data: people
```

### How to get basic statistics across my folds/stages


This code sample will return primary metric stats over folds/stages
```
cfg= segmentation.parse("./no_erosion_aug_on_masks/people-1.yaml")
metrics = cfg.info()
```


###I have some callbacks that are configured globally, but I need some extra callbacks for my last training stage?

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
