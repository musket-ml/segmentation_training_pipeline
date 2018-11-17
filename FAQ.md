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
