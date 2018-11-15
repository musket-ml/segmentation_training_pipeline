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

   