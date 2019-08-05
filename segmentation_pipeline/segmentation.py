import imgaug
import segmentation_models
import numpy as np
import tqdm
from segmentation_models.utils import set_trainable
import keras
from musket_core import configloader, datasets
import os
import musket_core.losses
from musket_core.datasets import DataSet, WriteableDataSet, DirectWriteableDS,CompressibleWriteableDS
import imageio
import inspect
keras.utils.get_custom_objects()["dice"]= musket_core.losses.dice
keras.utils.get_custom_objects()["iou"]= musket_core.losses.iou_coef
keras.utils.get_custom_objects()["iot"]= musket_core.losses.iot_coef
keras.utils.get_custom_objects()["lovasz_loss"]= musket_core.losses.lovasz_loss
keras.utils.get_custom_objects()["iou_loss"]= musket_core.losses.iou_coef_loss
keras.utils.get_custom_objects()["dice_loss"]= musket_core.losses.dice_coef_loss
keras.utils.get_custom_objects()["jaccard_loss"]= musket_core.losses.jaccard_distance_loss
keras.utils.get_custom_objects()["focal_loss"]= musket_core.losses.focal_loss

from segmentation_pipeline.impl.deeplab import model as dlm
import musket_core.generic_config as generic

ansemblePredictions=generic.ansemblePredictions
dataset_augmenters=generic.dataset_augmenters
extra_train=generic.extra_train

custom_models={
    "DeepLabV3":dlm.Deeplabv3
}


class PipelineConfig(generic.GenericImageTaskConfig):

    def evaluate(self, d, fold, stage, negatives="all", limit=16):
        mdl = self.load_model(fold, stage)
        ta = self.transformAugmentor()
        folds = self.kfold(d, range(0, len(d)))
        rs = folds.load(fold, False, negatives, limit)

        for z in ta.augment_batches([rs]):
            res = mdl.predict(np.array(z.images_aug))
            z.heatmaps_aug = [imgaug.HeatmapsOnImage(x, x.shape) for x in res];
            yield z
        pass

    def createStage(self,x):
        return SegmentationStage(x,self)

    def  __init__(self,**atrs):
        super().__init__(**atrs)
        self.dataset_clazz = datasets.ImageKFoldedDataSet
        self.flipPred=True
        
        pass

    def update(self,z,res):
        z.segmentation_maps_aug = [imgaug.SegmentationMapOnImage(x, x.shape) for x in res];
        pass

    def predict_to_directory(self, spath, tpath,fold=0, stage=0, limit=-1, batchSize=32,binaryArray=False,ttflips=False):
        generic.ensure(tpath)
        with tqdm.tqdm(total=len(generic.dir_list(spath)), unit="files", desc="segmentation of images from " + str(spath) + " to " + str(tpath)) as pbar:
            for v in self.predict_on_directory(spath, fold=fold, stage=stage, limit=limit, batch_size=batchSize, ttflips=ttflips):
                b:imgaug.Batch=v;
                for i in range(len(b.data)):
                    id=b.data[i];
                    orig=b.images[i];
                    map=b.segmentation_maps_aug[i]
                    scaledMap=imgaug.augmenters.Scale({"height": orig.shape[0], "width": orig.shape[1]}).augment_segmentation_maps([map])
                    if isinstance(tpath, datasets.ConstrainedDirectory):
                        tp=tpath.path
                    else:
                        tp=tpath
                    if binaryArray:
                        np.save(os.path.join(tp, id[0:id.index('.')]),scaledMap[0].arr);
                    else: imageio.imwrite(os.path.join(tp, id[0:id.index('.')] + ".png"), (scaledMap[0].arr*255).astype(np.uint8))
                pbar.update(batchSize)

    def predict_in_directory(self, spath, fold, stage,cb, data,limit=-1, batchSize=32,ttflips=False):
        with tqdm.tqdm(total=len(generic.dir_list(spath)), unit="files", desc="segmentation of images from " + str(spath)) as pbar:
            for v in self.predict_on_directory(spath, fold=fold, stage=stage, limit=limit, batch_size=batchSize, ttflips=ttflips):
                b:imgaug.Batch=v;
                for i in range(len(b.data)):
                    id=b.data[i];
                    orig=b.images[i];
                    map=b.segmentation_maps_aug[i]
                    scaledMap=imgaug.augmenters.Scale({"height": orig.shape[0], "width": orig.shape[1]}).augment_segmentation_maps([map])
                    cb(id,scaledMap[0],data)
                pbar.update(batchSize)

    def createNet(self):
        ac = self.all["activation"]
        if ac == "none":
            ac = None

        self.all["activation"]=ac
        if self.architecture in custom_models:
            clazz=custom_models[self.architecture]
        else: clazz = getattr(segmentation_models, self.architecture)
        t: configloader.Type = configloader.loaded['segmentation'].catalog['PipelineConfig']
        r = t.custom()
        cleaned = {}
        sig=inspect.signature(clazz)
        for arg in self.all:
            pynama = t.alias(arg)
            if not arg in r and pynama in sig.parameters:
                cleaned[pynama] = self.all[arg]

        self.clean(cleaned)


        if self.crops is not None:
            cleaned["input_shape"]=(cleaned["input_shape"][0]//self.crops,cleaned["input_shape"][1]//self.crops,cleaned["input_shape"][2])

        if "input_shape" in cleaned and cleaned["input_shape"][2]>3 and self.encoder_weights!=None and len(self.encoder_weights)>0:
            if os.path.exists(self.path + ".mdl-nchannel"):
                cleaned["encoder_weights"] = None
                model = clazz(**cleaned)
                model.load_weights(self.path + ".mdl-nchannel")
                return  model

            copy=cleaned.copy();
            copy["input_shape"] = (cleaned["input_shape"][0] , cleaned["input_shape"][1] , 3)
            model1=clazz(**copy);
            cleaned["encoder_weights"]=None
            model=clazz(**cleaned)
            self.adaptNet(model,model1,self.copyWeights);
            model.save_weights(self.path + ".mdl-nchannel")
            return model

        return clazz(**cleaned)


    def evaluateAll(self,ds, fold:int,stage=-1,negatives="real",ttflips=None):
        folds = self.kfold(ds, range(0, len(ds)))
        vl, vg, test_g = folds.generator(fold, False,negatives=negatives,returnBatch=True)
        indexes = folds.sampledIndexes(fold, False, negatives)
        m = self.load_model(fold, stage)
        num=0
        with tqdm.tqdm(total=len(indexes), unit="files", desc="segmentation of validation set from " + str(fold)) as pbar:
            try:
                for f in test_g():
                    if num>=len(indexes): break
                    x, y, b = f
                    z = self.predict_on_batch(m,ttflips,b)
                    ids=[]
                    augs=[]
                    for i in range(0,len(z)):
                        if num >= len(indexes): break
                        orig=b.images[i]
                        num = num + 1
                        ma=z[i]
                        id=b.data[i]
                        segmentation_maps_aug = [imgaug.SegmentationMapOnImage(ma, ma.shape)]
                        augmented = imgaug.augmenters.Scale(
                                    {"height": orig.shape[0], "width": orig.shape[1]}).augment_segmentation_maps(segmentation_maps_aug)
                        ids.append(id)
                        augs=augs+augmented

                    res=imgaug.Batch(images=b.images,data=ids,segmentation_maps=b.segmentation_maps)
                    res.predicted_maps_aug=augs
                    yield res
                    pbar.update(len(ids))
            finally:
                vl.terminate()
                vg.terminate()
        pass

    def get_eval_batch(self)->int:
        return self.inference_batch

    def load_writeable_dataset(self, ds, path)->DataSet:
        resName = (ds.name if hasattr(ds, "name") else "") + "_predictions"
        result = CompressibleWriteableDS(ds, resName, path, len(ds),asUints=self.compressPredictionsAsInts)
        return result

    def create_writeable_dataset(self, dataset:DataSet, dsPath:str)->WriteableDataSet:
        resName = (dataset.name if hasattr(dataset, "name") else "") + "_predictions"
        result = CompressibleWriteableDS(dataset, resName, dsPath,asUints=self.compressPredictionsAsInts)
        return result


def parse(path) -> PipelineConfig:
    cfg = configloader.parse("segmentation", path)
    cfg.path = path
    return cfg


class DrawResults(keras.callbacks.Callback):


    def __init__(self,cfg,folds,fold,stage,negatives,limit=16,train=False, drawingFunction = datasets.draw_test_batch):
            super().__init__()
            if train:
                self.ta = folds.augmentor(isTrain=True)
            else: self.ta = cfg.transformAugmentor()
            self.fold=fold
            self.stage=stage
            self.cfg=cfg
            self.train=train
            self.rs = folds.load(fold, train, negatives, limit)
            self.drawingFunction = drawingFunction
            pass

    def on_epoch_end(self, epoch, logs=None):
        def iter():
            for z in self.ta.augment_batches([self.rs]):
              res = self.model.predict(np.array(z.images_aug))
              z.heatmaps_aug = [imgaug.SegmentationMapOnImage(x>0.5, x.shape) for x in res]
              yield z
        num=0
        for i in iter():
            dr=os.path.join(os.path.dirname(self.cfg.path),"examples", str(self.stage), str(self.fold))
            generic.ensure(dr)
            if self.train:
                self.drawingFunction(i, os.path.join(dr, "t_epoch_train" + str(epoch) + "." + str(num) + '.jpg'))
            else: self.drawingFunction(i, os.path.join(dr, "t_epoch_" + str(epoch) + "." + str(num) + '.jpg'))
            num = num + 1
        pass

class SegmentationStage(generic.Stage):

    def add_visualization_callbacks(self, cb, ec, kf):
        drawingFunction = ec.drawingFunction
        if drawingFunction == None:
            drawingFunction = datasets.draw_test_batch
        cb.append(DrawResults(self.cfg, kf, ec.fold, ec.stage, negatives=self.negatives, drawingFunction=drawingFunction))
        if self.cfg.showDataExamples:
            cb.append(DrawResults(self.cfg, kf, ec.fold, ec.stage, negatives=self.negatives, train=True, drawingFunction=drawingFunction))

    def unfreeze(self, model):
        set_trainable(model)