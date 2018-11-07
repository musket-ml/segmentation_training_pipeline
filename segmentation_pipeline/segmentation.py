import imgaug
import segmentation_models
import numpy as np
import tqdm
from segmentation_models.utils import set_trainable
import keras.optimizers as opt
import keras
from segmentation_pipeline.impl import datasets, configloader
import os
import yaml
import segmentation_pipeline.impl.losses
import segmentation_pipeline.impl.focal_loss
import imageio
keras.utils.get_custom_objects()["dice"]= segmentation_pipeline.impl.losses.dice_coef
keras.utils.get_custom_objects()["dice_bool"]= segmentation_pipeline.impl.losses.dice
keras.utils.get_custom_objects()["iou"]= segmentation_pipeline.impl.losses.iou_coef
keras.utils.get_custom_objects()["iot"]= segmentation_pipeline.impl.losses.iot_coef

keras.utils.get_custom_objects()["lovasz_loss"]= segmentation_pipeline.impl.losses.lovasz_loss
keras.utils.get_custom_objects()["iou_loss"]= segmentation_pipeline.impl.losses.iou_coef_loss
keras.utils.get_custom_objects()["dice_loss"]= segmentation_pipeline.impl.losses.dice_coef_loss
keras.utils.get_custom_objects()["jaccard_loss"]= segmentation_pipeline.impl.losses.jaccard_distance_loss
keras.utils.get_custom_objects()["focal_loss"]= segmentation_pipeline.impl.focal_loss.focal_loss(gamma=1)
from segmentation_pipeline.impl.deeplab import model as dlm

def copy_if_exist(name: str, fr: dict, trg: dict):
    if name in fr:
        trg[name] = fr[name]


def create_with(names: [str], fr: dict):
    res = {}
    for v in names:
        copy_if_exist(v, fr, res)
    return res;

custom_models={
    "DeepLabV3":dlm.Deeplabv3
}
dataset_augmenters={

}
class AnsembleModel:
    def __init__(self,models):
        self.models=models;

    def predict(self,data):
        res=[]
        for m in self.models:
            res.append(m.predict(data))

        rm=res[0]
        for r in range(1,len(self.models)):
            rm=rm+res[r];
        return rm/float(len(self.models));

class PipelineConfig:

    def fit(self, d, subsample=1.0, foldsToExecute=None, start_from_stage=0):
        dn = os.path.dirname(self.path)
        if os.path.exists(os.path.join(dn, "summary.yaml")):
            raise ValueError("Experiment is already finished!!!!")
        folds = self.kfold(d, range(0, len(d)))

        for i in range(len(folds.folds)):
            if foldsToExecute:
                if not i in foldsToExecute:
                    continue
            model = self.createAndCompile()
            for s in range(0, len(self.stages)):
                if s<start_from_stage:
                    st: Stage = self.stages[s]
                    ec = ExecutionConfig(fold=i, stage=s, subsample=subsample, dr=os.path.dirname(self.path))
                    if os.path.exists(ec.weightsPath()):
                        model.load_weights(ec.weightsPath())
                    continue
                st: Stage = self.stages[s]
                ec = ExecutionConfig(fold=i, stage=s, subsample=subsample, dr=os.path.dirname(self.path))
                st.execute(folds, model, ec)

        with open(os.path.join(dn, "summary.yaml"), "w") as f:
            yaml.dump(
                {"completed": True, "cfgName": os.path.basename(self.path), "subsample": subsample,
                 "folds": foldsToExecute},
                f)

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

    def __init__(self, **atrs):
        self.batch = 8
        self.all = atrs
        self.augmentation = []
        self.transforms = []
        self.stages = []
        self.callbacks = []
        self.path = None
        self.primary_metric = "val_binary_accuracy"
        self.primary_metric_mode = "auto"
        self.dataset_augmenter=None
        for v in atrs:
            val = atrs[v];
            if v == 'augmentation':
                val = configloader.parse("augmenters", val)
            if v == 'transforms':
                val = configloader.parse("augmenters", val)
            if v == 'callbacks':
                val = configloader.parse("callbacks", val)
            if v == 'stages':
                val = [Stage(x, self) for x in val]
            setattr(self, v, val)
        pass

    def load_model(self, fold: int = 0, stage: int = -1):
        if isinstance(fold,list):
            mdl=[];
            for i in fold:
                mdl.append(self.load_model(i,stage))
            return AnsembleModel(mdl)
        if stage == -1: stage = len(self.stages) - 1
        ec = ExecutionConfig(fold=fold, stage=stage, subsample=1.0, dr=os.path.dirname(self.path))
        model = self.createAndCompile()
        model.load_weights(ec.weightsPath())
        return model

    def predict_on_directory(self, path, fold=0, stage=0, limit=-1, batchSize=32,ttflips=False):
        mdl = self.load_model(fold, stage)
        ta = self.transformAugmentor()
        for v in datasets.DirectoryDataSet(path, batchSize).generator(limit):
            for z in ta.augment_batches([v]):
                o1=np.array(z.images_aug);
                res = mdl.predict(o1)
                if ttflips:
                    another=imgaug.augmenters.Fliplr(1.0).augment_images(z.images_aug);
                    res1= mdl.predict(np.array(another))
                    res1=imgaug.augmenters.Fliplr(1.0).augment_images(res1)

                    another1 = imgaug.augmenters.Flipud(1.0).augment_images(z.images_aug);
                    res2 = mdl.predict(np.array(another1))
                    res2 = imgaug.augmenters.Flipud(1.0).augment_images(res2)

                    seq=imgaug.augmenters.Sequential([imgaug.augmenters.Fliplr(1.0), imgaug.augmenters.Flipud(1.0)])
                    another2 = seq.augment_images(z.images_aug);
                    res3 = mdl.predict(np.array(another2))
                    res3 = seq.augment_images(res3)

                    res=(res+res1+res2+res3)/4.0
                z.segmentation_maps_aug = [imgaug.SegmentationMapOnImage(x, x.shape) for x in res];
                yield z

    def predict_to_directory(self, spath, tpath,fold=0, stage=0, limit=-1, batchSize=32):
        ensure(tpath)
        with tqdm.tqdm(total=len(os.listdir(spath)),unit="files",desc="segmentation of images from "+spath+" to "+tpath) as pbar:
            for v in self.predict_on_directory(spath,fold=fold, stage=stage, limit=limit, batchSize=batchSize):
                b:imgaug.Batch=v;
                for i in range(len(b.data)):
                    id=b.data[i];
                    orig=b.images[i];
                    map=b.segmentation_maps_aug[i]
                    scaledMap=imgaug.augmenters.Scale({"height": orig.shape[0], "width": orig.shape[1]}).augment_segmentation_maps([map])
                    imageio.imwrite(os.path.join(tpath, id[0:id.index('.')] + ".png"), (scaledMap[0].arr*255).astype(np.uint8))
                pbar.update(batchSize)
    def predict_in_directory(self, spath, fold, stage,cb, data,limit=-1, batchSize=32,ttflips=False):

        with tqdm.tqdm(total=len(os.listdir(spath)),unit="files",desc="segmentation of images from "+spath) as pbar:
            for v in self.predict_on_directory(spath,fold=fold, stage=stage, limit=limit, batchSize=batchSize,ttflips=ttflips):
                b:imgaug.Batch=v;
                for i in range(len(b.data)):
                    id=b.data[i];
                    orig=b.images[i];
                    map=b.segmentation_maps_aug[i]
                    scaledMap=imgaug.augmenters.Scale({"height": orig.shape[0], "width": orig.shape[1]}).augment_segmentation_maps([map])
                    cb(id,scaledMap[0],data)
                pbar.update(batchSize)


    def createNet(self):
        if self.architecture in custom_models:
            clazz=custom_models[self.architecture]
        else: clazz = getattr(segmentation_models, self.architecture)
        t: configloader.Type = configloader.loaded['segmentation'].catalog['PipelineConfig']
        r = t.custom()
        cleaned = {}
        for arg in self.all:
            pynama = t.alias(arg);
            if not arg in r:
                cleaned[pynama] = self.all[arg];
        return clazz(**cleaned)

    def createOptimizer(self, lr=None):
        r = getattr(opt, self.optimizer)
        ds = create_with(["lr", "clipnorm", "clipvalue"], self.all);
        if lr:
            ds["lr"] = lr
        return r(**ds)

    def transformAugmentor(self):
        transforms = [] + self.transforms
        transforms.append(imgaug.augmenters.Scale({"height": self.shape[0], "width": self.shape[1]}))
        return imgaug.augmenters.Sequential(transforms)

    def compile(self, net: keras.Model, opt: keras.optimizers.Optimizer, loss=None):
        if loss:
            net.compile(opt, loss, self.metrics)
        else:
            net.compile(opt, self.loss, self.metrics)
        return net

    def createAndCompile(self, lr=None, loss=None):
        return self.compile(self.createNet(), self.createOptimizer(lr=lr), loss=loss);

    def evaluateAll(self,ds, fold:int,cb,stage=-1,negatives="real"):
        folds = self.kfold(ds, range(0, len(ds)))
        vl, vg, test_g = folds.generator(fold, False,returnBatch=True);
        indexes = folds.sampledIndexes(fold, False, negatives)
        m = self.load_model(fold, stage)
        num=0
        with tqdm.tqdm(total=len(indexes), unit="files", desc="segmentation of validation set from " + str(fold)) as pbar:
            try:
                for f in test_g():
                    if num>=len(indexes): break
                    x, y, b = f
                    z = m.predict(x)
                    ids=[]
                    augs=[]
                    for i in range(0,len(z)):
                        if num >= len(indexes): break
                        orig=b.images[i]
                        num = num + 1
                        ma=z[i]
                        id=b.data[i]
                        segmentation_maps_aug = [imgaug.SegmentationMapOnImage(ma, ma.shape)];
                        augmented = imgaug.augmenters.Scale(
                                    {"height": orig.shape[0], "width": orig.shape[1]}).augment_segmentation_maps(segmentation_maps_aug)
                        ids.append(id)
                        augs=augs+augmented

                    res=imgaug.Batch(images=b.images,data=ids,segmentation_maps=b.segmentation_maps)
                    res.segmentation_maps_aug=augs
                    yield res
                    pbar.update(len(ids))
            finally:
                vl.terminate();
                vg.terminate();
        pass

    def kfold(self, ds, indeces):
        transforms = [] + self.transforms
        transforms.append(imgaug.augmenters.Scale({"height": self.shape[0], "width": self.shape[1]}))
        kf= datasets.KFoldedDataSet(ds, indeces, self.augmentation, transforms, batchSize=self.batch)
        if self.dataset_augmenter is not None:
            args = dict(self.dataset_augmenter)
            del args["name"]
            ag=dataset_augmenters[self.dataset_augmenter["name"]](**args)
            kf=ag(kf)
            pass
        return kf


def parse(path) -> PipelineConfig:
    cfg = configloader.parse("segmentation", path)
    cfg.path = path;
    return cfg

def ensure(p):
    try:
        os.makedirs(p);
    except:
        pass
class ExecutionConfig:
    def __init__(self, fold=0, stage=0, subsample=1.0, dr: str = ""):
        self.subsample = subsample
        self.stage = stage
        self.fold = fold
        self.dirName = dr
        pass

    def weightsPath(self):
        ensure(os.path.join(self.dirName, "weights"))
        return os.path.join(self.dirName, "weights","best-" + str(self.fold) + "." + str(self.stage) + ".weights")

    def metricsPath(self):
        ensure(os.path.join(self.dirName, "metrics"))
        return os.path.join(self.dirName, "metrics","metrics-" + str(self.fold) + "." + str(self.stage) + ".csv")


class Stage:
    def __init__(self, dict, cfg: PipelineConfig):
        self.dict = dict
        self.cfg = cfg;
        if 'initial_weights' in dict:
            self.initial_weights=dict['initial_weights']
        else: self.initial_weights=None
        if 'negatives' in dict:
            self.negatives = dict['negatives']
        if 'validation_negatives' in dict:
            self.validation_negatives = dict['validation_negatives']
        else:
            self.validation_negatives=None
        self.epochs = dict["epochs"]
        if 'loss' in dict:
            self.loss = dict['loss']
        else:
            self.loss = None
        if 'lr' in dict:
            self.lr = dict['lr']
        else:
            self.lr = None

    def execute(self, kf: datasets.KFoldedDataSet, model: keras.Model, ec: ExecutionConfig):
        if 'unfreeze_encoder' in self.dict and self.dict['unfreeze_encoder']:
            set_trainable(model)
        if self.loss or self.lr:
            self.cfg.compile(model, self.cfg.createOptimizer(self.lr), self.loss)
        cb = [] + self.cfg.callbacks
        if self.initial_weights is not None:
            model.load_weights(self.initial_weights)
        if 'callbacks' in self.dict:
            cb = configloader.parse("callbacks", self.dict['callbacks'])
        cb.append(keras.callbacks.CSVLogger(ec.metricsPath()))
        md = self.cfg.primary_metric_mode
        cb.append(
            keras.callbacks.ModelCheckpoint(ec.weightsPath(), save_best_only=True, monitor=self.cfg.primary_metric,
                                            mode=md, verbose=1))
        cb.append(DrawResults(self.cfg,kf,ec.fold,ec.stage,negatives=self.negatives))
        kf.trainOnFold(ec.fold, model, cb, self.epochs, self.negatives, subsample=ec.subsample,validation_negatives=self.validation_negatives)
        pass


class DrawResults(keras.callbacks.Callback):
    def __init__(self,cfg,folds,fold,stage,negatives,limit=16):
            self.ta = cfg.transformAugmentor()
            self.fold=fold
            self.stage=stage
            self.cfg=cfg
            self.rs = folds.load(fold, False, negatives, limit)
            pass

    def on_epoch_end(self, epoch, logs=None):
        def iter():
            for z in self.ta.augment_batches([self.rs]):
              res = self.model.predict(np.array(z.images_aug))
              z.heatmaps_aug = [imgaug.HeatmapsOnImage(x, x.shape) for x in res];
              yield z
        num=0
        for i in iter():
            dr=os.path.join(os.path.dirname(self.cfg.path),"examples", str(self.stage), str(self.fold))
            ensure(dr)
            datasets.draw_test_batch(i, os.path.join(dr, "t_epoch_" + str(epoch) + "." + str(num) + '.jpg'))
            num = num + 1
        pass

