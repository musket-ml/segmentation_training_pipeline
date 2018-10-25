import imgaug
import segmentation_models
import numpy as np
from segmentation_models.utils import set_trainable
import keras.optimizers as opt
import keras
from impl import datasets, configloader
import os
import yaml
import impl.losses
import impl.focal_loss
keras.utils.get_custom_objects()["dice"]=impl.losses.dice_coef
keras.utils.get_custom_objects()["iou"]=impl.losses.iou_coef
keras.utils.get_custom_objects()["iot"]=impl.losses.iot_coef

keras.utils.get_custom_objects()["lovasz_loss"]=impl.losses.lovasz_loss
keras.utils.get_custom_objects()["iou_loss"]=impl.losses.iou_coef_loss
keras.utils.get_custom_objects()["dice_loss"]=impl.losses.dice_coef_loss
keras.utils.get_custom_objects()["jaccard_loss"]=impl.losses.jaccard_distance_loss
keras.utils.get_custom_objects()["focal_loss"]=impl.focal_loss.focal_loss(gamma=1)
from impl.deeplab import model as dlm
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

class PipelineConfig:

    def fit(self, d, subsample=1.0, foldsToExecute=None):
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
        if stage == -1: stage = len(self.stages) - 1
        ec = ExecutionConfig(fold=fold, stage=stage, subsample=1.0, dr=os.path.dirname(self.path))
        model = self.createAndCompile()
        model.load_weights(ec.weightsPath())
        return model

    def predict_on_directory(self, path, fold=0, stage=0, limit=-1, batchSize=32):
        mdl = self.load_model(fold, stage)
        ta = self.transformAugmentor()
        for v in datasets.DirectoryDataSet(path, batchSize).generator(limit):
            for z in ta.augment_batches([v]):
                res = mdl.predict(np.array(z.images_aug))
                z.segmentation_maps_aug = [imgaug.SegmentationMapOnImage(x, x.shape) for x in res];
                yield z

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
        transforms.append(imgaug.augmenters.Scale(size=(self.shape[0], self.shape[1])))
        return imgaug.augmenters.Sequential(transforms)

    def compile(self, net: keras.Model, opt: keras.optimizers.Optimizer, loss=None):
        if loss:
            net.compile(opt, loss, self.metrics)
        else:
            net.compile(opt, self.loss, self.metrics)
        return net

    def createAndCompile(self, lr=None, loss=None):
        return self.compile(self.createNet(), self.createOptimizer(lr=lr), loss=loss);

    def kfold(self, ds, indeces):
        transforms = [] + self.transforms
        transforms.append(imgaug.augmenters.Scale(size=(self.shape[0], self.shape[1])))
        return datasets.KFoldedDataSet(ds, indeces, self.augmentation, transforms, batchSize=self.batch)


def parse(path) -> PipelineConfig:
    cfg = configloader.parse("segmentation", path)
    cfg.path = path;
    return cfg


class ExecutionConfig:
    def __init__(self, fold=0, stage=0, subsample=1.0, dr: str = ""):
        self.subsample = subsample
        self.stage = stage
        self.fold = fold
        self.dirName = dr
        pass

    def weightsPath(self):
        return os.path.join(self.dirName, "best-" + str(self.fold) + "." + str(self.stage) + ".weights")

    def metricsPath(self):
        return os.path.join(self.dirName, "metrics-" + str(self.fold) + "." + str(self.stage) + ".csv")


class Stage:
    def __init__(self, dict, cfg: PipelineConfig):
        self.dict = dict
        self.cfg = cfg;
        if 'negatives' in dict:
            self.negatives = dict['negatives']
        else:
            self.negatives = "real"
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

        if 'callbacks' in self.dict:
            cb = configloader.parse("callbacks", self.dict['callbacks'])
        cb.append(keras.callbacks.CSVLogger(ec.metricsPath()))
        md = self.cfg.primary_metric_mode
        cb.append(
            keras.callbacks.ModelCheckpoint(ec.weightsPath(), save_best_only=True, monitor=self.cfg.primary_metric,
                                            mode=md, verbose=1))
        cb.append(DrawResults(self.cfg,kf,ec.fold,ec.stage,negatives=self.negatives))
        kf.trainOnFold(ec.fold, model, cb, self.epochs, self.negatives, subsample=ec.subsample)
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
            try:
                os.makedirs(dr);
            except:
                pass
            datasets.draw_test_batch(i, os.path.join(dr,"t_epoch_"+str(epoch)+"." + str(num) + '.jpg'))
            num = num + 1
        pass