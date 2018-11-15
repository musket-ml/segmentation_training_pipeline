import imgaug
import segmentation_models
import numpy as np
import tqdm
from segmentation_models.utils import set_trainable
import keras.optimizers as opt
import keras
import keras.backend as K
from keras.callbacks import  LambdaCallback
from segmentation_pipeline.impl import datasets, configloader
import os
import yaml
import segmentation_pipeline.impl.losses
import segmentation_pipeline.impl.focal_loss
import imageio
import csv
import six
import typing
import collections
import io
from segmentation_pipeline.impl.clr_callback import  CyclicLR
from  segmentation_pipeline.impl.lr_finder import LRFinder
keras.utils.get_custom_objects()["dice"]= segmentation_pipeline.impl.losses.dice_coef
keras.utils.get_custom_objects()["dice_bool"]= segmentation_pipeline.impl.losses.dice
keras.utils.get_custom_objects()["iou"]= segmentation_pipeline.impl.losses.iou_coef
keras.utils.get_custom_objects()["iot"]= segmentation_pipeline.impl.losses.iot_coef

keras.utils.get_custom_objects()["lovasz_loss"]= segmentation_pipeline.impl.losses.lovasz_loss
#keras.utils.get_custom_objects()["lovasz_hinge"]= segmentation_pipeline.impl.losses.lovasz_hinge_flat()
keras.utils.get_custom_objects()["iou_loss"]= segmentation_pipeline.impl.losses.iou_coef_loss
keras.utils.get_custom_objects()["dice_loss"]= segmentation_pipeline.impl.losses.dice_coef_loss
keras.utils.get_custom_objects()["jaccard_loss"]= segmentation_pipeline.impl.losses.jaccard_distance_loss
keras.utils.get_custom_objects()["focal_loss"]= segmentation_pipeline.impl.focal_loss.focal_loss
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
import keras.applications as app
from  segmentation_pipeline.impl import composite
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

def crop(x, y,x1,y1, image):
    return image[:,y:y1, x:x1, :]

def addCrop(image,x,y,x1,y1,image1):
    image[:,y:y1, x:x1, :]+=image1

class BatchCrop:

    def __init__(self,parts,mdl):
        self.parts=parts
        self.mdl=mdl

    def predict(self,images):
        cell_width=images.shape[2]/self.parts
        cell_height = images.shape[1]/self.parts
        rs=None
        for i in range(self.parts):
            for j in range(self.parts):
                x=round(i*cell_width)
                y=round(j*cell_height)
                x1=x+round(cell_width)
                y1=y+round(cell_height)
                predCrop=self.mdl.predict(crop(x,y,x1,y1,images));
                if rs is None:
                    rs=np.zeros((images.shape[0],images.shape[1],images.shape[2],predCrop.shape[-1]),dtype=np.float32)
                addCrop(rs,x,y,x1,y1,predCrop)

        return rs

class CropAndSplit:
    def __init__(self,orig,n):
        self.ds=orig
        self.parts=n
        self.lastPos=None


    def isPositive(self, item):
        pos = item // (self.parts * self.parts);
        return self.ds.isPositive(pos)

    def __getitem__(self, item):
        pos=item//(self.parts*self.parts);
        off=item%(self.parts*self.parts)
        if pos==self.lastPos:
            dm=self.lastImage
        else:
            dm=self.ds[pos]
            self.lastImage=dm
        row=off//self.parts
        col=off%self.parts
        x,y=dm.x,dm.y
        x1,y1= self.crop(row,col,x),self.crop(row,col,y)
        return datasets.PredictionItem(dm.id,x1,y1)

    def crop(self,y,x,image):
        h=image.shape[0]//self.parts
        w = image.shape[1] // self.parts
        return image[h*y:h*(y+1),w*x:w*(x+1), :]

    def __len__(self):
        return len(self.ds)*self.parts*self.parts

class PipelineConfig:

    def setAllowResume(self,resume):
        self.resume=resume

    def fit(self, d, subsample=1.0, foldsToExecute=None, start_from_stage=0):
        if self.crops is not None:
            d=CropAndSplit(d,self.crops)
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
                    self.skip_stage(i, model, s, subsample)
                    continue
                st: Stage = self.stages[s]
                ec = ExecutionConfig(fold=i, stage=s, subsample=subsample, dr=os.path.dirname(self.path))
                st.execute(folds, model, ec)

        with open(os.path.join(dn, "summary.yaml"), "w") as f:
            yaml.dump(
                {"completed": True, "cfgName": os.path.basename(self.path), "subsample": subsample,
                 "folds": foldsToExecute},
                f)

    def lr_find(self, d, foldsToExecute=None,stage=0,subsample=1.0,start_lr=0.000001,end_lr=1.0,epochs=5):
        dn = os.path.dirname(self.path)
        if os.path.exists(os.path.join(dn, "summary.yaml")):
            raise ValueError("Experiment is already finished!!!!")
        folds = self.kfold(d)

        for i in range(len(folds.folds)):
            if foldsToExecute:
                if not i in foldsToExecute:
                    continue
            model = self.createAndCompile()
            for s in range(0, len(self.stages)):
                if s<stage:
                    self.skip_stage(i, model, s, subsample)
                    continue
                st: Stage = self.stages[s]
                ec = ExecutionConfig(fold=i, stage=s, subsample=subsample, dr=os.path.dirname(self.path))
                return st.lr_find(folds, model, ec,start_lr,end_lr,epochs)

    def skip_stage(self, i, model, s, subsample):
        st: Stage = self.stages[s]
        ec = ExecutionConfig(fold=i, stage=s, subsample=subsample, dr=os.path.dirname(self.path))
        if os.path.exists(ec.weightsPath()):
            model.load_weights(ec.weightsPath())
            if 'unfreeze_encoder' in st.dict and st.dict['unfreeze_encoder']:
                set_trainable(model)

    def fit_classifier(self,d,fold:int,model:keras.Model,batchSize=None,stage=0):
        if batchSize==None:
            batchSize=self.batch
        fld = self.kfold(d,indeces=None,batch=batchSize);
        indeces = fld.indexes(fold, True)
        vindeces = fld.indexes(fold, False)

        r, v, rs = fld.classification_generator_from_indexes(indeces);
        r1, v1, rs1 = fld.classification_generator_from_indexes(vindeces);
        try:
            ec = ExecutionConfig(fold=fold, stage=stage, dr=os.path.dirname(self.path))
            cb=[]+self.callbacks
            cb.append(keras.callbacks.CSVLogger(ec.classifier_metricsPath()))
            cb.append(keras.callbacks.ModelCheckpoint(ec.classifier_weightsPath(), save_best_only=True, monitor="val_binary_accuracy",verbose=1))

            model.fit_generator(rs(), len(indeces) / batchSize, 20, validation_data=rs1(), validation_steps=len(vindeces) / batchSize,callbacks=cb)
            pass
        finally:
            r.terminate()
            v.terminate()
            r1.terminate()
            v1.terminate()

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
        self.bgr=None
        self.rate=0.5
        self.crops=None
        self.resume=False
        for v in atrs:
            val = atrs[v];
            if v == 'augmentation' and val is not None:
                if "BackgroundReplacer" in val:
                    bgr=val["BackgroundReplacer"]
                    erosion=0;
                    if "erosion" in val:
                        erosion=val["erosion"]
                    self.bgr=datasets.Backgrounds(bgr["path"],erosion=erosion)
                    self.bgr.rate = bgr["rate"]
                    del val["BackgroundReplacer"]
                val = configloader.parse("augmenters", val)
            if v == 'transforms':
                val = configloader.parse("augmenters", val)
            if v == 'callbacks':
                cs=[]
                if "CyclicLR" in val and val is not None:
                    bgr = val["CyclicLR"]
                    cs.append(CyclicLR(**bgr))
                    #self.bgr = datasets.Backgrounds(bgr["path"])
                    #self.bgr.rate = bgr["rate"]
                    del val["CyclicLR"]
                val = configloader.parse("callbacks", val)
                if val is not None:
                    val=val+cs
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
        if self.crops is not None:
            mdl=BatchCrop(self.crops,mdl)
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

    def predict_on_directory_with_model(self, mdl,path, limit=-1, batchSize=32,ttflips=False):

        ta = self.transformAugmentor()
        with tqdm.tqdm(total=len(self.dir_list(path)), unit="files", desc="classifying positive  images from " + path) as pbar:
            for v in datasets.DirectoryDataSet(path, batchSize).generator(limit):
                for z in ta.augment_batches([v]):
                    o1=np.array(z.images_aug);
                    res = mdl.predict(o1)
                    if ttflips:
                        another=imgaug.augmenters.Fliplr(1.0).augment_images(z.images_aug);
                        res1= mdl.predict(np.array(another))


                        another1 = imgaug.augmenters.Flipud(1.0).augment_images(z.images_aug);
                        res2 = mdl.predict(np.array(another1))


                        seq=imgaug.augmenters.Sequential([imgaug.augmenters.Fliplr(1.0), imgaug.augmenters.Flipud(1.0)])
                        another2 = seq.augment_images(z.images_aug);
                        res3 = mdl.predict(np.array(another2))


                        res=(res+res1+res2+res3)/4.0
                    z.predictions = res;
                    pbar.update(batchSize)
                    yield z

    def predict_to_directory(self, spath, tpath,fold=0, stage=0, limit=-1, batchSize=32):



        ensure(tpath)
        with tqdm.tqdm(total=len(self.dir_list(spath)), unit="files", desc="segmentation of images from " + str(spath) + " to " + str(tpath)) as pbar:
            for v in self.predict_on_directory(spath,fold=fold, stage=stage, limit=limit, batchSize=batchSize):
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
                    imageio.imwrite(os.path.join(tp, id[0:id.index('.')] + ".png"), (scaledMap[0].arr*255).astype(np.uint8))
                pbar.update(batchSize)

    def dir_list(self, spath):
        if isinstance(spath,datasets.ConstrainedDirectory):
            return spath.filters
        return os.listdir(spath)

    def predict_in_directory(self, spath, fold, stage,cb, data,limit=-1, batchSize=32,ttflips=False):

        with tqdm.tqdm(total=len(self.dir_list(spath)), unit="files", desc="segmentation of images from " + str(spath)) as pbar:
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
        if self.crops is not None:
            cleaned["input_shape"]=(cleaned["input_shape"][0]//self.crops,cleaned["input_shape"][1]//self.crops,cleaned["input_shape"][2])

        return clazz(**cleaned)

    def createAndCompileClassifier(self,lr=0.0001):
        mdl: keras.Model = app.DenseNet201(input_shape=self.shape, include_top=False)
        l0 = keras.layers.GlobalAveragePooling2D(name='avg_pool')(mdl.layers[-1].output)
        l = keras.layers.Dense(1, activation=keras.activations.sigmoid)(l0)
        mdl = keras.Model(mdl.layers[0].input, l)
        mdl.compile(keras.optimizers.Adam(lr=lr), keras.losses.binary_crossentropy, metrics=["binary_accuracy"])
        return mdl

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

    def compile(self, net: keras.Model, opt: keras.optimizers.Optimizer, loss:str=None):
        if loss is not None and "+" in loss:
            loss=composite.ps(loss)

        if (loss=='lovasz_loss' and isinstance(net.layers[-1],keras.layers.Activation)):
            net=keras.Model(net.layers[0].input,net.layers[-1].input);
            net.summary()
        if loss:
            net.compile(opt, loss, self.metrics)
        else:
            net.compile(opt, self.loss, self.metrics)
        return net

    def createAndCompile(self, lr=None, loss=None):
        return self.compile(self.createNet(), self.createOptimizer(lr=lr), loss=loss);

    def evaluateAll(self,ds, fold:int,stage=-1,negatives="real"):
        folds = self.kfold(ds, range(0, len(ds)))
        vl, vg, test_g = folds.generator(fold, False,negatives=negatives,returnBatch=True);
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
                    res.predicted_maps_aug=augs
                    yield res
                    pbar.update(len(ids))
            finally:
                vl.terminate();
                vg.terminate();
        pass

    def kfold(self, ds, indeces=None,batch=None):
        if batch==None:
            batch=self.batch
        if indeces is None: indeces=range(0,len(ds))
        transforms = [] + self.transforms
        transforms.append(imgaug.augmenters.Scale({"height": self.shape[0], "width": self.shape[1]}))
        if self.bgr is not None:
            ds=datasets.WithBackgrounds(ds,self.bgr)
        kf= datasets.KFoldedDataSet(ds, indeces, self.augmentation, transforms, batchSize=batch)
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

    def classifier_weightsPath(self):
        ensure(os.path.join(self.dirName, "classify_weights"))
        return os.path.join(self.dirName, "classify_weights","best-" + str(self.fold) + "." + str(self.stage) + ".weights")

    def metricsPath(self):
        ensure(os.path.join(self.dirName, "metrics"))
        return os.path.join(self.dirName, "metrics","metrics-" + str(self.fold) + "." + str(self.stage) + ".csv")

    def classifier_metricsPath(self):
        ensure(os.path.join(self.dirName, "classify_metrics"))
        return os.path.join(self.dirName, "classify_metrics","metrics-" + str(self.fold) + "." + str(self.stage) + ".csv")

def maxEpoch(file):
    if not os.path.exists(file):
        return -1;
    with open(file, 'r') as csvfile:
         spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
         epoch=-1;
         num=0;
         for row in spamreader:
             if num>0:
                epoch=max(epoch,int(row[0]))
             num = num + 1;
         return epoch;


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

    def lr_find(self, kf: datasets.KFoldedDataSet, model: keras.Model, ec: ExecutionConfig,start_lr,end_lr,epochs):
        if 'unfreeze_encoder' in self.dict and self.dict['unfreeze_encoder']:
            set_trainable(model)
        if self.loss or self.lr:
            self.cfg.compile(model, self.cfg.createOptimizer(self.lr), self.loss)
        cb = [] + self.cfg.callbacks
        if self.initial_weights is not None:
            model.load_weights(self.initial_weights)
        ll=LRFinder(model)
        num_batches=kf.numBatches(ec.fold,self.negatives,ec.subsample)*epochs
        ll.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(num_batches))
        K.set_value(model.optimizer.lr, start_lr)
        callback = LambdaCallback(on_batch_end=lambda batch, logs: ll.on_batch_end(batch, logs))
        cb.append(callback)
        kf.trainOnFold(ec.fold, model, cb,epochs, self.negatives, subsample=ec.subsample,validation_negatives=self.validation_negatives)
        return ll

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
        kepoch=-1;
        if self.cfg.resume:
            kepoch=maxEpoch(ec.metricsPath())
            if kepoch!=-1:
                self.epochs=self.epochs-kepoch
                if os.path.exists(ec.weightsPath()):
                    model.load_weights(ec.weightsPath())
                cb.append(CSVLogger(ec.metricsPath(),append=True,start=kepoch))
            else: cb.append(CSVLogger(ec.metricsPath()))
        else:    cb.append(CSVLogger(ec.metricsPath()))
        md = self.cfg.primary_metric_mode
        cb.append(
            keras.callbacks.ModelCheckpoint(ec.weightsPath(), save_best_only=True, monitor=self.cfg.primary_metric,
                                            mode=md, verbose=1))
        cb.append(DrawResults(self.cfg,kf,ec.fold,ec.stage,negatives=self.negatives))
        if self.epochs-kepoch==0:
            return;
        kf.trainOnFold(ec.fold, model, cb, self.epochs-kepoch, self.negatives, subsample=ec.subsample,validation_negatives=self.validation_negatives)
        pass


class CSVLogger(keras.callbacks.Callback):
    """Callback that streams epoch results to a csv file.

    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.

    # Example

    ```python
    csv_logger = CSVLogger('training.log')
    model.fit(X_train, Y_train, callbacks=[csv_logger])
    ```

    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, filename, separator=',', append=False,start=0):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.start=start
        self.append_header = True

        self.file_flags = ''
        self._open_args = {'newline': '\n'}
        super(CSVLogger, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'
        self.csv_file = io.open(self.filename,
                                mode + self.file_flags,
                                **self._open_args)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch=epoch+self.start
        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, collections.Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k] if k in logs else 'NA') for k in self.keys])

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep
                def __init__(self):
                    self.delimiter=","
            fieldnames = ['epoch'] + self.keys
            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=fieldnames,
                                         dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = collections.OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None

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



