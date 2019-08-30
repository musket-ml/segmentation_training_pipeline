import mmcv
import cv2
from mmcv.image import imread, imwrite
from mmcv.visualization.color import color_val
from mmcv import Config
from mmcv.runner import Runner, DistSamplerSeedHook, Hook
from mmcv.parallel import scatter, collate, MMDataParallel, MMDistributedDataParallel

from mmdet import datasets as mmdetDatasets
from mmdet.core import (DistOptimizerHook, DistEvalmAPHook,
                        CocoDistEvalRecallHook, CocoDistEvalmAPHook,
                        Fp16OptimizerHook)
from mmdet.datasets import build_dataloader
from mmdet.models import RPN
from pycocotools import mask as mask_util

import imgaug

import numpy as np
from mmcv.parallel import DataContainer as DC
import tqdm
from segmentation_models.utils import set_trainable
import keras
from musket_core import configloader, datasets
import os
import os.path as osp
import musket_core.losses
from musket_core.datasets import SubDataSet, PredictionItem
from musket_core.datasets import DataSet, WriteableDataSet, DirectWriteableDS,CompressibleWriteableDS
import imageio



from mmdet import __version__
# from mmdet.datasets import get_dataset
from mmdet.apis.train import build_optimizer, batch_processor
from mmdet.apis import init_dist, get_root_logger, set_random_seed, init_detector, inference_detector, show_result
from mmdet.models import build_detector
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.utils import to_tensor, random_scale
from mmdet.datasets.coco import CocoDataset
import torch
import torch.distributed

# from segmentation_pipeline.impl.deeplab import model as dlm
import musket_core.generic_config as generic
from musket_core.builtin_trainables import OutputMeta
from mmdet.datasets import get_dataset
from typing import Callable

class MMDetWrapper:
    def __init__(self, cfg:Config, weightsPath:str):
        self.cfg = cfg
        self.weightsPath = weightsPath
        self.output_dim = 4

    def __call__(self, *args, **kwargs):
        return OutputMeta(self.output_dim, self)

    def compile(self, *args, **kwargs):
        cfg = self.cfg
        self.model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)#init_detector(self.cfg, self.weightsPath, device='cuda:0')
        # custom_loss = args[1]
        #
        # if not custom_loss in ["multiclass", "regression"]:
        #     custom_loss_tf = keras.losses.get(custom_loss)
        #
        #     t_true = keras.layers.Input((self.output_dim,))
        #     t_pred = keras.layers.Input((self.output_dim,))
        #
        #     def grad1(y_true, y_pred):
        #         return tf.gradients(custom_loss_tf(y_true, y_pred), [y_true, y_pred], stop_gradients=[y_true])
        #
        #     def grad2(y_true, y_pred):
        #         return tf.gradients(grad1(y_true, y_pred), [y_true, y_pred], stop_gradients=[y_true])
        #
        #     def custom_loss_func(y_true, y_pred):
        #         true, pred = self.to_tf(y_true, y_pred)
        #
        #         pred[np.where(pred == 0)] = 0.000001
        #
        #         pred[np.where(pred == 1)] = 1.0 - 0.000001
        #
        #         s = tf.get_default_session()
        #
        #         res_1 = self.eval_func(true, pred, [grad1(t_true, t_pred), t_true, t_pred], s, False)[1]
        #         res_2 = self.eval_func(true, pred, [grad2(t_true, t_pred), t_true, t_pred], s, False)[1]
        #
        #         return self.loss_to_gb(res_1), self.loss_to_gb(res_2)
        #
        #     self.custom_loss_callable = custom_loss_func
        #
        # for item in args[2]:
        #     self.custom_metrics[item] = self.to_tensor(keras.metrics.get(item))
        pass

    def eval_func(self, y_true, y_pred, f, session, mean=True):
        func = f[0]

        arg1 = f[1]
        arg2 = f[2]

        if mean:
            return np.mean(session.run(func, {arg1: y_true, arg2: y_pred}))

        return session.run(func, {arg1: y_true, arg2: y_pred})

    def eval_metrics(self, y_true, y_pred, session):
        # result = {}
        #
        # for item in self.custom_metrics.keys():
        #     preds = y_pred
        #
        #     if generic_config.need_threshold(item):
        #         preds = (preds > 0.5).astype(np.float32)
        #
        #     result[item] = self.eval_func(y_true, preds, self.custom_metrics[item], session)
        #
        # return result
        print("eval_metrics")
        pass

    def to_tensor(self, func):
        # i1 = keras.layers.Input((self.output_dim,))
        # i2 = keras.layers.Input((self.output_dim,))
        #
        # return func(i1, i2), i1, i2
        pass

    def convert_data(self, generator):
        result_x = []
        result_y = []

        for item in generator:
            result_x.append(item[0])
            result_y.append(item[1])

        result_x = np.concatenate(result_x)
        result_y = np.concatenate(result_y)

        result_x = np.reshape(result_x, (len(result_x), -1))
        result_y = np.reshape(result_y, (len(result_y), -1))

        if self.output_dim > 1:
            result_y = np.argmax(result_y, 1)
        else:
            result_y = (result_y > 0.5).flatten()

        return result_x.astype(np.float32), result_y.astype(np.int32)

    def predict(self, *args, **kwargs):
        input = np.array(args)[0]

        input = np.reshape(input, (len(input), -1))

        self.model._n_features = input.shape[1]

        predictions = inference_detector(self.model, input)
        # predictions = self.model.predict(input)
        #
        # if self.output_dim in [1, 2]:
        #     return self.groups_to_vectors(predictions, len(predictions))

        return predictions

    def load_weights(self, path, val):
        # if os.path.exists(path):
        #     self.model._Booster = lightgbm.Booster(model_file=path)
        print("Load weights")
        pass

    def numbers_to_vectors(self, numbers):
        result = np.zeros((len(numbers), self.output_dim))

        count = 0

        if self.output_dim == 1:
            for item in numbers:
                result[count, 0] = item

                count += 1

            return result

        for item in numbers:
            result[count, item] = 1

            count += 1

        return result

    def groups_to_vectors(self, data, length):
        result = np.zeros((length, self.output_dim))

        if self.output_dim == 1:
            result[:, 0] = data

            return result

        if self.output_dim == 2:
            ids = np.array(range(length), np.int32)

            ids = [ids, (data > 0.5).astype(np.int32)]

            result[ids] = 1

            return result

        for item in range(self.output_dim):
            result[:, item] = data[length * item : length * (item + 1)]

        return result

    def to_tf(self, numbers, data):
        y_true = self.numbers_to_vectors(numbers)

        y_pred = self.groups_to_vectors(data, len(numbers))

        return y_true, y_pred

    def save(self, file_path, overwrite):
        if hasattr(self.model, "booster_"):
            self.model.booster_.save_model(file_path)

    def fit_generator(self, *args, **kwargs):
        # callbacks = kwargs["callbacks"]
        #
        # file_path = None
        # early_stopping_rounds = None
        #
        # for item in callbacks:
        #     if hasattr(item, "filepath"):
        #         file_path = item.filepath
        #
        # generator_train = args[0]
        # generator_test = kwargs["validation_data"]
        #
        # generator_test.batchSize = len(generator_test.indexes)
        #
        # train_x, train_y = self.convert_data(generator_train)
        # val_x, val_y = self.convert_data(generator_test)
        #
        # self.model.n_estimators = kwargs["epochs"]
        #
        # checkpoint_cb = None
        #
        # for item in callbacks:
        #     item.set_model(self)
        #     item.on_train_begin()
        #
        #     if "ModelCheckpoint" in str(item):
        #         checkpoint_cb = item
        #
        # def custom_metric(y_true, y_pred):
        #     true, pred = self.to_tf(y_true, y_pred)
        #
        #     results = self.eval_metrics(true, pred, tf.get_default_session())
        #
        #     for item in list(results.keys()):
        #         results["val_" + item] = results[item]
        #
        #     self.rgetter.__dict__ = results
        #
        #     return checkpoint_cb.monitor, np.mean(results[checkpoint_cb.monitor]), "great" in str(checkpoint_cb.monitor_op)
        #
        # def custom_callback(*args, **kwargs):
        #     iter = args[0][2]
        #
        #     self.model._Booster = args[0][0]
        #
        #     for item in callbacks:
        #         if "ReduceLROnPlateau" in str(item):
        #             continue
        #         item.on_epoch_end(iter, self.rgetter)
        #
        # if self.custom_loss_callable:
        #     self.model.objective = self.custom_loss_callable
        #     self.model._objective = self.custom_loss_callable
        #
        # self.model.fit(train_x, train_y, eval_set=[(val_x, val_y)], callbacks = [custom_callback], eval_metric = custom_metric)
        #
        # for item in callbacks:
        #     item.on_train_end()

        train_dataset = get_dataset(self.cfg.data.train)
        cfg = self.cfg
        if cfg.checkpoint_config is not None:
            # save mmdet version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__,
                config=cfg.text,
                CLASSES=train_dataset.CLASSES)
        logger = get_root_logger(self.cfg.log_level)
        self.model.CLASSES = train_dataset.CLASSES
        train_detector(
            self.model,
            train_dataset,
            self.cfg,
            distributed=False, #distributed,
            validate=True, #args_validate,
            logger=logger)

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
        return DetectionStage(x, self)

    def  __init__(self,**atrs):
        self.configPath = None
        self.weightsPath = None
        self.nativeConfig = None
        super().__init__(**atrs)
        self.dataset_clazz = datasets.ImageKFoldedDataSet
        self.flipPred=True

    def initNativeConfig(self):


        atrs = self.all
        self.nativeConfig = Config.fromfile(self.getNativeConfigPath())
        cfg = self.nativeConfig
        cfg.gpus = 1

        wd = os.path.dirname(self.path)
        cfg.work_dir = wd

        if 'bbox_head' in cfg.model and hasattr(atrs,'classes'):
            setCfgAttr(cfg.model.bbox_head, 'num_classes', atrs['classes'])

        if 'mask_head' in cfg.model and hasattr(atrs,'classes'):
            setCfgAttr(cfg.model.mask_head, 'num_classes', atrs['classes'])

        cfg.resume_from = self.getWeightsPath()
        cfg.model.pretrained = self.getWeightsPath()
        cfg.total_epochs = 15  # need to have more epoch then the checkpoint has been generated for
        cfg.data.imgs_per_gpu = max(1, self.batch // cfg.gpus)# batch size
        cfg.data.workers_per_gpu = 1
        cfg.log_config.interval = 1

        # # set cudnn_benchmark
        # if cfg.get('cudnn_benchmark', False):
        #     torch.backends.cudnn.benchmark = True
        # # update configs according to CLI args
        #
        # if args_resume_from is not None:
        #     cfg.resume_from = args_resume_from
        #

    def __setattr__(self, key, value):
        super().__setattr__(key,value)
        if key == 'path' and value is not None:
            self.initNativeConfig()

    def getWeightsPath(self):
        wd = os.path.dirname(self.path)
        joined = os.path.join(wd, self.weightsPath)
        result = os.path.normpath(joined)
        return result

    def getWeightsOutPath(self):
        wd = os.path.dirname(self.path)
        joined = os.path.join(wd, 'weights')
        result = os.path.normpath(joined)
        return result

    def getNativeConfigPath(self):
        wd = os.path.dirname(self.path)
        joined = os.path.join(wd, self.configPath)
        result = os.path.normpath(joined)
        return result

    def update(self,z,res):
        z.segmentation_maps_aug = [imgaug.SegmentationMapOnImage(x, x.shape) for x in res];
        pass

    def createNet(self):
        # ac = self.all["activation"]
        # if ac == "none":
        #     ac = None
        #
        # self.all["activation"]=ac
        # if self.architecture in custom_models:
        #     clazz=custom_models[self.architecture]
        # else: clazz = getattr(segmentation_models, self.architecture)
        # t: configloader.Type = configloader.loaded['segmentation'].catalog['PipelineConfig']
        # r = t.customProperties()
        # cleaned = {}
        # sig=inspect.signature(clazz)
        # for arg in self.all:
        #     pynama = t.alias(arg)
        #     if not arg in r and pynama in sig.parameters:
        #         cleaned[pynama] = self.all[arg]
        #
        # self.clean(cleaned)
        #
        #
        # if self.crops is not None:
        #     cleaned["input_shape"]=(cleaned["input_shape"][0]//self.crops,cleaned["input_shape"][1]//self.crops,cleaned["input_shape"][2])
        #
        # if cleaned["input_shape"][2]>3 and self.encoder_weights!=None and len(self.encoder_weights)>0:
        #     if os.path.exists(self.path + ".mdl-nchannel"):
        #         cleaned["encoder_weights"] = None
        #         model = clazz(**cleaned)
        #         model.load_weights(self.path + ".mdl-nchannel")
        #         return  model
        #
        #     copy=cleaned.copy();
        #     copy["input_shape"] = (cleaned["input_shape"][0] , cleaned["input_shape"][1] , 3)
        #     model1=clazz(**copy);
        #     cleaned["encoder_weights"]=None
        #     model=clazz(**cleaned)
        #     self.adaptNet(model,model1,self.copyWeights);
        #     model.save_weights(self.path + ".mdl-nchannel")
        #     return model
        result = MMDetWrapper(self.nativeConfig, self.getWeightsPath())

        return result


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
        result = CompressibleWriteableDS(ds, resName, path, len(ds))
        return result

    def create_writeable_dataset(self, dataset:DataSet, dsPath:str)->WriteableDataSet:
        resName = (dataset.name if hasattr(dataset, "name") else "") + "_predictions"
        result = CompressibleWriteableDS(dataset, resName, dsPath)
        return result

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

    # def predict_on_dataset(self, dataset, fold=0, stage=0, limit=-1, batch_size=None, ttflips=False, cacheModel=False):
    #     if self.testTimeAugmentation is not None:
    #         ttflips = self.testTimeAugmentation
    #     if batch_size is None:
    #         batch_size = self.inference_batch
    #
    #     if cacheModel:
    #         if self.mdl is None:
    #             self.mdl = self.load_model(fold, stage)
    #         mdl = self.mdl
    #     else:
    #         mdl = self.load_model(fold, stage)
    #
    #     if self.crops is not None:
    #         mdl = BatchCrop(self.crops, mdl)
    #     ta = self.transformAugmentor()
    #     for original_batch in datasets.batch_generator(dataset, batch_size, limit):
    #         for batch in ta.augment_batches([original_batch]):
    #             res = self.predict_on_batch(mdl, ttflips, batch)
    #             resList = [x for x in res]
    #             for ind in range(len(resList)):
    #                 img = resList[ind]
    #                 # FIXME
    #                 unaug = original_batch.images[ind]
    #                 if not self.manualResize and self.flipPred:
    #                     restored = imgaug.imresize_single_image(img, (unaug.shape[0], unaug.shape[1]), cv2.INTER_AREA)
    #                 else:
    #                     restored = img
    #                 resList[ind] = restored
    #             self.update(batch, resList)
    #             batch.results = resList
    #             yield batch

    def predict_on_batch(self, mdl, ttflips, batch):
        o1 = np.array(batch.images_aug)
        res = mdl.predict(o1)
        if ttflips == "Horizontal":
            another = imgaug.augmenters.Fliplr(1.0).augment_images(batch.images_aug)
            res1 = mdl.predict(np.array(another))
            if self.flipPred:
                res1 = imgaug.augmenters.Fliplr(1.0).augment_images(res1)
            res = (res + res1) / 2.0
        elif ttflips:
            res = self.predict_with_all_augs(mdl, ttflips, batch)
        return res

    def update(self,z,res):
        z.segmentation_maps_aug = [imgaug.SegmentationMapOnImage(x, x.shape) for x in res];
        pass




#
#
# def parse(path) -> PipelineConfig:
#     cfg = configloader.parse("segmentation", path)
#     cfg.path = path
#     return cfg

class DetectionStage(generic.Stage):

    def add_visualization_callbacks(self, cb, ec, kf):
        # drawingFunction = ec.drawingFunction
        # if drawingFunction == None:
        #     drawingFunction = datasets.draw_test_batch
        # cb.append(DrawResults(self.cfg, kf, ec.fold, ec.stage, negatives=self.negatives, drawingFunction=drawingFunction))
        # if self.cfg.showDataExamples:
        #     cb.append(DrawResults(self.cfg, kf, ec.fold, ec.stage, negatives=self.negatives, train=True, drawingFunction=drawingFunction))
        print("add_visualization_callbacks")

    def unfreeze(self, model):
        set_trainable(model)

    def _doTrain(self, kf, model, ec, cb, kepoch):
        torch.cuda.set_device(0)
        negatives = self.negatives
        fold = ec.fold
        numEpochs = self.epochs
        callbacks = cb
        subsample = ec.subsample
        validation_negatives = self.validation_negatives
        verbose = self.cfg.verbose
        initial_epoch = kepoch

        if validation_negatives == None:
            validation_negatives = negatives

        train_indexes = kf.sampledIndexes(fold, True, negatives)
        test_indexes = kf.sampledIndexes(fold, False, validation_negatives)

        train_indexes = train_indexes[:0]
        test_indexes = test_indexes[:10]

        trainDS = SubDataSet(kf.ds,train_indexes)
        valDS = SubDataSet(kf.ds, test_indexes)

        v_steps = len(test_indexes) // (round(subsample * kf.batchSize))

        if v_steps < 1: v_steps = 1

        iterations = len(train_indexes) // (round(subsample * kf.batchSize))
        if kf.maxEpochSize is not None:
            iterations = min(iterations, kf.maxEpochSize)

        cfg = model.cfg
        train_dataset = MyDataSet(ds=trainDS, **cfg.data.train)
        train_dataset.CLASSES = CocoDataset.CLASSES
        val_dataset = MyDataSet(ds=valDS, **cfg.data.val)
        val_dataset.CLASSES = CocoDataset.CLASSES
        val_dataset.test_mode = True
        cfg.data.val = val_dataset
        if cfg.checkpoint_config is not None:
            # save mmdet version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__,
                config=cfg.text,
                CLASSES=train_dataset.CLASSES)
            cfg.checkpoint_config.out_dir = self.cfg.getWeightsOutPath()
        logger = get_root_logger(cfg.log_level)
        model.model.CLASSES = train_dataset.CLASSES

        distributed = False
        # prepare data loaders
        data_loaders = [
            build_dataloader(
                train_dataset,
                cfg.data.imgs_per_gpu,
                cfg.data.workers_per_gpu,
                num_gpus=cfg.gpus,
                dist=distributed)
        ]

        runner = train_detector(
            model.model,
            train_dataset,
            cfg,
            distributed=distributed,  # distributed,
            validate=True,  # args_validate,
            logger=logger)

        dsh = DrawSamplesHook(val_dataset, list(range(min(len(test_indexes),10))), os.path.join(os.path.dirname(self.cfg.path),"examples"))
        runner.register_hook(HookWrapper(dsh, toSingleGPUModeBefore, toSingleGPUModeAfter))
        runner.run(data_loaders, cfg.workflow, cfg.total_epochs)

class MusketPredictionItemWrapper(object):

    def __init__(self, ind: int, ds: DataSet):
        self.ind = ind
        self.ds = ds
        self.callbacks:[Callable[[PredictionItem],None]] = []

    def getPredictionItem(self)->PredictionItem:
        predictionItem = self.ds[self.ind]
        for x in self.callbacks:
            x(predictionItem)
        return predictionItem

    def addCallback(self,cb:Callable[[PredictionItem],None]):
        self.callbacks.append(cb)

class MusketInfo(object):

    def __init__(self, predictionItemWrapper:MusketPredictionItemWrapper):
        self.initialized = False
        self.predictionItemWrapper = predictionItemWrapper
        self.predictionItemWrapper.addCallback(self.initializer)

    def checkInit(self):
        if not self.initialized:
            self.getPredictionItem()

    def getPredictionItem(self) -> PredictionItem:
        result = self.predictionItemWrapper.getPredictionItem()
        self.initialized = True
        return result

    def initializer(self, pi:PredictionItem):
        raise ValueError("Not implemented")

class MusketImageInfo(MusketInfo):

    def __init__(self, piw:MusketPredictionItemWrapper):
        super().__init__(piw)
        self.ann = MusketAnnotationInfo(piw)

    def image(self)->np.ndarray:
        pi = self.getPredictionItem()
        img = pi.x[0]
        return img

    def __getitem__(self, key):
        if key == "height":
            self.checkInit()
            return self.height
        elif key == "width":
            self.checkInit()
            return self.width
        elif key == "ann":
            return self.ann
        return None

    def initializer(self, pi: PredictionItem):
        img = pi.x[0]
        self.width = img.shape[1]
        self.height = img.shape[0]

class MusketAnnotationInfo(MusketInfo):

    def initializer(self, pi: PredictionItem):
        y = pi.y
        self.labels = y[0]
        self.bboxes = y[1]
        self.bboxes_ignore = np.zeros(shape=(0,4),dtype=np.float32)
        self.labels_ignore = np.zeros((0),dtype=np.int64)
        self.masks = y[2]

    def __getitem__(self, key):
        if key == "bboxes":
            self.checkInit()
            return self.bboxes
        elif key == "labels":
            self.checkInit()
            return self.labels
        elif key == "bboxes_ignore":
            self.checkInit()
            return self.bboxes_ignore
        elif key == 'labels_ignore':
            self.checkInit()
            return self.labels_ignore
        elif key == "masks":
            self.checkInit()
            return self.masks
        return None

class MyDataSet(CustomDataset):
    
    def __init__(self, ds:DataSet, **kwargs):
        self.ds = ds
        args = kwargs.copy()
        args.pop('type')
        self.type = 'VOCDataset'
        self.img_infos = []
        for idx in range(len(ds)):
            piw = MusketPredictionItemWrapper(idx, self.ds)
            img_info = MusketImageInfo(piw)
            self.img_infos.append(img_info)
        super().__init__(**args)

        self.with_crowd = True

    def __len__(self):
        return len(self.ds)

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def load_annotations(self, ann_file):
        img_infos = []
        for idx in range(len(self.ds)):
            piw = MusketPredictionItemWrapper(idx, self.ds)
            img_info = MusketImageInfo(piw)
            img_infos.append(img_info)
        return img_infos

    def _filter_imgs(self, min_size=32):
        print("filter_images")
        return list(range(len(self)))

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        # load image
        img = img_info.image() #mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes,
                                                       gt_labels)

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()
        gt_seg = None
        if self.with_seg:
            # gt_seg = mmcv.imread(
            #     osp.join(self.seg_prefix, img_info['file_name'].replace(
            #         'jpg', 'png')),
            #     flag='unchanged')
            # gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
            # gt_seg = mmcv.imrescale(
            #     gt_seg, self.seg_scale_factor, interpolation='nearest')
            # gt_seg = gt_seg[None, ...]
            pass
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            gt_masks = self.mask_transform(ann['masks'], pad_shape,
                                           scale_factor, flip)

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)))
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
        if self.with_seg:
            data['gt_semantic_seg'] = DC(to_tensor(gt_seg), stack=True)
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        img = img_info.image() #mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        if self.proposals is not None:
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None

        def prepare_single(img, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, scale, False, proposal)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data

    def show(self, img, result):
        show_result(img,result,self.CLASSES)


def getBB(mask):
    a = np.where(mask != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    minX = max(0, bbox[0] - 10)
    maxX = min(mask.shape[0], bbox[1] + 1 + 10)
    minY = max(0, bbox[2] - 10)
    maxY = min(mask.shape[1], bbox[3] + 1 + 10)
    return np.array([maxX, maxY, minX, minY])


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   logger=None)->Runner:
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        return _dist_train_runner(model, dataset, cfg, validate=validate)
    else:
        return _non_dist_train_runner(model, dataset, cfg, validate=validate)

def _dist_train_runner(model, dataset, cfg, validate=False)->Runner:

    # put model on gpus
    model = MMDistributedDataParallel(model.cuda())

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(model, batch_processor, optimizer, cfg.work_dir,
                    cfg.log_level)

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config,
                                             **fp16_cfg)
    else:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())
    # register eval hooks
    if validate:
        val_dataset_cfg = cfg.data.val
        eval_cfg = cfg.get('evaluation', {})
        if isinstance(model.module, RPN):
            # TODO: implement recall hooks for other datasets
            runner.register_hook(
                CocoDistEvalRecallHook(val_dataset_cfg, **eval_cfg))
        else:
            dataset_type = getattr(mmdetDatasets, val_dataset_cfg.type)
            if issubclass(dataset_type, mmdetDatasets.CocoDataset):
                runner.register_hook(
                    CocoDistEvalmAPHook(val_dataset_cfg, **eval_cfg))
            else:
                runner.register_hook(
                    DistEvalmAPHook(val_dataset_cfg, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    return runner


def toSingleGPUModeBefore(runner):
    result = {'device_ids': runner.model.device_ids}
    runner.model.device_ids = [torch.cuda.current_device()]
    return result

def toSingleGPUModeAfter(runner, beforeRes):
    runner.model.device_ids = beforeRes['device_ids']

def _non_dist_train_runner(model, dataset, cfg, validate=False)->Runner:

    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(model, batch_processor, optimizer, cfg.work_dir,
                    cfg.log_level)
    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=False)
    else:
        optimizer_config = cfg.optimizer_config

    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    before = toSingleGPUModeBefore
    after = toSingleGPUModeAfter

    # register eval hooks
    if validate:
        val_dataset_cfg = cfg.data.val
        eval_cfg = cfg.get('evaluation', {})
        if isinstance(model.module, RPN):
            # TODO: implement recall hooks for other datasets
            runner.register_hook(
                HookWrapper(CocoDistEvalRecallHook(val_dataset_cfg, **eval_cfg),before,after))
        else:
            dataset_type = getattr(mmdetDatasets, val_dataset_cfg.type)
            if issubclass(dataset_type, mmdetDatasets.CocoDataset):
                runner.register_hook(
                    HookWrapper(CocoDistEvalmAPHook(val_dataset_cfg, **eval_cfg),before,after))
            else:
                runner.register_hook(
                    HookWrapper(DistEvalmAPHook(val_dataset_cfg, **eval_cfg),before,after))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    return runner


class DrawSamplesHook(Hook):

    def __init__(self, dataset, indices, dstFolder):
        self.dataset = dataset
        self.indices = indices
        self.dstFolder = dstFolder
        self.score_thr = 0.3
        self.exampleWidth = 800

    def after_train_epoch(self, runner:Runner):

        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))
        for idx in self.indices:
            pi = self.dataset.ds[idx]
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            # compute output
            with torch.no_grad():
                result = runner.model(
                    return_loss=False, rescale=True, **data_gpu)
            results[idx] = (result, pi)

            #batch_size = runner.world_size
            if runner.rank == 0:
                prog_bar.update()


        gtImages = []
        predImages = []
        gtMaskedImages=[]
        predMaskedImages=[]

        classNames = self.dataset.CLASSES
        for r in results:
            imgOrig = r[1].x[0]
            scale = self.exampleWidth / imgOrig.shape[1]
            newY = self.exampleWidth
            newX = int(imgOrig.shape[0] * scale)
            img = imgaug.imresize_single_image(imgOrig,(newX, newY), 'cubic')

            gtLabels = r[1].y[0]
            gtBboxesRaw = r[1].y[1]
            gtBboxes = np.zeros((gtBboxesRaw.shape[0],5),dtype=np.float)
            gtBboxes[:,:4] = gtBboxesRaw * scale
            gtBboxes[:,4] = 1
            result = r[0]

            if isinstance(result, tuple):
                bbox_result, segm_result = result
            else:
                bbox_result, segm_result = result, None
            bboxes = np.vstack(bbox_result) * scale
            if segm_result is not None:
                masksShape = list(imgOrig.shape)
                masksShape[2] = len(self.dataset.CLASSES)
                gtMasks = r[1].y[2]
                maskIndices = set()
                if gtMasks is not None:
                    gtMasksArr = np.zeros(masksShape, dtype=np.float32)
                    for i in range(len(gtLabels)):
                        l = gtLabels[i]
                        gtm = gtMasks[i]
                        gtMasksArr[:,:,l] = gtm
                        maskIndices.add(l)

                predMasksArr = np.zeros(masksShape, dtype=np.float32)
                for i in range(len(gtLabels)):
                    pm = segm_result[i]
                    if len(pm) > 0:
                        predMasksDecoded = [mask_util.decode(x) for x in pm]
                        for x in predMasksDecoded:
                            predMasksArr[:,:,i] += x.astype(np.float32)
                        maskIndices.add(i)

                maskIndices = np.array(sorted(list(maskIndices)),dtype=np.int)
                gtMasksArr = gtMasksArr[:,:,maskIndices]
                predMasksArr = np.minimum(predMasksArr, 1.0)[:,:,maskIndices]

                gtMaskImg = imgaug.SegmentationMapOnImage(gtMasksArr, imgOrig.shape).draw_on_image(imgOrig)
                predMaskImg = imgaug.SegmentationMapOnImage(predMasksArr, imgOrig.shape).draw_on_image(imgOrig)
                #predMaskImg = imgaug.HeatmapsOnImage(predMasksArr,imgOrig.shape).draw_on_image(imgOrig)
                gtMaskedImages.append(imgaug.imresize_single_image(gtMaskImg, (newX, newY), 'cubic'))
                predMaskedImages.append(imgaug.imresize_single_image(predMaskImg,(newX, newY), 'cubic'))


            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
            predImg = imdraw_det_bboxes(img.copy(),bboxes,labels,class_names=classNames,score_thr=self.score_thr)
            gtImg = imdraw_det_bboxes(img.copy(), gtBboxes, gtLabels, class_names=classNames, score_thr=self.score_thr)
            gtImages.append(gtImg)
            predImages.append(predImg)

        gtImg = np.concatenate(gtImages,axis=0)
        predImg = np.concatenate(predImages, axis=0)
        exampleImg = np.concatenate([gtImg, predImg], axis=1)

        if len(gtMaskedImages) > 0:
            gtMaskImg = np.concatenate(gtMaskedImages, axis=0)
            exampleImg = np.concatenate([exampleImg, gtMaskImg], axis=1)

        if len(predMaskedImages) > 0:
            predMaskImg = np.concatenate(predMaskedImages, axis=0)
            exampleImg = np.concatenate([exampleImg, predMaskImg], axis=1)

        epoch = runner.epoch
        out_file = os.path.join(self.dstFolder,f"{epoch}.jpg")
        imwrite(exampleImg, out_file)



def imdraw_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      thickness=1,
                      font_scale=0.5,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = imread(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        if len(bbox) > 4:
            label_text += '|{:.02f}'.format(bbox[-1])
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

    return img

def setCfgAttr(obj, attrName, value):
    if isinstance(obj, list):
        for x in obj:
            setCfgAttr(x,attrName,value)
    else:
        setattr(obj,attrName,value)


class HookWrapper(Hook):

    def __init__(self, hook:Hook, before, after):
        self.hook = hook
        self.before = before
        self.after = after

    def before_run(self, runner):
        beforeRes = self.before(runner)
        self.hook.before_run(runner)
        self.after(runner, beforeRes)

    def after_run(self, runner):
        beforeRes = self.before(runner)
        self.hook.after_run(runner)
        self.after(runner, beforeRes)

    def before_epoch(self, runner):
        beforeRes = self.before(runner)
        self.hook.before_epoch(runner)
        self.after(runner, beforeRes)

    def after_epoch(self, runner):
        beforeRes = self.before(runner)
        self.hook.after_epoch(runner)
        self.after(runner, beforeRes)

    def before_iter(self, runner):
        beforeRes = self.before(runner)
        self.hook.before_iter(runner)
        self.after(runner, beforeRes)

    def after_iter(self, runner):
        beforeRes = self.before(runner)
        self.hook.after_iter(runner)
        self.after(runner, beforeRes)

    def before_train_epoch(self, runner):
        beforeRes = self.before(runner)
        self.hook.before_train_epoch(runner)
        self.after(runner, beforeRes)

    def before_val_epoch(self, runner):
        beforeRes = self.before(runner)
        self.hook.before_val_epoch(runner)
        self.after(runner, beforeRes)

    def after_train_epoch(self, runner):
        beforeRes = self.before(runner)
        self.hook.after_train_epoch(runner)
        self.after(runner, beforeRes)

    def after_val_epoch(self, runner):
        beforeRes = self.before(runner)
        self.hook.after_val_epoch(runner)
        self.after(runner, beforeRes)

    def before_train_iter(self, runner):
        beforeRes = self.before(runner)
        self.hook.before_train_iter(runner)
        self.after(runner, beforeRes)

    def before_val_iter(self, runner):
        beforeRes = self.before(runner)
        self.hook.before_val_iter(runner)
        self.after(runner, beforeRes)

    def after_train_iter(self, runner):
        beforeRes = self.before(runner)
        self.hook.after_train_iter(runner)
        self.after(runner, beforeRes)

    def after_val_iter(self, runner):
        beforeRes = self.before(runner)
        self.hook.after_val_iter(runner)
        self.after(runner, beforeRes)

    def every_n_epochs(self, runner, n):
        beforeRes = self.before(runner)
        result = self.hook.every_n_epochs(runner,n)
        self.after(runner, beforeRes)
        return result

    def every_n_inner_iters(self, runner, n):
        beforeRes = self.before(runner)
        result = self.hook.every_n_inner_iters(runner,n)
        self.after(runner, beforeRes)
        return result

    def every_n_iters(self, runner, n):
        beforeRes = self.before(runner)
        result = self.hook.every_n_iters(runner,n)
        self.after(runner, beforeRes)
        return result

    def end_of_epoch(self, runner):
        beforeRes = self.before(runner)
        result = self.hook.end_of_epoch(runner)
        self.after(runner, beforeRes)
        return result