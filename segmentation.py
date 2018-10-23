import imgaug
import segmentation_models
from segmentation_models.utils import set_trainable
import keras.optimizers as opt
import keras
from impl import datasets, configloader
import os
import yaml
def copy_if_exist(name:str, fr:dict, trg:dict):
    if name in fr:
        trg[name]=fr[name]

def create_with(names:[str], fr:dict):
    res={}
    for v in names:
        copy_if_exist(v, fr, res)
    return res;


class PipelineConfig:
    def __init__(self, **atrs):
        self.batch=8
        self.all = atrs
        self.augmentation=[]
        self.transforms=[]
        self.stages=[]
        self.callbacks=[]
        self.primary_metric="val_binary_accuracy"
        self.primary_metric_mode = "auto"

        for v in atrs:
            val=atrs[v];
            if v=='augmentation':
                val= configloader.parse("augmenters", val)
            if v=='transforms':
                val= configloader.parse("augmenters", val)
            if v=='callbacks':
                val= configloader.parse("callbacks", val)
            if v=='stages':
                val=[Stage(x,self) for x in val]
            setattr(self,v,val)
        pass

    def createNet(self):
        clazz=getattr(segmentation_models,self.architecture)
        t: configloader.Type= configloader.loaded['segmentation'].catalog['PipelineConfig']
        r=t.custom()
        cleaned={}
        for arg in self.all:
            pynama=t.alias(arg);
            if not arg in r:
                cleaned[pynama]=self.all[arg];
        return clazz(**cleaned)

    def createOptimizer(self,lr=None):
        r=getattr(opt,self.optimizer)
        ds=create_with(["lr", "clipnorm", "clipvalue"], self.all);
        if lr:
            ds["lr"]=lr
        return r(**ds)


    def compile(self,net:keras.Model,opt:keras.optimizers.Optimizer,loss=None):
        if loss:
            net.compile(opt, loss, self.metrics)
        else: net.compile(opt,self.loss,self.metrics)
        return net

    def createAndCompile(self,lr=None,loss=None):
        return self.compile(self.createNet(),self.createOptimizer(lr=lr),loss=loss);

    def kfold(self,ds,indeces):
        transforms=[]+self.transforms
        transforms.append(imgaug.augmenters.Scale(size=(self.shape[0], self.shape[1])))


        return datasets.KFoldedDataSet(ds, indeces, self.augmentation, transforms, batchSize=self.batch)


def parse(path)->PipelineConfig:
    return configloader.parse("segmentation", path)

class ExecutionConfig:
    def __init__(self,fold=0,stage=0,subsample=1.0,dr:str=""):
        self.subsample=subsample
        self.stage=stage
        self.fold=fold
        self.dirName=dr
        pass

class Stage:
    def __init__(self,dict,cfg:PipelineConfig):
        self.dict=dict
        self.cfg=cfg;
        if 'negatives' in dict:
            self.negatives=dict['negatives']
        else: self.negatives="real"
        self.epochs=dict["epochs"]
        if 'loss' in dict:
            self.loss=dict['loss']
        else:
            self.loss=None
        if 'lr' in dict:
            self.lr=dict['lr']
        else:
            self.lr=None

    def execute(self, kf: datasets.KFoldedDataSet, model:keras.Model, ec:ExecutionConfig):
        if 'unfreeze_encoder' in self.dict and self.dict['unfreeze_encoder']:
            set_trainable(model)
        if self.loss or self.lr:
            self.cfg.compile(model,self.cfg.createOptimizer(self.lr),self.loss)
        cb=[]+self.cfg.callbacks

        if 'callbacks' in self.dict:
            cb= configloader.parse("callbacks", self.dict['callbacks'])
        cb.append(keras.callbacks.CSVLogger(os.path.join(ec.dirName,"metrics-"+str(ec.fold)+"."+str(ec.stage)+".csv")))
        md=self.cfg.primary_metric_mode
        cb.append(keras.callbacks.ModelCheckpoint(os.path.join(ec.dirName,"best-" + str(ec.fold) + "." + str(ec.stage) + ".weights"),save_best_only=True,monitor=self.cfg.primary_metric,mode=md,verbose=1))
        kf.trainOnFold(ec.fold, model, cb, self.epochs,self.negatives,subsample=ec.subsample)
        pass


def execute(d,path,subsample=1.0,foldsToExecute=None):
    cfg=parse( path);
    dn = os.path.dirname(path)
    if os.path.exists(os.path.join(dn,"summary.yaml")):
        raise ValueError("Experiment is already finished!!!!")
    folds=cfg.kfold(d,range(0,len(d)))
    for i in range(len(folds.folds)):
        if foldsToExecute:
            if not i in foldsToExecute:
                continue
        model = cfg.createAndCompile()
        for s in range(0,len(cfg.stages)):
            st:Stage=cfg.stages[s]
            ec = ExecutionConfig(fold=i,stage=s,subsample=subsample,dr=os.path.dirname(path))
            st.execute(folds,model,ec)

    with open(os.path.join(dn,"summary.yaml"),"w") as f:
        yaml.dump({"completed": True,"cfgName":os.path.basename(path),"subsample":subsample,"folds":foldsToExecute},f)
