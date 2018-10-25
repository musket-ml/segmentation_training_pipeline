def warn(*args, **kwargs):
    pass
import warnings
old=warnings.warn
warnings.warn = warn
import sklearn.model_selection as ms
import imgaug
warnings.warn=old
import os
import keras
import random
import imageio
import numpy as np
import traceback

class PredictionItem:
    def __init__(self, path,x,y):
        self.x=x
        self.y=y
        self.id=path;

class DataSetLoader:
    def __init__(self,dataset,indeces,batchSize=16):
        self.dataset = dataset
        self.batchSize = batchSize
        self.indeces = indeces

    def generator(self):
        i = 0;
        bx = []
        by = []
        while True:
            if (i == len(self.indeces)):
                i = 0;
            try:
                item = self.dataset[self.indeces[i]]
                x, y = item.x, item.y
            except:
                traceback.print_exc()
                i = i + 1;
                continue
            i = i + 1;
            bx.append(x)
            by.append(y)
            if len(bx) == self.batchSize:

                yield imgaug.imgaug.Batch(images=bx,
                                              segmentation_maps=[imgaug.SegmentationMapOnImage(x, shape=x.shape) for x
                                                                 in by])
                bx = [];
                by = [];

    def load(self):
        i=0;
        bx=[]
        by=[]
        while True:
            if (i==len(self.indeces)):
                i=0;
            try:
                item=self.dataset[self.indeces[i]]
                x,y=item.x,item.y
            except:
                traceback.print_exc()
                i = i + 1;
                continue
            i=i+1;
            bx.append(x)
            by.append(y)
            if len(bx)==self.batchSize:

                return imgaug.imgaug.Batch(images=bx,segmentation_maps=[imgaug.SegmentationMapOnImage(x,shape=x.shape) for x in  by])
                bx = [];
                by = [];

def drawBatch(batch,path):
    cells = []
    for i in range(0, len(batch.segmentation_maps_aug)):
        cells.append(batch.images_aug[i])
        cells.append(batch.segmentation_maps_aug[i].draw_on_image(batch.images_aug[i]))  # column 2
    # Convert cells to grid image and save.
    grid_image = imgaug.draw_grid(cells, cols=2)
    imageio.imwrite(path, grid_image)

def draw_test_batch(batch,path):
    cells = []
    for i in range(0, len(batch.segmentation_maps_aug)):
        cells.append(batch.images_aug[i])
        cells.append(batch.segmentation_maps_aug[i].draw_on_image(batch.images_aug[i]))  # column 2
        cells.append(batch.heatmaps_aug[i].draw_on_image(batch.images_aug[i])[0])  # column 2
    # Convert cells to grid image and save.
    grid_image = imgaug.draw_grid(cells, cols=3)
    imageio.imwrite(path, grid_image)


class DirectoryDataSet:

    def __init__(self,imgPath,batchSize=32):
        self.imgPath=imgPath;
        self.ids=os.listdir(imgPath)
        self.batchSize=batchSize
        pass

    def __getitem__(self, item):
        return PredictionItem(self.ids[item], imageio.imread(os.path.join(self.imgPath,self.ids[item])),
                              None)

    def __len__(self):
        return len(self.masks)

    def generator(self, maxItems=-1):
        i = 0;
        bx = []
        ps = []
        im=len(self.ids)
        if maxItems!=-1:
            im=min(maxItems,im)
        for v in range(im):

            try:
                item = self[i]
                x, y = item.x, item.id
            except:
                traceback.print_exc()
                i = i + 1
                continue
            i = i + 1
            bx.append(x)
            ps.append(y)
            if len(bx) == self.batchSize:
                yield imgaug.Batch(images=bx,data=ps)
                bx = []
                ps = []
        if len(bx)>0:
            yield imgaug.Batch(images=bx)
        return

class KFoldedDataSet:

    def __init__(self,ds,indexes,aug,transforms,folds=5,rs=33,batchSize=16):
        self.ds=ds;
        if aug==None:
            aug=[]
        if transforms==None:
            transforms=[]
        self.aug=aug;

        self.transforms=transforms
        self.batchSize=batchSize
        self.positive={}
        self.kf=ms.KFold(folds,shuffle=True,random_state=rs);
        self.folds=[v for v in self.kf.split(indexes)]

    def foldIterations(self,foldNum,isTrain=True):
        indexes = self.indexes(foldNum, isTrain)
        return len(indexes)//self.batchSize

    def indexes(self, foldNum, isTrain):
        fold = self.folds[foldNum]
        if isTrain:
            indexes = fold[0]
        else:
            indexes = fold[1]
        return indexes

    def generator(self,foldNum, isTrain=True,negatives="all"):
        indexes = self.sampledIndexes(foldNum, isTrain, negatives)
        yield from self.generator_from_indexes(indexes,isTrain)

    def load(self,foldNum, isTrain=True,negatives="all",ln=16):
        indexes = self.sampledIndexes(foldNum, isTrain, negatives)
        samples = DataSetLoader(self.ds, indexes, ln).load()
        for v in self.augmentor(isTrain).augment_batches([samples]):
            return v

    def generator_from_indexes(self, indexes,isTrain=True):
        m = DataSetLoader(self.ds, indexes, self.batchSize).generator
        aug = self.augmentor(isTrain)
        l = imgaug.imgaug.BatchLoader(m)
        g = imgaug.imgaug.BackgroundAugmenter(l, augseq=aug)

        def r():
            num = 0;
            while True:
                r = g.get_batch();
                x,y= np.array(r.images_aug), np.array([x.arr for x in r.segmentation_maps_aug])
                num=num+1
                yield x,y

        return l,g,r

    def augmentor(self, isTrain)->imgaug.augmenters.Augmenter:
        allAug = [];
        if isTrain:
            allAug = allAug + self.aug
        allAug = allAug + self.transforms
        aug = imgaug.augmenters.Sequential(allAug);
        return aug

    def inner_isPositive(self,x):
        if x in self.positive:
            return self.positive[x]
        self.positive[x]=self.ds.isPositive(x);
        return self.positive[x];

    def sampledIndexes(self, foldNum, isTrain, negatives):
        indexes = self.indexes(foldNum, isTrain)

        if negatives == 'none':
            indexes = [x for x in indexes if self.inner_isPositive(x)]
        if type(negatives)==int:
            sindexes = []
            nindexes = []

            for x in indexes:
                if self.inner_isPositive(x):
                    sindexes.append(x)
                else:
                    nindexes.append(x)

            random.shuffle(nindexes)
            nindexes = nindexes[ 0 : min(len(nindexes),round(len(sindexes)*negatives))]
            r=[]+sindexes+nindexes
            random.shuffle(r)
            return r;
        return indexes

    def trainOnFold(self,fold:int,model:keras.Model,callbacks=[],numEpochs:int=100,negatives="all",subsample=1.0):
        train_indexes = self.sampledIndexes(fold, True, negatives)
        test_indexes = self.sampledIndexes(fold, False, negatives)

        tl,tg,train_g=self.generator_from_indexes(train_indexes)
        vl,vg,test_g = self.generator_from_indexes(test_indexes,isTrain=False)
        try:
            model.fit_generator(train_g(), len(train_indexes)//(round(subsample*self.batchSize)),
                             epochs=numEpochs,
                             validation_data=test_g(),
                             callbacks=callbacks,
                             validation_steps=len(test_indexes)//(round(subsample*self.batchSize)))
        finally:
            tl.terminate();
            tg.terminate();
            vl.terminate();
            vg.terminate();