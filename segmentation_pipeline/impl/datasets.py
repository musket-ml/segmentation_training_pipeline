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
import imageio
import numpy as np
import traceback
import random
import cv2 as cv


class PredictionItem:
    def __init__(self, path,x,y):
        self.x=x
        self.y=y
        self.id=path;

class DataSetLoader:
    def __init__(self,dataset,indeces,batchSize=16,isTrain=True):
        self.dataset = dataset
        self.batchSize = batchSize
        self.indeces = indeces
        self.isTrain=isTrain

    def generator(self):
        i = 0;
        bx = []
        by = []
        ids= []
        while True:
            if i == len(self.indeces):
                i = 0
            try:
                id, x, y = self.proceed(i)
            except:
                traceback.print_exc()
                i = i + 1
                continue
            i = i + 1
            ids.append(id)
            bx.append(x)
            by.append(y)
            if len(bx) == self.batchSize:
                yield self.createBatch(bx, by, ids)
                bx = []
                by = []
                ids= []

    def createBatch(self, bx, by, ids):
        if len(by[0].shape)>1:
            return imgaug.imgaug.Batch(data=ids, images=bx,
                                       segmentation_maps=[imgaug.SegmentationMapOnImage(x, shape=x.shape) for x
                                                          in by])
        else:
            r=imgaug.imgaug.Batch(data=[ids,by], images=bx)
            return r

    def load(self):
        i=0;
        bx=[]
        by=[]
        ids = []
        while True:
            if (i==len(self.indeces)):
                i=0;
            try:
                id, x, y = self.proceed(i)
            except:
                traceback.print_exc()
                i = i + 1;
                continue
            ids.append(id)
            bx.append(x)
            by.append(y)
            i=i+1;

            if len(bx)==self.batchSize:
                return self.createBatch(bx,by,ids)
                bx = [];
                by = [];

    def proceed(self, i):
        id = ""
        if hasattr(self.dataset, "item"):
            item = self.dataset.item(self.indeces[i], self.isTrain)
        else:
            item = self.dataset[self.indeces[i]]
        x, y = item.x, item.y
        if isinstance(item, PredictionItem):
            id = item.id
        return id, x, y


def drawBatch(batch,path):
    cells = []
    nc=2
    if not hasattr(batch, "segmentation_maps_aug") or batch.segmentation_maps_aug is None:
        batch.segmentation_maps_aug=batch.predicted_maps_aug
    if not hasattr(batch, "images_aug") or batch.images_aug is None:
        batch.images_aug=batch.images
        batch.segmentation_maps_aug=batch.predicted_maps_aug
    for i in range(0, len(batch.segmentation_maps_aug)):
        cells.append(batch.images_aug[i])
        if hasattr(batch,"predicted_maps_aug"):
            cells.append(batch.segmentation_maps[i].draw_on_image(batch.images_aug[i]))  # column 2
            nc=3
        cells.append(batch.segmentation_maps_aug[i].draw_on_image(batch.images_aug[i]))  # column 2
    # Convert cells to grid image and save.
    grid_image = imgaug.draw_grid(cells, cols=nc)
    imageio.imwrite(path, grid_image)

def draw_test_batch(batch,path):
    cells = []
    for i in range(0, len(batch.segmentation_maps_aug)):
        cells.append(batch.images_aug[i][:,:,0:3])
        cells.append(batch.segmentation_maps_aug[i].draw_on_image(batch.images_aug[i][:,:,0:3]))  # column 2
        cells.append(batch.heatmaps_aug[i].draw_on_image(batch.images_aug[i][:,:,0:3])[0])  # column 2
    # Convert cells to grid image and save.
    grid_image = imgaug.draw_grid(cells, cols=3)
    imageio.imwrite(path, grid_image)

class ConstrainedDirectory:
    def __init__(self,path,filters):
        self.path=path;
        self.filters=filters

    def __repr__(self):
        return self.path+" (with filter)"


class CompositeDataSet:

    def __init__(self,components):
        self.components=components
        sum=0;
        shifts=[]
        for i in components:
            sum=sum+len(i)
            shifts.append(sum)
        self.shifts=shifts
        self.len=sum

    def item(self, item, isTrain):
        i = item
        for j in range(len(self.shifts)):
            d = self.components[j]
            if i < self.shifts[j]:
                if hasattr(d,"item"):
                    return d.item(i,isTrain)
                return d[i]
            else:
                i = i - self.shifts[j]
        return None

    def __getitem__(self, item):
        i=item
        for j in range(len(self.shifts)):
            d=self.components[j]
            if i<self.shifts[j]:
                return d[i]
            else: i=i-self.shifts[j]
        return None

    def isPositive(self, item):
        i = item
        for j in range(len(self.shifts)):
            d = self.components[j]
            if i < self.shifts[j]:
                return d.isPositive(i)
            else:
                i = i - self.shifts[j]
        return False

    def __len__(self):
        return self.len

class DirectoryDataSet:

    def __init__(self,imgPath,batchSize=32):

        if isinstance(imgPath,ConstrainedDirectory):
            self.imgPath=imgPath.path
            self.ids = imgPath.filters
        else:
            self.imgPath = imgPath;
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
            yield imgaug.Batch(images=bx,data=ps)
        return
class Backgrounds:
    def __init__(self,path,erosion=0,augmenters:imgaug.augmenters.Augmenter=None):
        self.path=path;
        self.rate=0.5
        self.augs=augmenters
        self.erosion=erosion
        self.options=[os.path.join(path,x) for x in os.listdir(self.path)]

    def next(self,i,i2):
        fl=random.choice(self.options)
        im=imageio.imread(fl)
        r=cv.resize(im,(i.shape[1],i.shape[0]))
        if isinstance(self.erosion,list):
            er=random.randint(self.erosion[0],self.erosion[1])
            kernel = np.ones((er, er), np.uint8)
            i2 = cv.erode(i2, kernel)
        elif self.erosion>0:
            kernel = np.ones((self.erosion, self.erosion), np.uint8)
            i2=cv.erode(i2,kernel)
        i2=i2!=0
        i2=np.squeeze(i2)
        if i.shape[2]!=3:
           zr=np.copy(i)
           zr[:,:,0:3]=r
           zr[i2] = i[i2]
           return zr
        else:
            r[i2] = i[i2]
        return r;

    def augment_item(self,i):
        if self.augs!=None:

            b=imgaug.Batch(images=[i.x],
                                segmentation_maps=[imgaug.SegmentationMapOnImage(i.y, shape=i.y.shape)])
            for v in self.augs.augment_batches([b]):
                bsa:imgaug.Batch=v
                #print(bsa.images_aug)
                #print(bsa.images_aug[0].shape)
                #print(i.x.shape)
                break
            xa=bsa.images_aug[0]

            xa=cv.resize(xa,(i.x.shape[1],i.x.shape[0]))
            ya=bsa.segmentation_maps_aug[0].arr
            ya = cv.resize(ya, (i.x.shape[1],  i.x.shape[0]))
            r = self.next(xa, ya)
            return PredictionItem(i.id, r, ya>0.5)
        else:
            r=self.next(i.x,i.y)
            return PredictionItem(i.id,r,i.y)

class WithBackgrounds:
    def __init__(self, ds,bg):
        self.ds=ds
        self.bg=bg
        self.rate=bg.rate

    def __len__(self):
        return len(self.ds)

    def item(self,item,isTrain):
        if not isTrain:
            return self.ds[item]

        return self[item]

    def __getitem__(self, item):
        i=self.ds[item]
        if random.random()>self.rate:
            return self.bg.augment_item(i)
        return i
class SimplePNGMaskDataSet:
    def __init__(self, path, mask):
        self.path = path;
        self.mask = mask;
        self.ids=[x[0:x.index('.')] for x in os.listdir(path)]
        pass

    def __getitem__(self, item):
        return PredictionItem(self.ids[item] + str(), imageio.imread(os.path.join(self.path, self.ids[item]+".jpg")),
                              np.expand_dims(imageio.imread(os.path.join(self.mask, self.ids[item] + ".png")),axis=2).astype(np.float32)/255.0)

    def isPositive(self, item):
        return True

    def __len__(self):
        return len(self.ids)

AUGMENTER_QUEUE_LIMIT=10



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

    def addToTrain(self,dataset):
        ma = len(self.ds)
        self.ds = CompositeDataSet([self.ds, dataset])
        nf = []
        for fold in self.folds:
            train = fold[0]
            rrr = np.concatenate([train, np.arange(ma, ma + len(dataset))])
            np.random.shuffle(rrr)
            nf.append((rrr, fold[1]))
        self.folds = nf

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

    def generator(self,foldNum, isTrain=True,negatives="all",returnBatch=False):
        indexes = self.sampledIndexes(foldNum, isTrain, negatives)
        yield from self.generator_from_indexes(indexes,isTrain,returnBatch)

    def load(self,foldNum, isTrain=True,negatives="all",ln=16):
        indexes = self.sampledIndexes(foldNum, isTrain, negatives)
        samples = DataSetLoader(self.ds, indexes, ln,isTrain=isTrain).load()
        for v in self.augmentor(isTrain).augment_batches([samples]):
            return v

    def generator_from_indexes(self, indexes,isTrain=True,returnBatch=False):
        m = DataSetLoader(self.ds, indexes, self.batchSize,isTrain=isTrain).generator
        aug = self.augmentor(isTrain)
        l = imgaug.imgaug.BatchLoader(m)
        g = imgaug.imgaug.BackgroundAugmenter(l, augseq=aug,queue_size=AUGMENTER_QUEUE_LIMIT)

        def r():
            num = 0;
            while True:
                r = g.get_batch();
                x,y= np.array(r.images_aug), np.array([x.arr for x in r.segmentation_maps_aug])
                num=num+1
                if returnBatch:
                    yield x,y,r
                else: yield x,y

        return l,g,r

    def positive_negative_classification_generator_from_indexes(self, indexes, isTrain=True):
        m = DataSetLoader(self.ds, indexes, self.batchSize,isTrain=isTrain).generator
        aug = self.augmentor(isTrain)
        l = imgaug.imgaug.BatchLoader(m)
        g = imgaug.imgaug.BackgroundAugmenter(l, augseq=aug,queue_size=AUGMENTER_QUEUE_LIMIT)
        def r():
            num = 0;
            while True:
                r = g.get_batch();
                x,y= np.array(r.images_aug), np.array([x.arr for x in r.segmentation_maps_aug])
                rs=np.zeros((len(y)))
                for i in range(0,len(y)):
                    if y[i].max()>0.5:
                        rs[i]=1.0
                num=num+1
                yield x,rs
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

            random.shuffle(nindexes,23232)
            nindexes = nindexes[ 0 : min(len(nindexes),round(len(sindexes)*negatives))]
            r=[]+sindexes+nindexes
            random.shuffle(r,232772)
            return r;
        return indexes

    def numBatches(self,fold,negatives,subsample):
        train_indexes = self.sampledIndexes(fold, True, negatives)
        return len(train_indexes)//(round(subsample*self.batchSize))

    def trainOnFold(self,fold:int,model:keras.Model,callbacks=[],numEpochs:int=100,negatives="all",
                    subsample=1.0,validation_negatives=None):
        train_indexes = self.sampledIndexes(fold, True, negatives)
        if validation_negatives==None:
            validation_negatives=negatives
        test_indexes = self.sampledIndexes(fold, False, validation_negatives)

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


class KFoldedDataSetImageClassification(KFoldedDataSet):

    def generator_from_indexes(self, indexes,isTrain=True,returnBatch=False):
        m = DataSetLoader(self.ds, indexes, self.batchSize,isTrain=isTrain).generator
        aug = self.augmentor(isTrain)
        l = imgaug.imgaug.BatchLoader(m)
        g = imgaug.imgaug.BackgroundAugmenter(l, augseq=aug,queue_size=AUGMENTER_QUEUE_LIMIT)

        def r():
            num = 0;
            while True:
                r = g.get_batch();
                x,y= np.array(r.images_aug), np.array([x for x in r.data[1]])
                num=num+1
                if returnBatch:
                    yield x,y,r
                else: yield x,y

        return l,g,r


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
        return PredictionItem(dm.id,x1,y1)

    def crop(self,y,x,image):
        h=image.shape[0]//self.parts
        w = image.shape[1] // self.parts
        return image[h*y:h*(y+1),w*x:w*(x+1), :]

    def __len__(self):
        return len(self.ds)*self.parts*self.parts