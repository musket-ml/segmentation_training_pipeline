
import pandas as pd
from segmentation_pipeline.impl.datasets import PredictionItem
import os
from segmentation_pipeline.impl import rle
import imageio
from segmentation_pipeline import segmentation
import keras.applications as app

from keras.optimizers import Adam
class SegmentationRLE:

    def __init__(self,path,imgPath):
        self.data=pd.read_csv(path);
        self.values=self.data.values;
        self.imgPath=imgPath;
        self.ddd=self.data.groupby('ImageId');
        self.masks=self.ddd['ImageId'];
        self.ids=list(self.ddd.groups.keys())
        pass

    def __getitem__(self, item):
        pixels=self.ddd.get_group(self.ids[item])["EncodedPixels"]
        return PredictionItem(self.ids[item] + str(), imageio.imread(os.path.join(self.imgPath,self.ids[item])),
                              rle.masks_as_image(pixels) > 0.5)


    def get_masks(self,id):
        pixels = self.ddd.get_group(id)["EncodedPixels"]
        return rle.masks_as_images(pixels)

    def isPositive(self, item):
        pixels=self.ddd.get_group(self.ids[item])["EncodedPixels"]
        for mask in pixels:
            if isinstance(mask, str):
                return True;
        return False

    def __len__(self):
        return len(self.masks)

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

    def crop(self,x,y,image):
        z=image.shape[0]//self.parts
        return image[z*x:z*(x+1),z*y:z*(y+1), :]

    def __len__(self):
        return len(self.ds)*self.parts*self.parts


from skimage.morphology import binary_opening, disk
import skimage
import numpy as np
import keras
import matplotlib.pyplot as plt
def main():
    ds = SegmentationRLE("F:/all/train_ship_segmentations.csv", "D:/train_ships/train")
    #segmentation.execute(ds, "ship_config.yaml")
    # cfg=segmentation.parse("fpn/ship_config.yaml")
    # cfg.fit(ds)
    # cfg = segmentation.parse("linknet/ship_config.yaml")
    # cfg.fit(ds)
    # cfg = segmentation.parse("psp/ship_config.yaml")
    # cfg.fit(ds)
    ds0=ds
    cfg = segmentation.parse("fpn_full/ship_config.yaml")
    ds=CropAndSplit(ds,3)

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(ds0[2].x)

    # ax1 = fig.add_subplot(255)
    # ax1.imshow(ds[18].x)
    #
    # ax2 = fig.add_subplot(256)
    # ax2.imshow(ds[19].x)
    #
    # ax3 = fig.add_subplot(254)
    # ax3.imshow(ds[20].x)
    #
    # ax4 = fig.add_subplot(255)
    # ax4.imshow(ds[21].x)
    #
    # ax5 = fig.add_subplot(256)
    # ax5.imshow(ds[22].x)
    #
    # ax5 = fig.add_subplot(257)
    # ax5.imshow(ds[23].x)
    #
    # plt.show()

    cfg.fit(ds,foldsToExecute=[2])

    cfg0 = segmentation.parse("./fpn-resnext2/ship_config.yaml")
    mdl=cfg0.createAndCompileClassifier()
    mdl.load_weights("./fpn-resnext2/classify_weights/best-2.0.weights")
    exists={}
    goodC=0;
    for v in cfg0.predict_on_directory_with_model(mdl,"F:/all/test_v2",ttflips=True):
        for i in range(0,len(v.data)):
            if (v.predictions[i]>0.8):
                goodC=goodC+1;

            exists[v.data[i]]=v.predictions[i]
    print(goodC)
    #cfg0.fit_classifier(ds,2,mdl,12,stage=22)



    #print("A")
    num=0;


    #cfg.predict_to_directory("F:/all/test_v2","F:/all/test_v2_seg",batchSize=16)
    def onPredict(id, img, d):
        exists=d["exists"]
        out_pred_rows = d["pred"]
        if exists[id]<0.8:
            out_pred_rows += [{'ImageId': id, 'EncodedPixels': None}]
            return

        good = d["good"]
        num = d["num"]
        cur_seg = binary_opening(img.arr > 0.53, np.expand_dims(disk(2), -1))
        cur_rles = rle.multi_rle_encode(cur_seg)
        if len(cur_rles) > 0:
            good = good + 1;
            for c_rle in cur_rles:
                out_pred_rows += [{'ImageId': id, 'EncodedPixels': c_rle}]
        else:
            out_pred_rows += [{'ImageId': id, 'EncodedPixels': None}]
        num = num + 1;
        d["good"] = good
        d["num"] = num

        pass
    out_pred_rows=[]
    d = {"pred": out_pred_rows, "good": 0, "num": 0,"exists":exists}
    cfg.predict_in_directory("F:/all/test_v2",2,2,onPredict,d,ttflips=True)
    submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
    submission_df.to_csv('mySubmission.csv', index=False)
    print("Good:"+str(d["good"]))
    print("Num:" + str(d["num"]))

if __name__ == '__main__':
    main()