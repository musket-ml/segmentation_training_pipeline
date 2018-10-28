
import pandas as pd
from segmentation_pipeline.impl.datasets import PredictionItem
import os
from segmentation_pipeline.impl import rle
import imageio
from segmentation_pipeline import segmentation


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

    def isPositive(self, item):
        pixels=self.ddd.get_group(self.ids[item])["EncodedPixels"]
        for mask in pixels:
            if isinstance(mask, str):
                return True;
        return False

    def __len__(self):
        return len(self.masks)

ds = SegmentationRLE ("F:/all/train_ship_segmentations.csv","D:/train_ships/train")

def main():
    #segmentation.execute(ds, "ship_config.yaml")
    # cfg=segmentation.parse("fpn/ship_config.yaml")
    # cfg.fit(ds)
    # cfg = segmentation.parse("linknet/ship_config.yaml")
    # cfg.fit(ds)
    # cfg = segmentation.parse("psp/ship_config.yaml")
    # cfg.fit(ds)
    cfg = segmentation.parse("fpn-resnext/ship_config.yaml")
    cfg.fit(ds)
    #print("A")
    #num=0;


    #cfg.predict_to_directory("F:/all/test_v2","F:/all/test_v2_seg",batchSize=16)

    # for i in cfg.predict_on_directory("F:/all/test_v2",0,0,100):
    #     drawBatch(i,"batch"+str(num)+'.jpg')
    #     num=num+1
    #     print()

if __name__ == '__main__':
    main()