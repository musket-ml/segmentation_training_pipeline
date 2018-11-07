
import pandas as pd
from segmentation_pipeline.impl.datasets import PredictionItem
import os
from segmentation_pipeline.impl import rle
import imageio
from segmentation_pipeline import segmentation

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

ds = SegmentationRLE ("F:/all/train_ship_segmentations.csv","D:/train_ships/train")
from skimage.morphology import binary_opening, disk
import numpy as np
def main():
    #segmentation.execute(ds, "ship_config.yaml")
    # cfg=segmentation.parse("fpn/ship_config.yaml")
    # cfg.fit(ds)
    # cfg = segmentation.parse("linknet/ship_config.yaml")
    # cfg.fit(ds)
    # cfg = segmentation.parse("psp/ship_config.yaml")
    # cfg.fit(ds)
    cfg = segmentation.parse("fpn-resnext2/ship_config.yaml")
    cfg.fit(ds,foldsToExecute=[2],start_from_stage=4)
    #print("A")
    num=0;


    #cfg.predict_to_directory("F:/all/test_v2","F:/all/test_v2_seg",batchSize=16)

    out_pred_rows=[]


    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    def iou(img_true, img_pred):
        i = np.sum((img_true * img_pred) > 0)
        u = np.sum((img_true + img_pred) > 0) + 0.0000000000000000001  # avoid division by zero
        return i / u
    # original version of @raresbarbantan
    def f2_prev(masks_true, masks_pred):
        # a correct prediction on no ships in image would have F2 of zero (according to formula),
        # but should be rewarded as 1
        if np.sum(masks_true) == np.sum(masks_pred) == 0:
            return 1.0

        f2_total = 0
        for t in thresholds:
            tp, fp, fn = 0, 0, 0
            ious = {}
            for i, mt in enumerate(masks_true):
                found_match = False
                for j, mp in enumerate(masks_pred):
                    miou = iou(mt, mp)
                    ious[100 * i + j] = miou  # save for later
                    if miou >= t:
                        found_match = True
                if not found_match:
                    fn += 1

            for j, mp in enumerate(masks_pred):
                found_match = False
                for i, mt in enumerate(masks_true):
                    miou = ious[100 * i + j]
                    if miou >= t:
                        found_match = True
                        break
                if found_match:
                    tp += 1
                else:
                    fp += 1
            f2 = (5 * tp) / (5 * tp + 4 * fn + fp)
            f2_total += f2

        return f2_total / len(thresholds)

    def f2(masks_true, masks_pred):
        if np.sum(masks_true) == 0:
            return float(np.sum(masks_pred) == 0)

        ious = []
        mp_idx_found = []
        for mt in masks_true:
            for mp_idx, mp in enumerate(masks_pred):
                if mp_idx not in mp_idx_found:
                    cur_iou = iou(mt, mp)
                    if cur_iou > 0.5:
                        ious.append(cur_iou)
                        mp_idx_found.append(mp_idx)
                        break
        f2_total = 0
        for th in thresholds:
            tp = sum([iou > th for iou in ious])
            fn = len(masks_true) - tp
            fp = len(masks_pred) - tp
            f2_total += (5 * tp) / (5 * tp + 4 * fn + fp)

        return f2_total / len(thresholds)
    def onPredict(id,img,d):
        out_pred_rows=d["pred"]
        good=d["good"]
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
        d["good"]=good
        d["num"] = num

        pass



    d={"pred": out_pred_rows, "good": 0, "num": 0}
    cfg.predict_in_directory("F:/all/test_v2",2,3,onPredict,d,ttflips=True)
    submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
    submission_df.to_csv('mySubmission.csv', index=False)
    print("Good:"+str(d["good"]))
    print("Num:" + str(d["num"]))

    num=0;
    neg=0;
    total=0;
    folds = cfg.kfold(ds, range(0, len(ds)))
    ind=folds.indexes(2,False)
    neg0=0
    for i in ind:
        if not ds.isPositive(i):
            #ms=ds.get_masks(ds.ids[i])
            #if len(ms)>0:
            neg0=neg0+1
    print(len(ind))
    print(neg0/len(ind))
    total={}
    for d in range(1, 20):
        total[d]=0;
    for b in cfg.evaluateAll(ds,2,None):
        for i in range(len(b.segmentation_maps_aug)):
            masks = ds.get_masks(b.data[i])

            for d in range(1,20):
                cur_seg = binary_opening(b.segmentation_maps_aug[i].arr > d/20, np.expand_dims(disk(2), -1))
                cm = rle.masks_as_images(rle.multi_rle_encode(cur_seg))
                pr = f2(masks, cm);
                total[d]=total[d]+pr

            print(total)



            if (len(masks) > 0):
                pass
                #print(len(cm),len(masks),pr)
            else:
                #print(b.data[i])
                neg=neg+1;
            #total=total+pr;
            num=num+1
            if (len(masks)>0):
                print(total)
        #num=num+len(b.segmentation_maps_aug)

    print(num)
    # folds = cfg.kfold(ds, range(0, len(ds)))
    # vl, vg, test_g = folds.generator(2, False);
    # m = cfg.load_model(2,3)
    # l = 0
    # di = 0;
    # for f in test_g():
    #     x, y = f
    #     z = m.predict(x)
    #     #print(z.shape)
    #     cur_seg = binary_opening(z > 0.5, np.expand_dims(disk(2), -1))
    #
    #     #post_img = remove_small_holes(remove_small_objects(z > 0.5))
    #     l = l + 1
    #     if l == 100:
    #         break
    #     #di = di + f2(y, post_img)
    #     print(di / l)
    # # cfg.predict_in_directory("F:/all/test_v2",[0,2,3],3,onPredict,d,ttflips=True)
    # # submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
    # # submission_df.to_csv('mySubmission.csv', index=False)
    # # print("Good:"+str(d["good"]))
    # # print("Num:" + str(d["num"]))
if __name__ == '__main__':
    main()