import glob
import yaml
import os
import pandas as pd
import warnings
import argparse


def is_simple(t:list):
    for v in t:
        if not type(v) in [int,str,float,bool]: return False
    return True


def flatten(c:dict):
    res={}
    for v in c:
        m=c[v]
        if type(m)==dict:
            r=flatten(m)
            for q in r:
                res[v+"."+q] = r[q]
        if type(m)==list:
           if is_simple(m):
               res[v]=str(m)
           else:
               for i in range(len(m)):
                   val=m[i];
                   z=flatten(val);
                   for key in z:
                    res[v + "." + str(i)+"."+key] =z[key]
           pass
        if type(m) == str or type(m) == int or type(m) == float:
            res[v]=m
            pass

    return res

ignored_metrics=["lr","fold"]
only_use=[]
metrics_to_min=["loss","val_loss"]


def build_metrics(pattern):
    configs=[]
    metrics={}
    all_metrics_keys=set()
    for p in glob.glob(pattern,recursive=True):
        with open(p,"r") as f:
            configs.append((p,yaml.load(f)))
        all_metrics=[]
        for m in glob.glob(os.path.dirname(p)+"/metrics/metrics-*.csv"):
            try:
                df=pd.read_csv(m)
                ev=m[m.rfind('-')+1:]
                fold=int(ev[0:ev.index('.')])
                stage = int(ev[ev.index('.')+1:ev.rfind('.')])
                all_metrics.append(df)
                df["fold"]=fold
                df["stage"]=stage
                all_metrics_keys=all_metrics_keys.union(set(df.keys()))
            except:
                warnings.warn("unreadable or empty metrics:"+m)
                pass
        if len(all_metrics)>0:
            metrics[p]=pd.concat(all_metrics)

    flattened=[]
    all_keys=set()
    for p,cfg in configs:
        fl=flatten(cfg);
        flattened.append((p,fl))
        all_keys=all_keys.union(set(fl.keys()))
    voc={}
    for k in all_keys: voc[k]=set()
    for p,cfg in flattened:
        for k in all_keys:
            val=""
            if k in cfg:
                val=cfg[k]
            voc[k].add(val)
    meaningfullkeys=[]
    for k in all_keys:
        if len(voc[k])>1 : meaningfullkeys.append(k)

    res=[]
    columns=["path","fold_count"]
    if "loss" in meaningfullkeys:
        columns.append("loss_function")
    for k in meaningfullkeys:
        columns.append(k)

    if len(only_use)==0:
        for k in sorted(list(all_metrics_keys)):
            if not k in ignored_metrics:
                if "val_" not in k:
                    columns.append(k)
                pass
        for k in sorted(list(all_metrics_keys)):
            if not k in ignored_metrics:
                if "val_"  in k:
                    columns.append(k)
                pass
    else:
        columns=columns+only_use
    for p,cfg in flattened:
        r={}
        for k in meaningfullkeys:
            val=""
            if k in cfg:
                val=cfg[k]
            r[k]=val
        r["path"]=p;

        if "loss" in r:
            r["loss_function"]=r["loss"]
        if p in metrics:
            exp_metrics = metrics[p]
            res.append(r)
            for k in sorted(list(all_metrics_keys)):
                if len(only_use)>0:
                    if not k in only_use:
                        continue
                if not k in ignored_metrics:
                    count=0;
                    mv=0;
                    for fold in range(exp_metrics['fold'].min(),exp_metrics['fold'].max()+1):
                        count=count+1
                        if k in metrics_to_min:
                            mv =mv+ exp_metrics.query("fold=="+str(fold))[k].min()
                            continue
                        mv=mv+ exp_metrics.query("fold=="+str(fold))[k].max()
                    r[k] = mv/count
            r["fold_count"]=exp_metrics['fold'].max()-exp_metrics['fold'].min()+1
    return res,columns


def main():
    parser = argparse.ArgumentParser(description='Analize experiment metrics.')
    parser.add_argument('--inputFolder',  type=str, default=".",
                        help='folder to search for experiments')
    parser.add_argument('--output', type=str, default="report.csv",
                        help='file to store aggregated metrics')
    parser.add_argument('--onlyMetric', type=str, default="",
                        help='file to store aggregated metrics')
    parser.add_argument('--sortBy', type=str, default="val_loss",
                        help='metric to sort results')

    args = parser.parse_args()
    if len(args.onlyMetric)>0:
        only_use.append(args.onlyMetric)
        args.sortBy=args.onlyMetric
    pattern = args.inputFolder+"/**/*config.yaml";

    rrr, columns = build_metrics(pattern)
    res = pd.DataFrame(rrr)
    res = res[columns]
    res.sort_index = args.sortBy
    res.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
