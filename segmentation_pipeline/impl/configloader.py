import importlib
import os
import yaml


class Module:

    def __init__(self,dict):
        self.catalog={};
        self.entry=None
        for v in dict["types"]:
            t=Type(self, dict["types"][v]);
            if t.entry:
                self.entry=v;
            self.catalog[v]=t
        self.pythonModule=importlib.import_module(dict["(meta.module)"])
        pass

    def instantiate(self,dct):
        if self.entry:
            typeDefinition = self.catalog[self.entry];
            clazz = getattr(self.pythonModule, self.entry)
            args = typeDefinition.constructArgs(dct);
            return clazz(**args)

        if (type(dct)==dict):
            result=[];
            for v in dct:
                clazz=getattr(self.pythonModule, v)

                typeDefinition=self.catalog[v];
                args=typeDefinition.constructArgs(dct[v]);
                if type(args)==dict:
                    result.append(clazz(**args))
                else:
                    result.append(clazz(args))
            return result


class Type:

    def constructArgs(self,dct):
        #for c in dct:
        if type(dct)==str:
            return dct
        return dct

    def alias(self,name:str):
        if name in self.properties:
            p:Property=self.properties[name]
            if p.alias!=None:
                return p.alias
        return name

    def custom(self):
        return [v for v in self.properties if self.properties[v].custom]

    def __init__(self,m:Module,dict):
        self.module=m;
        self.properties={};
        self.entry="(meta.entry)" in dict
        if type(dict)!=str:
            self.type=dict["type"]
            if 'properties' in dict:
                for p in dict['properties']:
                    pOrig=p;
                    if p[-1]=='?':
                        p=p[:-1]
                    self.properties[p]=Property(self,dict['properties'][pOrig])
        else:
            self.type = dict
        pass


class Property:
    def __init__(self,t:Type,dict):
        self.type=t;
        self.alias=None
        self.positional="(meta.positional)" in dict
        self.custom = "(meta.custom)" in dict
        if "(meta.alias)" in dict:
            self.alias=dict["(meta.alias)"]
        pass

loaded={}

def load(name: str)  -> Module:
    if name in loaded:
        return loaded[name]
    pth = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(pth,"..","schemas", name+".raml"), "r") as f:
        cfg = yaml.load(f);
    loaded[name]=Module(cfg);
    return loaded[name]


def parse(name:str,p)->object:
    m=load(name)
    if type(p)==str:
        with open(p, "r") as f:
            return m.instantiate(yaml.load(f));
    return m.instantiate(p);