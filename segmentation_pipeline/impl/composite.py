import k
def plus(s1,f1,s2,f2):
    def impl(x,y):
        return s1*f1(x,y)+s2*f2(x,y)
    return impl

def ps(s:str):
    res=s.split("+")
    ops=[]
    for arg in res:
        elm=arg.split("*")

        scales=1.0
        func=None
        for v in elm:
            if v[0].isalpha():
                if func is not None:
                    raise ValueError("Only one member per component")
                func=v
            if v[0].isdecimal():
                scales=scales*float(v)
        ops.append((scales,func))

    def loss(x,y):
        fr=None
        for v in ops:
            val=v[1](x,y)*v[0]
            if fr:
                fr=val
            else:
                fr=fr+val
        return fr
    return loss

print(ps("a*2.2+b"))