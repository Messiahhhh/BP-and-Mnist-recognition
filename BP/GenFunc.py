import numpy as np
from . import ANN
import random
import math
def sinx(low,high):
    hori={}
    for i in range(500):
        a = random.uniform(low, high)
        hori[a] = math.sin(a*math.pi)
    a = sorted(hori.items(), key=lambda x: x[1])
    hori = []
    verti = []
    x=[]
    for key in a:
        hori.append([[key[0]]])
        x.append(key[0])
        verti.append(key[1])
    return hori,verti,x
def x_2(low,high):
    hori={}
    for i in range(500):
        a = random.uniform(low,high)
        hori[a] = a*a
    a = sorted(hori.items(), key=lambda x: x[1])
    hori = []
    verti = []
    x=[]
    for key in a:
        hori.append([[key[0]]])
        x.append(key[0])
        verti.append(key[1])
    return hori,verti,x
def x1_add_x2(low,high):
    X = []
    Y = []
    Z = []
    hori=[]
    verti=[]
    for i in range(500):
        a = random.uniform(low, high)
        b = random.uniform(low, high)
        y = a + b
        hori.append([[a],[b]])
        verti.append(y)
        X.append(a)
        Y.append(b)
    return hori,verti,X,Y

