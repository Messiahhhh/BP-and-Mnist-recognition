from django.shortcuts import render
import json
# Create your views here.
from django.http import HttpResponse
from django.http import JsonResponse
from . import ANN
import random
from . import GenFunc
import numpy as np
lr=""
IP=""
HN=""
NH=""
OA=""
TM=""
kind=""
def index(request):
    return render(request, 'inde.html')
def InitBP(request):
    if request.is_ajax():
        if request.method=="GET":
            global lr
            global IP
            global HN
            global NH
            global OA
            global TM
            global kind
            lr = request.GET.get('lr')#学习率
            lr=float(lr)#学习率
            IP = request.GET.get('inputArea')
            IP=int(IP)#输入神经元数
            HN = request.GET.get('HidelayerNum')
            HN=int(HN)#隐层数
            NH = request.GET.get('NumHidelayer')
            NH=NH.split(",")#隐层各层神经元
            for i in range(0,len(NH)):
                NH[i]=int(NH[i])
            OA = request.GET.get('OutputArea')
            OA=int(OA)#输出神经元数
            TM=request.GET.get("TM")#迭代次数
            TM=int(TM)
            Data={}
            Data['lr']=lr
            Data['IP']=IP
            Data['HN']=HN
            Data['NH']=NH
            Data['OA']=OA
            Data['TM']=TM
            return JsonResponse({'lr':lr,'IP':IP,'HN':HN,'OA':OA,'TM':TM})
    '''loss_x = []
    loss_y = []
    standard_x=[]
    standard_y=[]
    re_y=[]
    hori = {}
    verti = []
    layer=[]
    layer.append(IP)
    for i in range(0,HN):
        layer.append(NH[i])
    layer.append(OA)
    hori, verti, standard_x= GenFunc.sinx()
    X= ANN.ANN(layer, hori, verti, lr, TM)
    loss_x,loss_y=X.run()
    for i in range(0,len(hori)):
        tmp=X.forwar(i)
        re_y.append(tmp[0][0])
    Data={}
    Data["loss_x"]=loss_x
    Data["loss_y"]=loss_y
    Data["re_y"]=re_y
    Data["standard_x"]=standard_x
    Data["lr"]=lr
    Data["HN"]=HN
    Data["NH"]=NH
    Data["OA"]=OA
    Data["TM"]=TM
   {'data': json.dumps(Data)}
    return render(request, 'SinX.html')'''
def sin_x(request):
    hori, verti, standard_x = GenFunc.sinx(-1,1)
    print(hori,verti,standard_x)
    Data={}
    Data['IP']=IP
    Data["lr"]=lr
    Data["HN"]=HN
    Data["NH"]=NH
    Data["OA"]=OA
    Data["TM"]=TM
    Data['hori']=hori
    Data['verti']=verti
    Data['standard_x']=standard_x
    return render(request,'SinX.html',{'data': json.dumps(Data)})
def X_2(request):
    hori, verti, standard_x = GenFunc.x_2(-2, 2)
    print(hori, verti, standard_x)
    Data = {}
    Data['IP'] = IP
    Data["lr"] = lr
    Data["HN"] = HN
    Data["NH"] = NH
    Data["OA"] = OA
    Data["TM"] = TM
    Data['hori'] = hori
    Data['verti'] = verti
    Data['standard_x'] = standard_x
    return render(request, 'X^2.html', {'data': json.dumps(Data)})
def X1_X2(request):
    hori, verti, standard_x,standard_y = GenFunc.x1_add_x2(-2, 2)
    Data = {}
    Data['IP'] = IP
    Data["lr"] = lr
    Data["HN"] = HN
    Data["NH"] = NH
    Data["OA"] = OA
    Data["TM"] = TM
    Data['hori'] = hori
    Data['verti'] = verti#Z值
    Data['standard_x'] = standard_x#X值
    Data['standard_y']=standard_y#Y值
    return render(request, 'X1_X2.html', {'data': json.dumps(Data)})
def train_Sinx(request):
    loss_x = []
    loss_y = []
    standard_x = []
    standard_y = []
    Gene_hori=[]
    Gene_verti=[]
    Gene_x=[]
    Gene_y=[]
    re_y = []
    hori = {}
    verti = []
    layer = []
    layer.append(IP)
    for i in range(0, HN):
        layer.append(NH[i])
    layer.append(OA)
    hori, verti, standard_x = GenFunc.sinx(-1,1)
    Gene_hori, Gene_verti, Gene_x=GenFunc.sinx(1,3)#泛化X样本，泛化结果存入Gene_y
    X = ANN.ANN(layer, hori, verti, lr, TM)
    loss_x, loss_y = X.run()
    for i in range(0, len(hori)):
        tmp = X.forwar(i)
        re_y.append(tmp[0][0])
    X.InitTest(Gene_hori)
    for i in range(0,len(Gene_hori)):
        tmp=X.TestForwar(i)
        Gene_y.append(tmp[0][0])#泛化曲线
    return JsonResponse({'loss_x':loss_x,'loss_y':loss_y,'re_y':re_y,'standard_x':standard_x,'Gene_x':Gene_x,'Gene_y':Gene_y,'Gene_verti':Gene_verti})
def train_X1_X2(request):
    print("pp")
    loss_x = []
    loss_y = []
    standard_x = []#标准输入x
    standard_y = []#标准输入y
    Gene_hori=[]#x,y的
    Gene_verti=[]
    Gene_x=[]
    Gene_y=[]
    Gene_z=[]
    re_y = []
    hori = {}
    verti = []
    layer = []
    layer.append(IP)
    for i in range(0, HN):
        layer.append(NH[i])
    layer.append(OA)
    hori, verti, standard_x, standard_y = GenFunc.x1_add_x2(-2,2)#standard_x,standard_y为x1,x2,z为verti
    Gene_hori, Gene_verti, Gene_x,Gene_z=GenFunc.x1_add_x2(2,6)#泛化X样本，泛化结果存入Gene_y
    print(Gene_z)
    print(Gene_x)
    X = ANN.ANN(layer, hori, verti, lr, TM)
    loss_x, loss_y = X.run()
    for i in range(0, len(hori)):
        tmp = X.forwar(i)
        re_y.append(tmp[0][0])
    X.InitTest(Gene_hori)
    for i in range(0,len(Gene_hori)):
        tmp=X.TestForwar(i)
        Gene_y.append(tmp[0][0])#泛化曲线
    return JsonResponse({'loss_x':loss_x,'loss_y':loss_y,'re_y':re_y,'standard_x':standard_x,'standard_y':standard_y,'Gene_x':Gene_x,'Gene_z':Gene_z,'Gene_y':Gene_y,'Gene_verti':Gene_verti})
def train_X_2(request):
    loss_x = []
    loss_y = []
    standard_x = []
    standard_y = []
    Gene_hori=[]
    Gene_verti=[]
    Gene_x=[]
    Gene_y=[]
    re_y = []
    hori = {}
    verti = []
    layer = []
    layer.append(IP)
    for i in range(0, HN):
        layer.append(NH[i])
    layer.append(OA)
    hori, verti, standard_x = GenFunc.x_2(-2,2)#hori是
    Gene_hori, Gene_verti, Gene_x=GenFunc.x_2(2,4)#泛化X样本，泛化结果存入Gene_y
    X = ANN.ANN(layer, hori, verti, lr, TM)
    loss_x, loss_y = X.run()
    for i in range(0, len(hori)):
        tmp = X.forwar(i)
        re_y.append(tmp[0][0])
    X.InitTest(Gene_hori)
    for i in range(0,len(Gene_hori)):
        tmp=X.TestForwar(i)
        Gene_y.append(tmp[0][0])#泛化曲线
    return JsonResponse({'loss_x':loss_x,'loss_y':loss_y,'re_y':re_y,'standard_x':standard_x,'Gene_x':Gene_x,'Gene_y':Gene_y,'Gene_verti':Gene_verti})