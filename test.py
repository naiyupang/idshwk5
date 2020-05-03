from sklearn.ensemble import RandomForestClassifier
import numpy as np
from collections import *
import math

def Count_numbers(string):
    c=0
    for character in string:
        if(character>='0' and character<='9'):
            c+=1
    return c

def Calculate_entropy(string):
    Map=Counter(string)
    entropy=0.0
    for val in Map.values():
        entropy-=val/len(string)*math.log(val/len(string))
    return entropy

clf = RandomForestClassifier(random_state=0)
domainlist = []
class Domain:
    def __init__(self,string,label):
        self.length=len(string)
        self.numbers=Count_numbers(string)
        self.entropy=Calculate_entropy(string)
        self.label=label
    
    def Return_data(self):
        return [self.length,self.numbers,self.entropy]
    
    def Return_label(self):
        if(self.label=="dga"):
            return 0
        else:
            return 1

def initData(filename):
    with open(filename,"r") as f:
        lines=f.readlines()
        for line in lines:
            line=line.strip()
            if line.startswith("#") or line =="":
                continue
            tokens=line.split(",")
            domainlist.append(Domain(tokens[0],tokens[1]))

def predictData(testfile,outputfile):
    with open(testfile,"r") as f1:
        with open(outputfile,"w") as f2:
            lines=f1.readlines()
            for line in lines:
                line=line.strip()
                if line.startswith("#") or line =="":
                    continue
                P=clf.predict([[len(line),Count_numbers(line),Calculate_entropy(line)]])
                if(P==[0]):
                    f2.write(line+",dga\n")
                else:
                    f2.write(line+",notdga\n")

if __name__=='__main__':
    initData("train.txt")
    featureMatrix=[]
    labelList=[]
    for item in domainlist:
	    featureMatrix.append(item.Return_data())
	    labelList.append(item.Return_label())
    clf.fit(featureMatrix,labelList)
    predictData("test.txt","result.txt")
