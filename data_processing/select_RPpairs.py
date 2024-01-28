import numpy as np
import pandas as pd
import rdkit as rd
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.DataStructs import FingerprintSimilarity as fs
from rdkit.Chem import RDKFingerprint 

file = pd.read_csv("chem.csv")
list = []
lines = file.iloc[:,0]
for i in range(len(lines)):
    list.append([eval(file.iloc[i,1]),eval(file.iloc[i,2])])

newlist = []
count = 0
degre = []
for i in range(len(list)):
    similist = []
    my = []
    ay = []
    for y in list[i][1]:
        caly = Chem.RemoveHs(Chem.MolFromSmiles(y))
        my.append(ExactMolWt(caly))
        ay.append(caly.GetNumAtoms())
    placey = np.argmax(my)
    for x in list[i][0]:    
        xfing = RDKFingerprint(Chem.RemoveHs(Chem.MolFromSmiles(x)))
        yfing = RDKFingerprint(Chem.RemoveHs(Chem.MolFromSmiles(list[i][1][placey])))
        similist.append(fs(xfing,yfing))
    placex = np.argmax(similist)
    newlist.append([list[i][0][placex],list[i][1][placey]])

print(len(newlist))   
print(newlist[:5])

df = pd.DataFrame(newlist)
df.to_csv("chem2.csv")

