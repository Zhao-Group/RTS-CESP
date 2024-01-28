import numpy as np
import pandas as pd
import rdkit as rd
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.DataStructs import FingerprintSimilarity as fs
from rdkit.Chem import RDKFingerprint 


file = pd.read_csv("ecreact.csv")
list = []
newlist = []
lines = file.iloc[:,0]
for line in lines:
    list.append(line)
for str in list:
    x,y = str.split(">>")
    newlist.append([x.split("|")[0].split("."),y.split(".")])
molecules = ["O=P([O-])([O-])[O-]","O=P([O-])([O-])O","C[N+](C)(C)CCO","NCCO","O=P([O-])([O-])OP(=O)([O-])[O-]",\
    "O=P([O-])([O-])OP(=O)([O-])O","O=C([O-])CCC(=O)C(=O)[O-]","CC(=O)[O-]","CC(=O)C(=O)[O-]"]
count = 0
for i in range(len(newlist)):
    for j in range(2):
        for str in molecules:
            if str in newlist[i][j]:
                count += 1
                del newlist[i][j][newlist[i][j].index(str)]

out = []
out.append(["newlist",len(newlist)])
empty = []
for i in range (len(newlist)):
    if newlist[i][0] == [] or newlist[i][1] == []:
        empty.append(i)
empty.reverse()
for str in empty:
    del newlist[str]
out.append(["cleaned newlist",len(newlist)])



degre = []
count = 0
for i in range(len(newlist)):
    m1 = []
    m2 = []
    a1 = []
    a2 = []
    for str in newlist[i][0]:
        cal1 = Chem.RemoveHs(Chem.MolFromSmiles(str))
        m1.append(ExactMolWt(cal1))
        a1.append(cal1.GetNumAtoms())
    mass1 = max(m1) if m1!=[] else 0
    atom1 = max(a1) if a1!=[] else 0
    for str in newlist[i][1]:
        cal2 = Chem.RemoveHs(Chem.MolFromSmiles(str))
        m2.append(ExactMolWt(cal2))
        a2.append(cal2.GetNumAtoms())
    mass2 = max(m2) if m2!=[] else 0
    atom2 = max(a2) if a2!=[] else 0
    if mass1 > mass2 and atom1 > atom2:
        degre.append(i)
        count +=1

out.append(["first removed count",count])
examlist = []
degre.reverse()
for str in degre:
    examlist.append(newlist[str])
    del newlist[str]
out.append(["deleted newlist",len(newlist)])
out.append(["formed examlist",len(examlist)])

count = 0
degre = []
for i in range(len(examlist)):
    similist = []
    mx = []
    my = []
    ax = []
    ay = []
    for y in examlist[i][1]:
        caly = Chem.RemoveHs(Chem.MolFromSmiles(y))
        my.append(ExactMolWt(caly))
        ay.append(caly.GetNumAtoms())
    placey = np.argmax(my)
    for x in examlist[i][0]:    
        xfing = RDKFingerprint(Chem.RemoveHs(Chem.MolFromSmiles(x)))
        yfing = RDKFingerprint(Chem.RemoveHs(Chem.MolFromSmiles(examlist[i][1][placey])))
        similist.append(fs(xfing,yfing))
    placex = np.argmax(similist)
    calx = Chem.RemoveHs(Chem.MolFromSmiles(examlist[i][0][placex]))
    mx.append(ExactMolWt(calx))
    ax.append(calx.GetNumAtoms())
    if mx > my and ax > ay:
        degre.append(i)
        count +=1
out.append(["second removed count",count])
degre.reverse()
for str in degre:
    del examlist [str]
out.append(["final examlist",len(examlist)])


df1 = pd.DataFrame(newlist)
df2 = pd.DataFrame(examlist)
dflist = [df1,df2]
newlist = pd.concat(dflist)
out.append(["final newlist",len(newlist)])

df = pd.DataFrame(newlist)
df.to_csv("whole.csv")
print(out)



    
