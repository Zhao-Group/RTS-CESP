import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import RDKFingerprint 
from rdkit.DataStructs.cDataStructs import BitVectToText


def read_reaction_pair_file(file_name):
    file = pd.read_csv(file_name)
    reaction_list = []
    lines = file.iloc[:,0]
    for i in range(len(lines)):
        reaction_list.append([file.iloc[i,1],file.iloc[i,2]])
    # print(len(reaction_list))
    # print(reaction_list[:5])
    return reaction_list

def convert_smiles_to_fingerprint_and_write_file(reaction_list):
    fingerprint_list = []
    for i in range(len(reaction_list)):
        x = Chem.MolFromSmiles(reaction_list[i][0])
        y = Chem.MolFromSmiles(reaction_list[i][1])
        fpx = AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=1024)
        fpy = AllChem.GetMorganFingerprintAsBitVect(y,2,nBits=1024)
        fingerprint_list.append([BitVectToText(fpx),BitVectToText(fpy)])
    # print(len(fingerprint_list))
    # print(fingerprint_list)

    return fingerprint_list


def write_forward_reaction(fingerprint_list):
    forward_whole_list = []
    for i in range(len(fingerprint_list)):
        forwardlist = [eval(i) for i in list(fingerprint_list[i][0])+list(fingerprint_list[i][1])]
        forward_whole_list.append(forwardlist)
    # print(forward_whole_list[:2])
    forward_array = np.array(forward_whole_list)
    forward_array.tofile("chem_forward_list.dat", sep = "", format = "%d")


def write_backward_reaction(fingerprint_list):
    backward_whole_list = []
    for i in range(len(fingerprint_list)):
        backwardlist = [eval(i) for i in list(fingerprint_list[i][1])+list(fingerprint_list[i][0])]
        backward_whole_list.append(backwardlist)
    # print(backward_whole_list[:2])
    backward_array = np.array(backward_whole_list)
    backward_array.tofile("chem_backward_list.dat", sep = "", format = "%d")


# chem_reaction_list = read_reaction_pair_file("uspto_single_reaction_pair.csv")
# chem_fingerprint_list = convert_smiles_to_fingerprint_and_write_file(chem_reaction_list)
# write_forward_reaction(chem_fingerprint_list)
# write_backward_reaction(chem_fingerprint_list)

qianarray = np.fromfile("chem_forward_list.dat", dtype=int, count=-1, sep='').reshape(-1,2048)
houarray = np.fromfile("chem_backward_list.dat", dtype=int, count=-1, sep='').reshape(-1,2048)

print(qianarray[:2])
print(houarray[:2])




