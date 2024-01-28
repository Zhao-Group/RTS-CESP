from rdkit import Chem
from rdkit.Chem import AllChem
from bio_score import *
from rdkit.DataStructs.cDataStructs import BitVectToText as bvt


def calculate_RTscore_bio(molpair):
    x = Chem.MolFromSmiles(molpair[0][0])
    y = Chem.MolFromSmiles(molpair[0][1])            
    fpx = AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=1024)
    fpy = AllChem.GetMorganFingerprintAsBitVect(y,2,nBits=1024)
    molfp = [eval(i) for i in bvt(fpx)+bvt(fpy)]
    tenfp = (torch.tensor(molfp, dtype = torch.float)).view(1,1,2048).expand(512,1,2048)
    bio_model.load_state_dict(torch.load('biockpt.mdl'))
    stepscore = bio_model.forward(tenfp)[0][0]
    stepscore_number = stepscore.item()
    print(molpair, stepscore_number)

    return stepscore_number


bio_molpair = [['Nc1ccc(O)cc1', 'Nc1ccccc1']]
calculate_RTscore_bio(bio_molpair)
