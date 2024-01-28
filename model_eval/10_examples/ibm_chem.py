import time
import torch
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.DataStructs import FingerprintSimilarity as fs
from rdkit.Chem import RDKFingerprint 
from rdkit.DataStructs.cDataStructs import BitVectToText as bvt
from rxn4chemistry import RXN4ChemistryWrapper


class IBM_chem():
    def get_result(molecule):
        api_key = 'apk-ce6eaef0a1a0304d40f2934f6d83d200ef3a710b6f6e90205602916c580ff6e9951974add05d9e4457efb903d2d7c09d41d858c29ff730ac7dc0ad42588693dfff1215102c68e50aaf9316432c958c36'
        rxn4chemistry_wrapper = RXN4ChemistryWrapper(api_key=api_key)
        #rxn4chemistry_wrapper.set_project('first')
        rxn4chemistry_wrapper.create_project('no1')
        print(rxn4chemistry_wrapper.project_id)

        response = rxn4chemistry_wrapper.predict_automatic_retrosynthesis(
            molecule, ai_model='12class-tokens-2021-05-14'
        )

        while True:
            time.sleep(90)
            results = rxn4chemistry_wrapper.get_predict_automatic_retrosynthesis_results(
            response['prediction_id']
            )
            if results['status'] == 'SUCCESS':
                print(results['status'])
                break
        # print(results['retrosynthetic_paths'])

        return results


    def collect_reactions(tree):
        reactions = []
        if 'children' in tree and len(tree['children']):
            reactions.append([tree['smiles'],'.'.join([node['smiles'] for node in tree['children']])])
        for node in tree['children']:
            reactions.extend(IBM_chem.collect_reactions(node))
        return reactions

    def generate_reaction_list(molecule):
        try:
            chem_list = []
            results = IBM_chem.get_result(molecule)
            for index, path in enumerate(results['retrosynthetic_paths']):
                chem_list.append(IBM_chem.collect_reactions(path))

        except KeyError:
             chem_list = [[[molecule,molecule]]]

        print(chem_list)
        return chem_list
time.sleep(30)
chem_target = 'C1C([C@@H](CN)O)=CC=CC=1'
IBM_chem.generate_reaction_list(chem_target)



# allroutepair = []
# for i in range(len(chem_list)):
#     allmolpair = []
#     for reaction in chem_list[i]:
#         product = reaction[0]
#         reactant = reaction[1]
#         if "." in reactant:
#             reactant = reactant.split(".")
#             molecule_similar_list = []
#             product_fingerprint = RDKFingerprint(Chem.MolFromSmiles(product))
#             for a in range(len(reactant)):
#                 reactant_fingerprint = RDKFingerprint(Chem.MolFromSmiles(reactant[a]))
#                 molecule_similar_list.append(fs(reactant_fingerprint,product_fingerprint))
#             placex = np.argmax(molecule_similar_list)
#             major_reactant = reactant[placex]
#             allmolpair.append([product,major_reactant])
#         else:
#             allmolpair.append([product,reactant])
#     allroutepair.append(allmolpair)
# print(allroutepair)

