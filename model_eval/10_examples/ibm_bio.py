import time
import torch
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.DataStructs import FingerprintSimilarity as fs
from rdkit.Chem import RDKFingerprint 
from rdkit.DataStructs.cDataStructs import BitVectToText as bvt
from rxn4chemistry import RXN4ChemistryWrapper


class IBM_bio():
    def get_result(molecule):
        api_key = 'apk-ce6eaef0a1a0304d40f2934f6d83d200ef3a710b6f6e90205602916c580ff6e9951974add05d9e4457efb903d2d7c09d41d858c29ff730ac7dc0ad42588693dfff1215102c68e50aaf9316432c958c36'
        rxn4chemistry_wrapper = RXN4ChemistryWrapper(api_key=api_key)
        #rxn4chemistry_wrapper.set_project('first')
        rxn4chemistry_wrapper.create_project('no2')
        print(rxn4chemistry_wrapper.project_id)

        response = rxn4chemistry_wrapper.predict_automatic_retrosynthesis(
            # molecule, ai_model='enzymatic-2021-04-16', max_steps = 1
            molecule, ai_model='enzymatic-2021-04-16'
        )

        while True:
            time.sleep(90)
            results = rxn4chemistry_wrapper.get_predict_automatic_retrosynthesis_results(
            response['prediction_id']
            )
            if results['status'] == 'SUCCESS':
                print(results['status'])
                break

        return results
        #print(results['retrosynthetic_paths'][0])

    def collect_reactions(tree):
        reactions = []
        if 'children' in tree and len(tree['children']):
            reactions.append([tree['smiles'],'.'.join([node['smiles'] for node in tree['children']])])
        for node in tree['children']:
            reactions.extend(IBM_bio.collect_reactions(node))
        return reactions

    def generate_reaction_list(molecule):
        try:
            bio_list = []
            results = IBM_bio.get_result(molecule)
            for index, path in enumerate(results['retrosynthetic_paths']):
                bio_list.append(IBM_bio.collect_reactions(path))
            # print(bio_list)

        except KeyError:
             bio_list = [[[molecule,molecule]]]

        return bio_list

# bio_target = 'O=Cc1ccccc1'
# IBM_bio.generate_reaction_list(bio_target)

