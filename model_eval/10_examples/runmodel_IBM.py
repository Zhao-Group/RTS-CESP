import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import FingerprintSimilarity as fs
from rdkit.Chem import RDKFingerprint 
from rdkit.DataStructs.cDataStructs import BitVectToText as bvt
from bio_score import *
from chem_score import *
from ibm_bio import *
from ibm_chem import *
import os
import sys
import time


class get_synthesis_result():
    def get_retrosynthesis_result(target):
        time.sleep(90)
        chem_routes_list = IBM_chem.generate_reaction_list(target)
    
        #print(chem_routes_list)
        return chem_routes_list
        

    def get_retrobiosynthesis_result(target):
        time.sleep(90)
        bioresult = IBM_bio.generate_reaction_list(target)

        # print(bioresult)
        return bioresult

class ChemScore():
    def molpair_generator(routes):
        allroutepair = []
        for i in range(len(routes)):
            allmolpair = []
            one_route = routes[i]
            for reaction in range(len(one_route)):
                product = one_route[reaction][0]
                reactant = one_route[reaction][1]
                if "." in reactant:
                    reactant = reactant.split(".")
                    molecule_similar_list = []
                    product_fingerprint = RDKFingerprint(Chem.MolFromSmiles(product))
                    for count in range(len(reactant)):
                        reactant_fingerprint = RDKFingerprint(Chem.MolFromSmiles(reactant[count]))
                        molecule_similar_list.append(fs(reactant_fingerprint,product_fingerprint))
                    placex = np.argmax(molecule_similar_list)
                    major_reactant = reactant[placex]
                    #one_route[reaction][1] = major_reactant
                    allmolpair.append([product,major_reactant])
                    # for i in range(len(molecule_similar_list)):
                    #     if molecule_similar_list[i] >= 0.1 and i!= placex:
                    #         allmolpair.append([product,reactant[i]])
                else:
                    allmolpair.append([product,reactant])
            allroutepair.append(allmolpair)

            delist = []
            for i in range(len(allroutepair)): 
                for a in range(i):
                    if allroutepair[i] == allroutepair[a]:
                        delist.append(i)
            delist = list(set(delist))
            delist.reverse()
            for item in delist:
                del allroutepair[item]

        #print(allroutepair)
        return allroutepair


    def routescore(allmolpair):
        score_result_list = []
        for i in range(len(allmolpair)):
            routepair = allmolpair[i]
            routescore = 0
            score_result_dict = {}
            step_result_list = []
            for i in range(len(routepair)):
                molpair = routepair[i]
                #print(molpair)
                x = Chem.MolFromSmiles(molpair[0])
                y = Chem.MolFromSmiles(molpair[1])            
                fpx = AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=1024)
                fpy = AllChem.GetMorganFingerprintAsBitVect(y,2,nBits=1024)
                molfp = [eval(i) for i in bvt(fpx)+bvt(fpy)]
                tenfp = (torch.tensor(molfp, dtype = torch.float)).view(1,1,2048).expand(512,1,2048)
                chem_model.load_state_dict(torch.load('chemckpt.mdl'))
                stepscore = chem_model.forward(tenfp)[0][0]
                stepscore_number = stepscore.item()
                routescore += stepscore_number
                step_result = [molpair,  stepscore_number, "chem"]
                step_result_list.append(step_result)
                #score_result_dict["step"] = molpair
                #score_result_dict["step_score"] = stepscore
            
            score_result_dict["route"] = step_result_list
            score_result_dict["route_score"] = routescore
            score_result_list.append(score_result_dict)
            score_result_list.sort(key = lambda x: x.get("route_score"))

        print(score_result_list) 

        return score_result_list

class BioScore():
    def molpair_generator(routes):
        allmolpair = []
        for i in range(len(routes)):
            molstr = routes[i][0]
            #print(molstr)
            product = molstr[0]
            reactant = molstr[1]
            # if more than one reactant, select major one(most similar to product)
            if "." in reactant:
                reactant = reactant.split(".")
                molecule_similar_list = []
                product_fingerprint = RDKFingerprint(Chem.MolFromSmiles(product))
                for a in range(len(reactant)):
                    reactant_fingerprint = RDKFingerprint(Chem.MolFromSmiles(reactant[a]))
                    molecule_similar_list.append(fs(reactant_fingerprint,product_fingerprint))
                placex = np.argmax(molecule_similar_list)
                major_reactant = reactant[placex]
                allmolpair.append([product,major_reactant])
            else:
                allmolpair.append([product,reactant])
        delist = []
        for i in range(len(allmolpair)): 
            for a in range(i):
                if allmolpair[i] == allmolpair[a]:
                    delist.append(i)
        delist = list(set(delist))
        delist.reverse()
        for item in delist:
            del allmolpair[item]
        # print(allmolpair)
        return allmolpair

    def routescore(allmolpair):
        score_result_list = []
        for i in range(len(allmolpair)):
            score_result_dict = {}
            molpair = allmolpair[i]
            x = Chem.MolFromSmiles(molpair[0])
            y = Chem.MolFromSmiles(molpair[1])   
            if x == y:
               routescore = 1
            else:         
                fpx = AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=1024)
                fpy = AllChem.GetMorganFingerprintAsBitVect(y,2,nBits=1024)
                molfp = [eval(i) for i in bvt(fpx)+bvt(fpy)]
                tenfp = (torch.tensor(molfp, dtype = torch.float)).view(1,1,2048).expand(512,1,2048)
                bio_model.load_state_dict(torch.load('biockpt.mdl'))
                routescore = bio_model.forward(tenfp)[0][0].item()
            score_result_dict["bio_route"] = [molpair, routescore, "bio"]
            score_result_dict["bio_score"] = routescore
            score_result_dict["chem_target"] = molpair[1]
            score_result_list.append(score_result_dict)
        score_result_list.sort(key = lambda x: x.get("bio_score"))

        # print(score_result_list) 

        return score_result_list



class Searching_Algorithm():
    def extract_chem_intermediate(chem_target):
        chem_routes = get_synthesis_result.get_retrosynthesis_result(chem_target)
        chem_route_pair = ChemScore.molpair_generator(chem_routes)
        scored_chem_routes = ChemScore.routescore(chem_route_pair)
        chem_routes_list = []
        for item in scored_chem_routes:
            chem_routes_dict = {}
            chem_routes = item["route"]
            score = item["route_score"]
            if len(chem_routes) == 1:
                chem_routes_dict = {}
                bio_target = chem_routes[0][0][1]
                reserved_chem_score = score
                reserved_chem_routes = chem_routes
                chem_routes_dict["bio_target"] = bio_target
                chem_routes_dict["reserved_chem_score"] = reserved_chem_score
                chem_routes_dict["reserved_chem_routes"] = reserved_chem_routes
                # chem_routes_dict["original_chem_route"] = chem_routes
                # chem_routes_dict["original_whole_score"] = score
                if chem_routes_dict["reserved_chem_score"] != 0:
                    chem_routes_list.append(chem_routes_dict)
                # new_chem_target = chem_routes[-1][0][1]
                # new_chem_routes = get_synthesis_result.get_retrosynthesis_result(new_chem_target)
                # new_chem_route_pair = ChemScore.molpair_generator(new_chem_routes)
                # new_scored_chem_routes = ChemScore.routescore(new_chem_route_pair)
                # for i in range(len(new_scored_chem_routes)):
                #     new_chem_routes_dict = {}
                #     new_chem_routes = new_scored_chem_routes[i]["route"]
                #     new_score = new_scored_chem_routes[i]["route_score"]
                #     new_chem_routes_dict["bio_target"] = new_chem_routes[-1][0][1]
                #     new_chem_routes_dict["reserved_chem_score"] = reserved_chem_score + new_score
                #     new_chem_routes_dict["reserved_chem_routes"] = [reserved_chem_routes[0]+new_chem_routes[:]]
                #     if new_chem_routes_dict["reserved_chem_score"] != 0:
                #         chem_routes_list.append(new_chem_routes_dict)
            else:
                chem_routes_dict = {}
                step_score = []
                for i in range(len(chem_routes)):
                    step_score.append(chem_routes[i][1])
                place = np.argmax(step_score)
                bio_target = chem_routes[place][0][0]
                reserved_chem_score = 0
                for a in range(place):
                    reserved_chem_score += chem_routes[a][1]
                reserved_chem_routes = chem_routes[:place]
                chem_routes_dict["bio_target"] = bio_target
                chem_routes_dict["reserved_chem_score"] = reserved_chem_score
                chem_routes_dict["reserved_chem_routes"] = reserved_chem_routes
                # chem_routes_dict["original_chem_route"] = chem_routes
                # chem_routes_dict["original_whole_score"] = score
                if chem_routes_dict["reserved_chem_score"] != 0:
                    chem_routes_list.append(chem_routes_dict)
                # new_chem_routes_dict = {}
                # new_chem_routes_dict["bio_target"] = chem_routes[-1][0][1]
                # new_chem_routes_dict["reserved_chem_score"] = score
                # new_chem_routes_dict["reserved_chem_routes"] = chem_routes
                # if new_chem_routes_dict["reserved_chem_score"] != 0:
                #     chem_routes_list.append(new_chem_routes_dict)   
        target_list = []
        del_list = []
        for i in range(len(chem_routes_list)):
            target_list.append(chem_routes_list[i]["bio_target"])
        for i in range(len(target_list)): 
            for a in range(i):
                if target_list[i] == target_list[a]:
                    del_list.append(i)
        del_list = list(set(del_list))
        del_list.reverse()
        for item in del_list:
            del chem_routes_list[item]
        chem_routes_list.sort(key = lambda x: x.get("reserved_chem_score"))

        print(chem_routes_list)

        return(chem_routes_list)


    def bio2chem(chem_target):
        chembio_route_list_total = []
        chem_intermediate = Searching_Algorithm.extract_chem_intermediate(chem_target)
        for item in chem_intermediate:
            bio_target = item["bio_target"]
            bioroutes = get_synthesis_result.get_retrobiosynthesis_result(bio_target)
            biomolpair = BioScore.molpair_generator(bioroutes) 
            bio_route_score = BioScore.routescore(biomolpair)
            #print(bio_route_score)
            for i in range(len(bio_route_score)):
                chembio_route_dict = item
                output_dict = {}
                chembio_route_dict["biostep"] = bio_route_score[i]["bio_route"]
                chembio_route_dict["bioscore"] = bio_route_score[i]["bio_score"]
                if chembio_route_dict["bioscore"] != 1:
                    output_dict["overall_routes"] = chembio_route_dict["reserved_chem_routes"][:] + [chembio_route_dict["biostep"]]
                    output_dict["overall_score"] = chembio_route_dict["reserved_chem_score"] + chembio_route_dict["bioscore"] 
                    chembio_route_list_total.append(output_dict)
        
        chembio_route_list_total.sort(key = lambda x: x.get("overall_score"))
        routes_count = len(chembio_route_list_total)
        chembio_route_list_total.append(routes_count)
        print(chembio_route_list_total)

        return chembio_route_list_total

    
    def wrap_up(chem_target):
        chembiochem_route_list_total = []
        bio_intermediate = Searching_Algorithm.bio2chem(chem_target)
        for item in bio_intermediate:
            chem_target_new = item["chemtarget"]
            #print(chem_target)
            chemroutes_new = get_synthesis_result.get_retrosynthesis_result(chem_target_new)
            chemmolpair_new = ChemScore.molpair_generator(chemroutes_new) 
            chem_route_score_new = ChemScore.routescore(chemmolpair_new)
            for i in range(len(chem_route_score_new)):
                chembiochem_route_dict = item
                chembiochem_route_dict["chem_next_step"] = chem_route_score_new[i]["route"]
                chembiochem_route_dict["chem_next_score"] = chem_route_score_new[i]["route_score"]
                chembiochem_route_dict["chembiochem_overall_score"] = chembiochem_route_dict["reserved_chem_score"] + chembiochem_route_dict["bioscore"] + chembiochem_route_dict["chem_next_score"]
                chembiochem_route_dict["chembiochem_overall_routes"] = chembiochem_route_dict["reserved_chem_routes"][:] + [chembiochem_route_dict["biostep"]] + chembiochem_route_dict["chem_next_step"][:]
                chembiochem_route_list_total.append(chembiochem_route_dict)

        chembiochem_route_list_total.sort(key = lambda x: x.get("chembiochem_overall_score"))
            


        print(chembiochem_route_list_total)

        return chembiochem_route_list_total


def main():
    target_list = [

        "C1N(C[C@H](C2C=CC=CC=2)O)CCOC1"

    ]
    for i in range(len(target_list)):
        time.sleep(120)
        chem_target = target_list[i]
        results = Searching_Algorithm.bio2chem(chem_target)

        with open("results/"+chem_target+".txt", "w") as f:
            for item in results:
                f.write("%s\n" % item)

main()











