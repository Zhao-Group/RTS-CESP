import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import FingerprintSimilarity as fs
from rdkit.Chem import RDKFingerprint 
from rdkit.DataStructs.cDataStructs import BitVectToText as bvt
from bio_score import *
from chem_score import *
from aizynthfinder.aizynthfinder import AiZynthFinder
from aizynthfinder.aizynthfinder import AiZynthExpander
from rdchiral.main import *

smile_list = []
with open('buyables.json', 'r') as f:
    line = f.readline()
    data = json.loads(line)
    for item in data:
        smile_list.append(item["smiles"])

bio_template = pd.read_csv('whole_enzymatic_template.csv').loc[0:, 'template'].values.tolist()

# target = ["C/C(N)=N/CC[C@@H](F)C[C@H](N)C(=O)O"]
# target = ["C=CCOC1=C(Cl)C=C(CC(O)=O)C=C1"]
target = pd.read_csv('hybrid_only_111.csv').loc[10:20, 'smiles'].values
target = ["C=CCOC1=C(Cl)C=C(CC(O)=O)C=C1"]

filename = "config.yml"
finder = AiZynthFinder(configfile=filename)
finder.stock.select("stock")
finder.expansion_policy.select("uspto")
finder.filter_policy.select("uspto")

for chem_target in target:
    time_list = []
    class get_chemical_synthesis():
        def get_synthesis(chem_target):            
            print(chem_target)            
            chem_target = chem_target
            finder.target_smiles = chem_target
            chem_time = finder.tree_search()
            finder.build_routes()
            stats = finder.extract_statistics()
            trees = finder.routes.dicts
            time_list.append(chem_time)
            print(chem_time)
            chem_target_name = chem_target.replace("/", "_").replace("\\", "-")
            with open("original_result/"+chem_target_name+".txt", "a") as f:
                f.write("%s\n" % stats)
                f.write("%s\n" % trees)
                f.write("%s\n" % time_list)
            return trees

        def extract_routes(routes):
            # routes = eval(lines[1])[1]
            all_list = []
            children_list = []
            for route in routes:
                smile = route["smiles"]
                stock = route["in_stock"]
                if "children" in route:
                    precursor = route["children"][0]["children"]
                    # print(precursor)
                    children_list.append([smile,"|".join(items["smiles"] for items in precursor),stock,[items["in_stock"] for items in precursor]])
                    for nodes in precursor:
                        if "children" in nodes:
                            children_list.extend(get_chemical_synthesis.extract_routes(precursor))
                # all_list.append(children_list)        
            return children_list
            # print(children_list)

        def reaction_list(output):
            listed_reactions = []
            for items in output:
                result = get_chemical_synthesis.extract_routes([items])
                listed_reactions.append(result)
            return(listed_reactions)
        
        # result = get_synthesis(target)
        # reactions = reaction_list(result)
        # print(reactions)
        # return reactions

    class get_biological_synthesis():
        def get_retrobiocat_result(product):
            
            # with open('retrobiocat_database.pkl', 'rb') as f:
            #     data = pickle.load(f)

            # # reaction_smarts = '[C:1][OH:2]>>[C:1][O:2][C]'
            # outlist = []
            # # product = 'OCC(=O)OCCCO'
            # start = time.time()
            # for template in data["Smarts"]:
            #     outcomes = rdchiralRunText(template, product)
            #     if outcomes != []:
            #         outlist.append(outcomes)
            
            outlist = []
            # product = 'OCC(=O)OCCCO'
            start = time.time()
            for template in bio_template:
                outcomes = rdchiralRunText(template, product)
                if outcomes != []:
                    outlist.append(outcomes)

            end = time.time()
            biotime = end-start
            time_list.append(biotime)

            biorxn_list = []
            precursor_list = []
            for items in outlist:
                for precursors in items:
                    precursor_list.append(precursors.replace(".","|"))

            for items in precursor_list:
                if "|" not in items:
                    if items in smile_list:
                        biorxn_list.append([[product,items,False,[True]]])
                    else:
                        biorxn_list.append([[product,items,False,[False]]])
                else:
                    prelist = []
                    precursor_count = items.count("|")
                    prelist.append([product,items,False,[False]])
                    for i in range(precursor_count):
                        prelist[0][3].append(False)
                    biorxn_list.append(prelist)

            # print(outlist)
            # print(precursor_list)
            # print(biorxn_list)

           
            # return chem_routes_list
            # print(data)
            return biorxn_list
            

    class ChemScore():
        def aiz_molpair_generator(routes):

            allroutepair = []
            for i in range(len(routes)):
                allmolpair = []
                one_route = routes[i]
                for reaction in range(len(one_route)):
                    product = one_route[reaction][0]
                    reactant = one_route[reaction][1]

            # allroutepair = []
            # for i in range(len(routes)):
            #     # print(i)
            #     allmolpair = []
            #     # print(routes)
            #     try:
            #         one_route = eval(routes)[i]
            #     except IndexError:
            #         continue
            #     # print(one_route)
            #     for reaction in range(len(one_route)):
                    
            #         product = one_route[reaction][0]
            #         # print(one_route[reaction])
            #         # print(reaction)
            #         reactant = one_route[reaction][1]
                
                    if "|" in reactant:
                        reactant = reactant.split("|")
                        molecule_similar_list = []
                        product_fingerprint = RDKFingerprint(Chem.MolFromSmiles(product))
                        for count in range(len(reactant)):
                            reactant_fingerprint = RDKFingerprint(Chem.MolFromSmiles(reactant[count]))
                            molecule_similar_list.append(fs(reactant_fingerprint,product_fingerprint))
                        placex = np.argmax(molecule_similar_list)
                        major_reactant = reactant[placex]
                        #one_route[reaction][1] = major_reactant
                        commercial_available = one_route[reaction][3][placex]
                        if major_reactant in smile_list:
                            commercial_available = True
                        if major_reactant not in smile_list:
                            commercial_available = False
                        allmolpair.append([[product,major_reactant], commercial_available])
                        # for i in range(len(molecule_similar_list)):
                        #     if molecule_similar_list[i] >= 0.1 and i!= placex:
                        #         allmolpair.append([product,reactant[i]])
                    else:
                        commercial_available = one_route[reaction][3][0]
                        if reactant in smile_list:
                            commercial_available = True
                        if reactant not in smile_list:
                            commercial_available = False
                        allmolpair.append([[product,reactant],commercial_available])
                # print(allmolpair)
                count = 0
                while count <= len(allmolpair):    
                    try:
                        # print(count)        
                        if allmolpair[count][0][1] != allmolpair[count+1][0][0]:
                            del allmolpair[count+1]
                        else:
                            count +=1
                    except IndexError:
                        break


                allroutepair.append(allmolpair)

                delist = []
                for i in range(len(allroutepair)): 
                    for a in range(i):
                        if allroutepair[i] == allroutepair[a]:
                            delist.append(i)
                delist = list(set(delist))
                delist.reverse()
                for item in delist:
                    try:
                        del allroutepair[item]
                    except IndexError:
                        continue

            print(allroutepair)
            return allroutepair
        

        # def molpair_generator(routes):
        #     allroutepair = []
        #     for i in range(len(routes)):
        #         allmolpair = []
        #         one_route = routes[i]
        #         for reaction in range(len(one_route)):
        #             product = one_route[reaction][0]
        #             reactant = one_route[reaction][1]
                
        #             if "|" in reactant:
        #                 reactant = reactant.split("|")
        #                 molecule_similar_list = []
        #                 product_fingerprint = RDKFingerprint(Chem.MolFromSmiles(product))
        #                 for count in range(len(reactant)):
        #                     reactant_fingerprint = RDKFingerprint(Chem.MolFromSmiles(reactant[count]))
        #                     molecule_similar_list.append(fs(reactant_fingerprint,product_fingerprint))
        #                 placex = np.argmax(molecule_similar_list)
        #                 major_reactant = reactant[placex]
        #                 #one_route[reaction][1] = major_reactant
        #                 commercial_available = one_route[reaction][3][placex]
        #                 allmolpair.append([[product,major_reactant], commercial_available])
        #                 # for i in range(len(molecule_similar_list)):
        #                 #     if molecule_similar_list[i] >= 0.1 and i!= placex:
        #                 #         allmolpair.append([product,reactant[i]])
        #             else:
        #                 commercial_available = one_route[reaction][3][0]
        #                 allmolpair.append([[product,reactant],commercial_available])
                    
        #         allroutepair.append(allmolpair)

        #         delist = []
        #         for i in range(len(allroutepair)): 
        #             for a in range(i):
        #                 if allroutepair[i] == allroutepair[a]:
        #                     delist.append(i)
        #         delist = list(set(delist))
        #         delist.reverse()
        #         for item in delist:
        #             try:
        #                 del allroutepair[item]
        #             except IndexError:
        #                 continue

        #     print(allroutepair)
        #     return allroutepair

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
                    x = Chem.MolFromSmiles(molpair[0][0])
                    y = Chem.MolFromSmiles(molpair[0][1])            
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

            # count = 0
            # while count <= len(score_result_list):
            #     try:
            #         if abs(score_result_list[count]["route_score"] - score_result_list[count+1]["route_score"]) < 0.000001:
            #             del score_result_list[count+1]
            #         else:
            #             count +=1
            #     except IndexError:
            #         break

             


            # chem_success_list = []
            # chem_failed_list = []
            # for item in score_result_list:
            #     if item["route"][-1][0][1] == True:
            #         chem_success_list.append(item)
            #     else:
            #         chem_failed_list.append(item)
            # chem_success_dict = {}
            # if len(chem_success_list) != 0:
            #     route_lenth = []
            #     for item in chem_success_list:
            #         route_lenth.append(len(item["route"]))
            #     chem_success_dict["target_molecule"] = score_result_list[0]["route"][0][0][0][0]
            #     chem_success_dict["all_route"] = chem_success_list
            #     chem_success_dict["shortest_route_lenth"] = min(route_lenth)
            #     chem_success_dict["chem_route_count"] = len(chem_success_list)
            #     chem_success_dict["failed_routes"] = chem_failed_list
            # else:
            #     chem_success_dict["target_molecule"] = score_result_list[0]["route"][0][0][0][0]
            #     chem_success_dict["all_route"] = []
            #     chem_success_dict["failed_routes"] = chem_failed_list
            
            # with open("results/"+chem_target+".txt", "w") as f:
            #     f.write("%s\n" % chem_success_dict)

            print(score_result_list) 
            commercial = False

            for item in score_result_list:
                print(item)
                if item["route"][-1][0][1] == True:
                    commercial = True
                    
            return score_result_list, commercial

    class BioScore():
        def molpair_generator(routes):
            allmolpair = []
            print(routes)
            for i in range(len(routes)):
                molstr = routes[i][0]
                print(molstr)
                #print(molstr)
                product = molstr[0]
                reactant = molstr[1]
                # if more than one reactant, select major one(most similar to product)
                if "|" in reactant:
                    reactant = reactant.split("|")
                    molecule_similar_list = []
                    product_fingerprint = RDKFingerprint(Chem.MolFromSmiles(product))
                    for a in range(len(reactant)):
                        reactant_fingerprint = RDKFingerprint(Chem.MolFromSmiles(reactant[a]))
                        molecule_similar_list.append(fs(reactant_fingerprint,product_fingerprint))
                    placex = np.argmax(molecule_similar_list)
                    major_reactant = reactant[placex]
                    commercial_available = molstr[3][placex]
                    if major_reactant in smile_list:
                        commercial_available = True
                    allmolpair.append([[product,major_reactant], commercial_available])
                else:
                    commercial_available = molstr[3][0]
                    if reactant in smile_list:
                        commercial_available = True
                    allmolpair.append([[product,reactant],commercial_available])
            delist = []
            for i in range(len(allmolpair)): 
                for a in range(i):
                    if allmolpair[i] == allmolpair[a]:
                        delist.append(i)
            delist = list(set(delist))
            delist.reverse()
            for item in delist:
                try:
                    del allmolpair[item]
                except IndexError:
                    continue
            print(allmolpair)
            return allmolpair

        def routescore(allmolpair):
            score_result_list = []
            for i in range(len(allmolpair)):
                score_result_dict = {}
                molpair = allmolpair[i]
                x = Chem.MolFromSmiles(molpair[0][0])
                y = Chem.MolFromSmiles(molpair[0][1])   
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
                score_result_dict["chem_target"] = molpair[0][1]
                score_result_list.append(score_result_dict)
            score_result_list.sort(key = lambda x: x.get("bio_score"))

            # print(score_result_list) 
            commercial = False
            for item in score_result_list:
                if item["bio_route"][0][1] == True:
                    commercial = True

            return score_result_list, commercial


    # result = get_chemical_synthesis.get_synthesis(chem_target)
    # reactions = get_chemical_synthesis.reaction_list(result)

    class Searching_Algorithm():
        def extract_chem_intermediate(chem_target):
            success = False
            result = get_chemical_synthesis.get_synthesis(chem_target)
            reactions = get_chemical_synthesis.reaction_list(result)
            print("reactions",reactions)
            if [] in reactions:
                reactions.remove([])
            chem_route_pair = ChemScore.aiz_molpair_generator(reactions)
            scored_chem_routes, commercial = ChemScore.routescore(chem_route_pair)

            del_list = []
            for i in range(len(scored_chem_routes)):
                for reaction in scored_chem_routes[i]["route"]:
                    if reaction[0][0][0] == reaction[0][0][1]:
                        del_list.append(i)
            del_list = list(set(del_list))
            del_list.reverse()
            for item in del_list:
                try:
                    del scored_chem_routes[item]
                except IndexError:
                    continue
            count = 0
            # print(scored_chem_routes[count],scored_chem_routes[count+1])
            # while count <= len(scored_chem_routes):
            #     try:
            #         # print(scored_chem_routes[count],scored_chem_routes[count+1],abs(scored_chem_routes[count]["route_score"] - scored_chem_routes[count+1]["route_score"]))
            #         if abs(scored_chem_routes[count]["route_score"] - scored_chem_routes[count+1]["route_score"]) < 0.000001:
            #             del scored_chem_routes[count+1]
            #         else:
            #             count +=1
            #     except IndexError:
            #         break

            chem_success_list = []
            chem_failed_list = []
            for item in scored_chem_routes:
                for i in range(len(item["route"])):
                    if item["route"][i][0][1] == True:
                        chem_success_list.append(item)
            for item in scored_chem_routes:
                if item not in chem_success_list:
                    chem_failed_list.append(item)
            chem_success_dict = {}
            if len(chem_success_list) != 0:
                route_lenth = []
                for item in chem_success_list:
                    route_lenth.append(len(item["route"]))
                chem_success_dict["target_molecule"] = scored_chem_routes[0]["route"][0][0][0][0]
                chem_success_dict["shortest_route_lenth"] = min(route_lenth)
                chem_success_dict["chem_route_count"] = len(chem_success_list)
                chem_success_dict["all_route"] = chem_success_list
                if chem_success_dict["all_route"] != []:
                    success = True
                chem_success_dict["failed_routes"] = chem_failed_list
            else:
                chem_success_dict["target_molecule"] = chem_target
                chem_success_dict["all_route"] = []
                chem_success_dict["failed_routes"] = chem_failed_list
            
            chem_target_name = chem_target.replace("/", "_").replace("\\", "-")
            with open("results/"+chem_target_name+".txt", "w") as f:
                f.write("%s\n" % chem_success_dict)
                f.write("%s\n" % time_list)

            chem_routes_list = []
            for item in scored_chem_routes:
                chem_routes_dict = {}
                chem_routes = item["route"]
                score = item["route_score"]
                if len(chem_routes) == 1:
                    chem_routes_dict = {}
                    bio_target = chem_routes[0][0][0][1]
                    reserved_chem_score = score
                    reserved_chem_routes = chem_routes
                    chem_routes_dict["bio_target"] = bio_target
                    chem_routes_dict["reserved_chem_score"] = reserved_chem_score
                    chem_routes_dict["reserved_chem_routes"] = reserved_chem_routes
                    # chem_routes_dict["original_chem_route"] = chem_routes
                    # chem_routes_dict["original_whole_score"] = score
                    # if chem_routes_dict["reserved_chem_score"] != 0:
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
                    bio_target = chem_routes[place][0][0][0]
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
                try:
                    del chem_routes_list[item]
                except IndexError:
                    continue
            chem_routes_list.sort(key = lambda x: x.get("reserved_chem_score"))

            print(chem_routes_list)

            return(chem_routes_list, success)


        def bio2chem(chem_target):
            chembio_route_list_total = []
            hybrid_route_found = False
            chem_intermediate, success = Searching_Algorithm.extract_chem_intermediate(chem_target)
           
            # if success == True:
            #     pass
            # else:
            for item in chem_intermediate:
                bio_target = item["bio_target"]
                bio = get_biological_synthesis.get_retrobiocat_result(bio_target)
                biomolpair = BioScore.molpair_generator(bio) 
                bio_route_score, commercial = BioScore.routescore(biomolpair)
                if commercial == True:
                    print("commercial=true")
                    # break
                print("bio_route_score",bio_route_score)
                for i in range(len(bio_route_score)):
                    chembio_route_dict = item
                    output_dict = {}
                    chembio_route_dict["biostep"] = bio_route_score[i]["bio_route"]
                    chembio_route_dict["bioscore"] = bio_route_score[i]["bio_score"]
                    chembio_route_dict["chem_target"] = bio_route_score[i]["chem_target"]
                    # if chembio_route_dict["bioscore"] != 0:
                    output_dict["chem_target"] = chembio_route_dict["chem_target"]
                    output_dict["overall_routes"] = chembio_route_dict["reserved_chem_routes"][:] + [chembio_route_dict["biostep"]]
                    output_dict["overall_score"] = chembio_route_dict["reserved_chem_score"] + chembio_route_dict["bioscore"] 
                    chembio_route_list_total.append(output_dict)
            
            chembio_route_list_total.sort(key = lambda x: x.get("overall_score"))
            # routes_count = len(chembio_route_list_total)
            # chembio_route_list_total.append(routes_count)
            print(chembio_route_list_total)

            
            del_list = []
            for i in range(len(chembio_route_list_total)):
                for reaction in chembio_route_list_total[i]["overall_routes"]:
                    if reaction[0][0][0] == reaction[0][0][1]: 
                        del_list.append(i)
            del_list = list(set(del_list))
            del_list.reverse()
            for item in del_list:
                try:
                    del chembio_route_list_total[item]
                except IndexError:
                    continue

            chembio_success_list = []
            chembio_failed_list = []
            for item in chembio_route_list_total:
                for i in range(len(item["overall_routes"])):
                    if item["overall_routes"][i][0][1] == True:
                        chembio_success_list.append(item)
            for item in chembio_route_list_total:
                if item not in chembio_success_list:
                    chembio_failed_list.append(item)
            chembio_success_dict = {}
            if len(chembio_success_list) != 0:
                hybrid_route_found = True
                route_lenth = []
                for item in chembio_success_list:
                    route_lenth.append(len(item["overall_routes"]))
                chembio_success_dict["target_molecule"] = chembio_route_list_total[0]["overall_routes"][0][0][0][0]
                chembio_success_dict["shortest_route_lenth"] = min(route_lenth)
                chembio_success_dict["chembio_route_count"] = len(chembio_success_list)
                chembio_success_dict["all_route"] = chembio_success_list
                chembio_success_dict["failed_routes"] = chembio_failed_list
            else:
                hybrid_route_found = False
                chembio_success_dict["target_molecule"] = chem_target
                chembio_success_dict["all_route"] = []
                chembio_success_dict["failed_routes"] = chembio_failed_list
            
            chem_target_name = chem_target.replace("/", "_").replace("\\", "-")
            with open("results/"+chem_target_name+".txt", "a") as f:
                f.write("%s\n" % chembio_success_dict)
                f.write("%s\n" % time_list)

            return chembio_route_list_total, hybrid_route_found, success

        
        def wrap_up(chem_target):
            chembiochem_route_list_total = []
            bio_intermediate, hybrid_result, success = Searching_Algorithm.bio2chem(chem_target)
            
            if hybrid_result == True or success == True:
                pass
            else:
                for item in bio_intermediate:
                    chem_target_new = item["chem_target"]
                    #print(chem_target)
                    result = get_chemical_synthesis.get_synthesis(chem_target_new)
                    reactions = get_chemical_synthesis.reaction_list(result)
                    chemmolpair_new = ChemScore.aiz_molpair_generator(reactions) 
                    chem_route_score_new, commercial = ChemScore.routescore(chemmolpair_new)
                    
                    for i in range(len(chem_route_score_new)):
                        chembiochem_route_dict = {}
                        # chembiochem_route_dict["chem_next_step"] = chem_route_score_new[i]["route"]
                        # chembiochem_route_dict["chem_next_score"] = chem_route_score_new[i]["route_score"]
                        chembiochem_route_dict["chembiochem_overall_score"] = item["overall_score"] + chem_route_score_new[i]["route_score"]
                        chembiochem_route_dict["chembiochem_overall_routes"] = item["overall_routes"][:] + chem_route_score_new[i]["route"][:]
                        chembiochem_route_list_total.append(chembiochem_route_dict)

                    if commercial == True or sum(time_list) >= 180:
                        break

                chembiochem_route_list_total.sort(key = lambda x: x.get("chembiochem_overall_score"))
                
                print(chembiochem_route_list_total)   


                del_list = []
                for i in range(len(chembiochem_route_list_total)):
                    for reaction in chembiochem_route_list_total[i]["chembiochem_overall_routes"]:
                        if reaction[0][0][0] == reaction[0][0][1]:
                            del_list.append(i)
                del_list = list(set(del_list))
                del_list.reverse()
                for item in del_list:
                    try:
                        del chembiochem_route_list_total[item]
                    except IndexError:
                        continue
                

                chembiochem_success_list = []
                chembiochem_failed_list = []
                for item in chembiochem_route_list_total:
                    for i in range(len(item["chembiochem_overall_routes"])):
                        if item["chembiochem_overall_routes"][i][0][1] == True:
                            chembiochem_success_list.append(item)
                for item in chembiochem_route_list_total:
                    if item not in chembiochem_success_list:
                        chembiochem_failed_list.append(item)
                chembiochem_success_dict = {}
                if len(chembiochem_success_list) != 0:
                    route_lenth = []
                    for item in chembiochem_success_list:
                        route_lenth.append(len(item["chembiochem_overall_routes"]))
                    chembiochem_success_dict["target_molecule"] = chembiochem_route_list_total[0]["chembiochem_overall_routes"][0][0][0][0]
                    chembiochem_success_dict["shortest_route_lenth"] = min(route_lenth)
                    chembiochem_success_dict["chembiochem_route_count"] = len(chembiochem_success_list)
                    chembiochem_success_dict["all_route"] = chembiochem_success_list
                    chembiochem_success_dict["failed_routes"] = chembiochem_failed_list
                else:
                    chembiochem_success_dict["target_molecule"] = chem_target
                    chembiochem_success_dict["all_route"] = []
                    chembiochem_success_dict["failed_routes"] = chembiochem_failed_list
                
                chem_target_name = chem_target.replace("/", "_").replace("\\", "-")
                with open("results/"+chem_target_name+".txt", "a") as f:
                    f.write("%s\n" % chembiochem_success_dict)
                    f.write("%s\n" % time_list)                

                return chembiochem_route_list_total
            

    def main():
        # chem_target = 'CC(=O)[C@@](C)(O)C(=O)O'
        # unconcaed_chem_target = "FC1=CC=C(C2=CC(C(CCNC3)=C3N4)=C4C=C2)C=N1"
        # chem_target = Chem.CanonSmiles(unconcaed_chem_target) 
        # time.sleep(120)
        # chem_routes = get_synthesis_result.get_retrosynthesis_result(chem_target)
        # chem_route_pair = ChemScore.molpair_generator(chem_routes)
        # scored_chem_routes = ChemScore.routescore(chem_route_pair)
        # with open("results/"+chem_target+".txt", "w") as f:
        #     f.write("%s\n" % chem_dict_written)
        # time.sleep(120)
        # results = Searching_Algorithm.extract_chem_intermediate(chem_target)
        # results = Searching_Algorithm.wrap_up(chem_target)
        results = Searching_Algorithm.bio2chem(chem_target)

        # result = get_chemical_synthesis.get_synthesis(chem_target)
        # reactions = get_chemical_synthesis.reaction_list(result)
        # print(reactions)

        # bio = get_biological_synthesis.get_retrobiocat_result()
        # biomolpair = BioScore.molpair_generator(bio) 
        # bio_route_score, commercial = BioScore.routescore(biomolpair)
        # print(bio_route_score)
        # results = Searching_Algorithm.bio2chem(chem_target)



    main()

            

