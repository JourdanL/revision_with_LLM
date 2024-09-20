from transformers import AutoTokenizer, T5ForConditionalGeneration
import os
import json
import torch


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("coedit-xl")
print("tokenizer ok")
model = T5ForConditionalGeneration.from_pretrained("coedit-xl").to(device)
print("frompretrained ok")


#device = torch.device("cuda")
#model.cuda()
#model.to(device)
#print("to device ok")
#model.eval()
#print("chargement fini")

#input_text = 'Fix grammatical errors in this sentence: When I grow up, I start to understand what he said is quite right.'
#input_ids = tokenizer(input_text, return_tensors="pt").to(device).input_ids
#input_ids=input_ids.format("torch")#.to(device)
#outputs = model.generate(input_ids, max_length=256)
#edited_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#print(edited_text)

#TODO: MAJ avec mes labels
dict_intentions={"Concision":'Make this paragraph shorter',
                 "Content_deletion":'Remove unnecessary details',
                 "Rewritting_light":'Improve the English of this paragraph',
                 "Rewritting_medium":'Rewrite some sentences to make them more clear and easily readable',
                 "Rewritting_heavy":'Rewrite and reorganize the paragraph for better readability'}
#Improve the writing in this paragraph.
def revision_from_labels_approche1_separate(parag):
    a_retourner=[]
    for intention in parag[2]:
        # prepare input to the model: <intention> before_sent
        before_text = parag[1]
        edit_intent =dict_intentions[intention]
        input_text = edit_intent+": \""+before_text+"\""
        
        input_ids = tokenizer(input_text, return_tensors="pt").to(device).input_ids
        outputs = model.generate(input_ids, max_length=256)
        edited_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        a_retourner.append({"intention":intention,"revised_paragraph":edited_text})
    return {"id_paragraph":parag[0],"revisions":a_retourner,"type_approach":parag[3]}#a_retourner


def revision_from_labels_approche2_iterative(parag,dict_deja_fait):
    revisions=[]
    before_text = parag[1]
    past_intentions=[]
    for idx,intention in enumerate(parag[2],start=1):
        # prepare input to the model: <intention> before_sent
        if idx==1:
            edited_text=dict_deja_fait["1-"+intention]
            past_intentions.append(intention)
        elif idx==2:
            if "1-"+past_intentions[0]+"-2-"+intention in list(dict_deja_fait.keys()):
                edited_text=dict_deja_fait["1-"+past_intentions[0]+"-2-"+intention]
            else:
                edit_intent =dict_intentions[intention]
                input_text = edit_intent+": \""+before_text+"\""

                input_ids = tokenizer(input_text, return_tensors="pt").to(device).input_ids
                outputs = model.generate(input_ids, max_length=256)
                edited_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                dict_deja_fait["1-"+past_intentions[0]+"-2-"+intention]=edited_text
                past_intentions.append(intention)
        elif idx==3:
            if "1-"+past_intentions[0]+"-2-"+past_intentions[1]+"-3-"+intention in list(dict_deja_fait.keys()):
                edited_text=dict_deja_fait["1-"+past_intentions[0]+"-2-"+past_intentions[1]+"-3-"+intention]
            else:
                edit_intent =dict_intentions[intention]
                input_text = edit_intent+": \""+before_text+"\""

                input_ids = tokenizer(input_text, return_tensors="pt").to(device).input_ids
                outputs = model.generate(input_ids, max_length=256)
                edited_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                dict_deja_fait["1-"+past_intentions[0]+"-2-"+past_intentions[1]+"-3-"+intention]=edited_text
                past_intentions.append(intention)
        elif idx==4:
            if "1-"+past_intentions[0]+"-2-"+past_intentions[1]+"-3-"+past_intentions[2]+"-4-"+intention in list(dict_deja_fait.keys()):
                edited_text=dict_deja_fait["1-"+past_intentions[0]+"-2-"+past_intentions[1]+"-3-"+past_intentions[2]+"-4-"+intention]
            else:
                edit_intent =dict_intentions[intention]
                input_text = edit_intent+": \""+before_text+"\""

                input_ids = tokenizer(input_text, return_tensors="pt").to(device).input_ids
                outputs = model.generate(input_ids, max_length=256)
                edited_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                dict_deja_fait["1-"+past_intentions[0]+"-2-"+past_intentions[1]+"-3-"+past_intentions[2]+"-4-"+intention]=edited_text
                past_intentions.append(intention)
        else:
            print("Probleme")
        
        before_text = edited_text
        
        revisions.append({"depth": idx,"intention":intention, "revised_paragraph": edited_text})
    return {"id_paragraph":parag[0],"revisions":revisions,"type_approach":parag[3]},dict_deja_fait

def revision_from_instructions(parag):
    instruction=parag[2]
    # prepare input to the model: <intention> before_sent
    before_text = parag[1]
    input_text = instruction+": \""+before_text+"\""

    input_ids = tokenizer(input_text, return_tensors="pt").to(device).input_ids
    outputs = model.generate(input_ids, max_length=256)
    edited_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #edited_text
        
    return {"id_paragraph":parag[0],"instruction":instruction,"revised_paragraph":edited_text,"type_approach":parag[3]}

path_json="../revision_parag/data_paparev/" #TODO
filename="aditionnal_parags.jsonl"
path_sortie="../revision_parag/predictions_coeditxl_rev_parag/" #TODO


def get_list_inputs(diff_parags):
    liste_data_inputs=[]
    for element in diff_parags:
        if ("annot_1" in element) and ("annot_2" in element):
            if (type(element["annot_1"]["instruction"])==list) and (type(element["annot_2"]["instruction"])==list):
                annot1=element["annot_1"]["annotation"]
                annot1=[label for label in annot1 if label in list(dict_intentions.keys())]
                annot2=element["annot_2"]["annotation"]
                annot2=[label for label in annot2 if label in list(dict_intentions.keys())]
                #print(element["annot_1"]["instruction"])
                #print(element["annot_2"]["instruction"])
                insts={"annot_1":' '.join(element["annot_1"]["instruction"]),"annot_2":' '.join(element["annot_2"]["instruction"])}
                intersect_annot=list(set(annot1).intersection(set(annot2)))
                union_annot=list(set(annot1).union(set(annot2)))
                #if intersect_annot!=union_annot:
                dict_var_set_intentions={"union":union_annot,"intersection":intersect_annot}
                #if intersect_annot!=annot1:
                dict_var_set_intentions["annot_1"]=annot1
                #if intersect_annot!=annot2:
                dict_var_set_intentions["annot_2"]=annot2
                #else:
                #    dict_var_set_intentions={"union":union_annot}
                liste_data_inputs.append((element['id_paragraph'],element['parag_1'],dict_var_set_intentions,insts))
    return liste_data_inputs

with open(path_json+filename, 'r') as parags_file:
    diff_parags=[json.loads(line.strip('\n')) for line in parags_file]
    #for line in parags_file:
    #    print(line)
    #    json.loads(line.strip('\n'))
print("Longueur totale:",len(diff_parags))
liste_data_inputs=get_list_inputs(diff_parags)

print("Longueur après:",len(liste_data_inputs))
with open(path_sortie+"predict_"+filename, 'w') as file_sortie:  
    for idx in range(len(liste_data_inputs)):
        #try:
            id_parag=liste_data_inputs[idx][0]
            #SEPARATE
            #union
            result=revision_from_labels_approche1_separate((id_parag,liste_data_inputs[idx][1],liste_data_inputs[idx][2]["union"],"separate-labels-union"))
            dict_deja_gen={"1-"+gen["intention"]:gen["revised_paragraph"] for gen in result["revisions"]}
            json.dump(result,file_sortie)
            file_sortie.write('\n')
            #intersection
            result={"id_paragraph":id_parag,"revisions":[{"intention":intention,"revised_paragraph":dict_deja_gen["1-"+intention]} for intention in liste_data_inputs[idx][2]["intersection"]],"type_approach":"separate-labels-intersection"}
            json.dump(result,file_sortie)
            file_sortie.write('\n')
            #annot_1
            result={"id_paragraph":id_parag,"revisions":[{"intention":intention,"revised_paragraph":dict_deja_gen["1-"+intention]} for intention in liste_data_inputs[idx][2]["annot_1"]],"type_approach":"separate-labels-annot_1"}
            json.dump(result,file_sortie)
            file_sortie.write('\n')
            #annot_2
            result={"id_paragraph":id_parag,"revisions":[{"intention":intention,"revised_paragraph":dict_deja_gen["1-"+intention]} for intention in liste_data_inputs[idx][2]["annot_2"]],"type_approach":"separate-labels-annot_2"}
            json.dump(result,file_sortie)
            file_sortie.write('\n')

            #ITERATIVE
            #for approche,labels in liste_data_inputs[idx][2].items():
            #    result=revision_from_labels_approche1_separate((liste_data_inputs[idx][0],liste_data_inputs[idx][1],labels,"separate-labels-"+approche))
            #    json.dump(result,file_sortie)
            #    file_sortie.write('\n')
            for approche,labels in liste_data_inputs[idx][2].items():
                result,dict_deja_gen=revision_from_labels_approche2_iterative((liste_data_inputs[idx][0],liste_data_inputs[idx][1],labels,"iterative-labels-"+approche),dict_deja_gen)
                json.dump(result,file_sortie)
                file_sortie.write('\n')
            for approche,inst in liste_data_inputs[idx][3].items():
                result=revision_from_instructions((liste_data_inputs[idx][0],liste_data_inputs[idx][1],inst,"instruction-"+approche))
                json.dump(result,file_sortie)
                file_sortie.write('\n')
        #except:
         #   print("Erreur:",filename)
          #  pass
                 