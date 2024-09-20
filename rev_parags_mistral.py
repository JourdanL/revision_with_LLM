from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import os
import json
import torch
print("lancement")
device = "cuda" # the device to load the model onto
print("Device:",device)

root_path = os.environ['DSDIR'] + '/HuggingFace_Models'
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

print("model_path:",root_path+'/'+model_name)
model = AutoModelForCausalLM.from_pretrained(root_path+'/'+model_name)
print("chargement model ok")
tokenizer = AutoTokenizer.from_pretrained(root_path+'/'+model_name)
print("chargement tokenizer ok")

print("chargement fini")

dict_intentions={"Concision":'Make this paragraph shorter',
                 "Content_deletion":'Remove unnecessary details',
                 "Rewritting_light":'Improve the English of this paragraph',
                 "Rewritting_medium":'Rewrite some sentences to make them more clear and easily readable',
                 "Rewritting_heavy":'Rewrite and reorganize the paragraph for better readability'}
#Improve the writing in this paragraph.



def generate_revision(messages,parag):
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    model.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=len(parag)+100, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    reponse=decoded[0]
    reponse=reponse.split("[/INST]")[-1]
    if reponse[-4:]=="</s>":
        reponse=reponse[:-4]
    return reponse

def fill_pattern(instruct,parag):
    input_text= f"""You are a writting assistant specialised in academic writing. Your task is to revise the paragraph from a research paper draft that will be given according to the user's instructions.           
Please answer only by \"Revised paragaraph: <revised_version_of_the_paragraph>\"

{instruct} : \"{parag}\""""
     
    return input_text


def revision_from_labels_approche1_separate(parag):
    a_retourner=[]
    for intention in parag[2]:
        before_text = parag[1]
        edit_intent =dict_intentions[intention]
        messages = [
        {"role": "user", "content": fill_pattern(edit_intent,before_text)}]
        edited_text=generate_revision(messages,before_text)
        
        a_retourner.append({"intention":intention,"revised_paragraph":edited_text})
    return {"id_paragraph":parag[0],"revisions":a_retourner,"type_approach":parag[3]}


def revision_from_labels_approche2_iterative(parag,dict_deja_fait):
    revisions=[]
    before_text = parag[1]
    past_intentions=[]
    for idx,intention in enumerate(parag[2],start=1):
        if idx==1:
            edited_text=dict_deja_fait["1-"+intention]
            past_intentions.append(intention)
        elif idx==2:
            if "1-"+past_intentions[0]+"-2-"+intention in list(dict_deja_fait.keys()):
                edited_text=dict_deja_fait["1-"+past_intentions[0]+"-2-"+intention]
            else:
                edit_intent =dict_intentions[intention]
                
                messages = [
                {"role": "user", "content": fill_pattern(edit_intent,before_text)}]

                edited_text=generate_revision(messages,before_text)

                dict_deja_fait["1-"+past_intentions[0]+"-2-"+intention]=edited_text
                past_intentions.append(intention)
        elif idx==3:
            if "1-"+past_intentions[0]+"-2-"+past_intentions[1]+"-3-"+intention in list(dict_deja_fait.keys()):
                edited_text=dict_deja_fait["1-"+past_intentions[0]+"-2-"+past_intentions[1]+"-3-"+intention]
            else:
                edit_intent =dict_intentions[intention]
                messages = [
                {"role": "user", "content": fill_pattern(edit_intent,before_text)}]
                edited_text=generate_revision(messages,before_text)

                dict_deja_fait["1-"+past_intentions[0]+"-2-"+past_intentions[1]+"-3-"+intention]=edited_text
                past_intentions.append(intention)
        elif idx==4:
            if "1-"+past_intentions[0]+"-2-"+past_intentions[1]+"-3-"+past_intentions[2]+"-4-"+intention in list(dict_deja_fait.keys()):
                edited_text=dict_deja_fait["1-"+past_intentions[0]+"-2-"+past_intentions[1]+"-3-"+past_intentions[2]+"-4-"+intention]
            else:
                edit_intent =dict_intentions[intention]
                messages = [
                {"role": "user", "content": fill_pattern(edit_intent,before_text)}]
                edited_text=generate_revision(messages,before_text)

                dict_deja_fait["1-"+past_intentions[0]+"-2-"+past_intentions[1]+"-3-"+past_intentions[2]+"-4-"+intention]=edited_text
                past_intentions.append(intention)                
        else:
            print("Probleme")
        
        before_text = edited_text
        
        revisions.append({"depth": idx,"intention":intention, "revised_paragraph": edited_text})
        
    return {"id_paragraph":parag[0],"revisions":revisions,"type_approach":parag[3]},dict_deja_fait

def revision_from_instructions(parag):
    instruction=parag[2]
    before_text = parag[1]
    messages = [
        {"role": "user", "content": fill_pattern(instruction,before_text)}]
    edited_text=generate_revision(messages,before_text)  
    return {"id_paragraph":parag[0],"instruction":instruction,"revised_paragraph":edited_text,"type_approach":parag[3]}

path_json="data_paparev/"
filename="aditionnal_parags.jsonl"
path_sortie="predictions_mistral_rev_parag/"


def get_list_inputs(diff_parags):
    liste_data_inputs=[]
    for element in diff_parags:
        if ("annot_1" in element) and ("annot_2" in element):
            if (type(element["annot_1"]["instruction"])==list) and (type(element["annot_2"]["instruction"])==list):
                annot1=element["annot_1"]["annotation"]
                annot1=[label for label in annot1 if label in list(dict_intentions.keys())]
                annot2=element["annot_2"]["annotation"]
                annot2=[label for label in annot2 if label in list(dict_intentions.keys())]
                insts={"annot_1":' '.join(element["annot_1"]["instruction"]),"annot_2":' '.join(element["annot_2"]["instruction"])}
                intersect_annot=list(set(annot1).intersection(set(annot2)))
                union_annot=list(set(annot1).union(set(annot2)))
                dict_var_set_intentions={"union":union_annot,"intersection":intersect_annot}
                dict_var_set_intentions["annot_1"]=annot1
                dict_var_set_intentions["annot_2"]=annot2
                liste_data_inputs.append((element['id_paragraph'],element['parag_1'],dict_var_set_intentions,insts))
    return liste_data_inputs

with open(path_json+filename, 'r') as parags_file:
    diff_parags=[json.loads(line.strip('\n')) for line in parags_file]
print("Longueur totale:",len(diff_parags))
liste_data_inputs=get_list_inputs(diff_parags)

print("Longueur après:",len(liste_data_inputs))
for idx in range(len(liste_data_inputs)):
        file_sortie= open(path_sortie+"predict_mistral"+filename, 'a')  
    #try:
        id_parag=liste_data_inputs[idx][0]
        #SEPARATE
        #union
        result=revision_from_labels_approche1_separate((id_parag,liste_data_inputs[idx][1],liste_data_inputs[idx][2]["union"],"separate-labels-union"))
        dict_deja_gen={"1-"+gen["intention"]:gen["revised_paragraph"] for gen in result["revisions"]}
        print(result)
        json.dump(result,file_sortie)
        file_sortie.write('\n')
        #intersection
        result={"id_paragraph":id_parag,"revisions":[{"intention":intention,"revised_paragraph":dict_deja_gen["1-"+intention]} for intention in liste_data_inputs[idx][2]["intersection"]],"type_approach":"separate-labels-intersection"}
        print(result)
        json.dump(result,file_sortie)
        file_sortie.write('\n')
        #annot_1
        result={"id_paragraph":id_parag,"revisions":[{"intention":intention,"revised_paragraph":dict_deja_gen["1-"+intention]} for intention in liste_data_inputs[idx][2]["annot_1"]],"type_approach":"separate-labels-annot_1"}
        print(result)
        json.dump(result,file_sortie)
        file_sortie.write('\n')
        #annot_2
        result={"id_paragraph":id_parag,"revisions":[{"intention":intention,"revised_paragraph":dict_deja_gen["1-"+intention]} for intention in liste_data_inputs[idx][2]["annot_2"]],"type_approach":"separate-labels-annot_2"}
        print(result)
        json.dump(result,file_sortie)
        file_sortie.write('\n')
        #ITERATIVE
        for approche,labels in liste_data_inputs[idx][2].items():
            result,dict_deja_gen=revision_from_labels_approche2_iterative((liste_data_inputs[idx][0],liste_data_inputs[idx][1],labels,"iterative-labels-"+approche),dict_deja_gen)
            print(result)
            json.dump(result,file_sortie)
            file_sortie.write('\n')
        for approche,inst in liste_data_inputs[idx][3].items():
            result=revision_from_instructions((liste_data_inputs[idx][0],liste_data_inputs[idx][1],inst,"instruction-"+approche))
            print(result)
            json.dump(result,file_sortie)
            file_sortie.write('\n')
        file_sortie.close()
    #except:
     #   print("Erreur:",filename)
      #  pass
#exec sur tout les donnees