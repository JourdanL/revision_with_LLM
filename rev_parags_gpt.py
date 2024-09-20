from openai import OpenAI
import json
import os.path
client = OpenAI()



path_json=""
#filename="aditionnal_parags.jsonl"
filename="full_manual_annot_list_by_parag.jsonl"
path_sortie="rev_with_gpt/"

def get_list_inputs(diff_parags):
    liste_data_inputs=[]
    for element in diff_parags:
        if ("annot_1" in element) and ("annot_2" in element):
            if (type(element["annot_1"]["instruction"])==list) and (type(element["annot_2"]["instruction"])==list):
                insts={"annot_1":' '.join(element["annot_1"]["instruction"]),"annot_2":' '.join(element["annot_2"]["instruction"])}

                liste_data_inputs.append((element['id_paragraph'],element['parag_1'],insts))
    return liste_data_inputs

def generate_revision(messages_history,parag):

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=len(parag)+50,
        messages=messages_history
    )
    reponse=completion.choices[0].message.content
    return reponse

def fill_pattern_syst():
    input_text= f"""You are a writting assistant specialised in academic writing. Your task is to revise the paragraph from a research paper draft that will be given according to the user's instructions.           
Please answer only by \"Revised paragaraph: <revised_version_of_the_paragraph>\""""
     
    return input_text

def fill_pattern_user(instruct,parag):
    input_text= f"""{instruct} : \"{parag}\""""
     
    return input_text

def revision_from_instructions(parag):
    instruction=parag[2]
    before_text = parag[1]
    messages = [
        {"role": "system", "content": fill_pattern_syst()},
        {"role": "user", "content": fill_pattern_user(instruction,before_text)}]
    edited_text=generate_revision(messages,before_text)  
    return {"id_paragraph":parag[0],"instruction":instruction,"revised_paragraph":edited_text,"type_approach":parag[3]}



#Check already imported data
if os.path.isfile(path_sortie+"predict_gpt"+filename):
    with open(path_sortie+"predict_gpt"+filename, 'r') as done_file:
       liste_revisions=[json.loads(line.strip('\n')) for line in done_file] 
       already_done=set([revision["id_paragraph"]for revision in liste_revisions]) 
else:
    already_done=set()


with open(path_json+filename, 'r') as parags_file:
    diff_parags=[json.loads(line.strip('\n')) for line in parags_file]
print("Longueur totale:",len(diff_parags))
liste_data_inputs=get_list_inputs(diff_parags)
print("Longueur après:",len(liste_data_inputs))

for idx in range(len(liste_data_inputs)):
    id_parag=liste_data_inputs[idx][0]
    if id_parag not in already_done:
        file_sortie=open(path_sortie+"predict_gpt"+filename, 'a') 

        for approche,inst in liste_data_inputs[idx][2].items():
            result=revision_from_instructions((liste_data_inputs[idx][0],liste_data_inputs[idx][1],inst,"instruction-"+approche))
            print(result)
            json.dump(result,file_sortie)
            file_sortie.write('\n')
        file_sortie.close()

