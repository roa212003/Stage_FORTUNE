#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 17:23:49 2026

@author: steeltitanunbrk
"""

import pandas as pd
from tqdm import tqdm

from utils import *
import asyncio
from pathlib import Path


import nest_asyncio
nest_asyncio.apply()

TEST_MODE=False
SAUVEGARDE_CSV = True
SAUVEGARDE_EXCEL = False

if __name__ == '__main__':

    # Chargement du fichier exemple
    with open(EXAMPLE_FILE, "r") as fichier:
        exemple_CSV = fichier.read()


    if TEST_MODE:
        # for No,file in enumerate(all_text_files[START:END]):
        # Marquer le début du traitement du fichier dans le log
        with open(MODEL_RES_LOG_FILE, "a") as log_file:
            log_file.write("######## TEST MODE : \n")

        # Récupration des colonnes
        batches = get_columns(TEST_INPUT_FILE)

        result = []
        error = []

        # Lancement
        result = asyncio.run(process_all_columns(
            batches, PROMPT, exemple_CSV,
            MODEL or get_model_name(), VLLM_URL, API_KEY, OUTPUT_FORMAT,
            max_concurrent=4
        ))


        pred_df = pd.DataFrame(result).reset_index(drop=True)
        pred_df.to_csv("pred.csv", index=False)

        true_df = pd.read_csv(TEST_OUTPUT_FILE, index_col=False).reset_index(drop=True)

        global_error, errors, errors_lines = evaluate(pred_df=pred_df, true_df=true_df)

        results_display(global_error=global_error,
                        errors=errors, nbr_entries=len(pred_df))

        # diff_df = true_df.compare(
        #     pred_df, result_names=("TRUE", "PRED"), align_axis=0)
        
        with open(TEST_DIFF_FILE, mode="w", encoding="utf-8") as fichier:
            fields = ["NOM & PRÉNOMS", "PROFESSION & COMPLÉMENTS", "NOM DE VOIE", "NUMÉRO DE VOIE",
                      "FAUBOURG", "PROPRIÉTAIRE", "NOM DE JEUNE FILLE", "SITUATION", "ADRESSE PROFESSIONNELLE"]
            for err in errors_lines:
                if err[1] < 1:
                    fichier.write(f"Ligne {err[0]+2} & Colonne {fields[err[1]]}\n")
                    fichier.write(f"TRUE : {true_df.iloc[err[0], err[1]]}\n")
                    fichier.write(f"PRED : {pred_df.iloc[err[0], err[1]]}")
                    fichier.write("\n\n")
                elif err[1] == 1:
                    fichier.write(f"Ligne {err[0]+2} & Colonne {fields[err[1]]}\n")
                    fichier.write(
                        f"TRUE : {true_df.iloc[err[0], err[1]+1]} {true_df.iloc[err[0], err[1]+2]}\n")
                    fichier.write(
                        f"PRED : {pred_df.iloc[err[0], err[1]+1]} {pred_df.iloc[err[0], err[1]+2]}")
                    fichier.write("\n\n")
                else:
                    fichier.write(f"Ligne {err[0]+2} & Colonne {fields[err[1]]}\n")
                    fichier.write(f"TRUE : {true_df.iloc[err[0], err[1]+2]}\n")
                    fichier.write(f"PRED : {pred_df.iloc[err[0], err[1]+2]}")
                    fichier.write("\n\n")

        print("Les différences sont enregistrées dans le fichiers diff.txt")
    else:
        # boucle de traitement sur tous les fichiers de l'année sélectionnée
        for No, file in tqdm(enumerate(all_text_files),
                             desc="Traitement des fichiers",
                             total=len(all_text_files),
                             ncols=80,
                             unit="fichiers",
                             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                             "[{elapsed}<{remaining}, {rate_fmt}]"
                             ):
            # Marquer le début du traitement du fichier dans le log
            with open(MODEL_RES_LOG_FILE, "a") as log_file:
                log_file.write("######## Traitement du fichier :"+file+'\n')
                
            batches = []
            
            year = file.split('_')[2][2:]
            output_path =  DEST_EXTRACT_DIR+year+'/'
            
            output_dir = Path(output_path)
            output_dir.mkdir(exist_ok=True)

            csv_dir = Path(output_path+'csv/')
            csv_dir.mkdir(exist_ok=True)

            txt_dir = Path(output_path+'txt/')
            txt_dir.mkdir(exist_ok=True)
            
                
            if file.split('_')[2] in ['001873', '001874','001875', '001895']:
                batches = get_columns_no_tag(file)
            else:
                batches = get_columns(file)

            if len(batches) == 0:
                continue
            

            result = []
            error = []

            result, texts = asyncio.run(process_all_columns(
                batches, PROMPT, exemple_CSV,
                MODEL or get_model_name(), VLLM_URL, API_KEY, OUTPUT_FORMAT,
                max_concurrent=4
            ))

            with open(str(txt_dir) + '/' + all_text_file_names[No]+ '.txt', 'w') as file:
                for text in texts:
                    file.write(text+'\n')


            # SAUVEGARDER LE FICHIER CSV après validation du format CSV
            if SAUVEGARDE_CSV:

                df = pd.DataFrame(result).reset_index(drop=True)
                
                # Remettre les titres à la fin des noms
                df['NOM'] = df['NOM'].apply(lambda x: correct_names_post(x))
                
                df.to_csv(str(csv_dir) + '/' +
                          all_text_file_names[No]+'.csv', index=False)
                
    print("\n\n##### FIN DE L'EXTRACTION ######")


def worker_extraction(queue_in, queue_out):
    print("[Step 1] Module d'extraction prêt et à l'écoute...")
    
    # Chargement du fichier exemple (fait une seule fois au démarrage)
    with open(EXAMPLE_FILE, "r") as fichier:
        exemple_CSV = fichier.read()

    while True:
        # 1. Attend qu'un fichier soit disponible dans la queue (bloquant)
        file = queue_in.get()
        
        # Récupération du chemin du fichier à traiter
        file_name = os.path.basename(file)
        
        print(f"[Step 1] Fichier reçu pour extraction LLM : {file_name}")

        # 2. Marquer le début du traitement du fichier dans le log
        with open(MODEL_RES_LOG_FILE, "a") as log_file:
            log_file.write(f"[Step 1] Début traitement du fichier : {file}\n")
            
        batches = []
        
        # Extraction de l'année pour tes dossiers spécifiques
        # Exemple basé sur ton code : 'nom_fichier_001873_abc.txt' -> '73'
        year = file.split('_')[2][2:]
        output_path = DEST_EXTRACT_DIR + year + '/'
        
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)

        csv_dir = Path(output_path + 'csv/')
        csv_dir.mkdir(exist_ok=True)

        txt_dir = Path(output_path + 'txt/')
        txt_dir.mkdir(exist_ok=True)
        
        # Sélection du parser selon l'année historique (registres allemands)
        # if file.split('_')[2] in ['001873', '001874', '001875', '001895']:
        #     batches = get_columns_no_tag(file)
        # else:
        batches = get_columns(file)

        if len(batches) == 0:
            batches = get_columns_no_tag(file)
            if len(batches) == 0:
                # queue_in.task_done()
                continue
        
        result = []
        texts = []

        try:
            result, texts = asyncio.run(process_all_columns(
                batches, PROMPT, exemple_CSV,
                MODEL or get_model_name(), VLLM_URL, API_KEY, OUTPUT_FORMAT,queue_in=queue_in,
                max_concurrent=4
            ))
        except Exception as e:
            print(f"[Step 1] [ERREUR API] Impossible de traiter {file_name} : {e}")
            queue_in.put(file)
            q# ueue_in.task_done()
            continue

        # Sauvegarde du fichier TXT brut extrait
        # Utilisation de Path pour éviter les erreurs de slashes
        nom_base, _ = os.path.splitext(file_name)
        chemin_txt_final = txt_dir / f"{nom_base}.txt"
        with open(chemin_txt_final, 'w', encoding='utf-8') as f_out:
            for text in texts:
                f_out.write(text + '\n')

        # 4. Traitement et sauvegarde du CSV intermédiaire
        chemin_csv_final = csv_dir / f"{nom_base}.csv"
        
        if SAUVEGARDE_CSV:
            df = pd.DataFrame(result).reset_index(drop=True)
            
            # Application de ta fonction de post-processing sur les noms
            if 'NOM' in df.columns:
                df['NOM'] = df['NOM'].apply(lambda x: correct_names_post(x))
            
            df.to_csv(chemin_csv_final, index=False)
            print(f"[Step 1] Fin de l'extraction du fichier : {file_name}\n")
        
        
        queue_out.put(str(chemin_csv_final))
        os.remove(file)