#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 17:23:49 2026

@author: steeltitanunbrk
"""

from utils import *
import urllib.request
import urllib.error
import asyncio

import json


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
                
            if file.split('_')[2] == '001873':
                batches = get_columns_no_tag(file)
            else:
                batches = get_columns(file)
            

            result = []
            error = []

            result = asyncio.run(process_all_columns(
                batches, PROMPT, exemple_CSV,
                MODEL or get_model_name(), VLLM_URL, API_KEY, OUTPUT_FORMAT,
                max_concurrent=4
            ))


            # SAUVEGARDER LE FICHIER CSV après validation du format CSV
            if SAUVEGARDE_CSV:

                df = pd.DataFrame(result).reset_index(drop=True)
                
                # Remettre les titres à la fin des noms
                df['NOM'] = df['NOM'].apply(lambda x: correct_names_post(x))
                
                df.to_csv(DEST_EXTRACT_DIR +
                          all_text_file_names[No]+'.csv', index=False)

            elif SAUVEGARDE_EXCEL:
                df = pd.DataFrame(result).reset_index(drop=True)

                df.to_excel(
                    all_excel_files[No], sheet_name=all_text_file_names[No+START], engine="openpyxl", index=False)
                
    print("\n\n##### FIN DE L'EXTRACTION ######")