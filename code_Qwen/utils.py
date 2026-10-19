#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 17:24:36 2026

@author: steeltitanunbrk
"""

import glob
import pandas as pd
import re
from Levenshtein import distance
import numpy as np
from tqdm import tqdm
import os
from dotenv import load_dotenv
import json
import urllib.request
import urllib.error
from itertools import islice
from rapidfuzz import fuzz
from tqdm.asyncio import tqdm_asyncio
import aiohttp
import nest_asyncio
import asyncio

nest_asyncio.apply()
load_dotenv()


EXAMPLE_FILE = "Exemple-CSV_jupyter.txt"
MODEL_RES_LOG_FILE = "log.txt"
TEST_INPUT_FILE = "test_inputs.txt"
TEST_OUTPUT_FILE = "test_outputs.csv"
LINE_ERRORS_FILE = "line_errors.txt"
TEST_DIFF_FILE = "diff.txt"


DEST_TEXT_DIR = "echantillon/text/"
DEST_EXTRACT_DIR = "echantillon/csv/1878/"


# récup les chemins complets des fichiers text
all_text_files = sorted(glob.glob(DEST_TEXT_DIR+'*.txt'))
# récup les noms des fichiers
all_text_files = [x[x.rfind('/')+1:] for x in all_text_files]
# all_text_file_names = [x.replace('.txt','') for x in all_text_files if x[x.rfind('_')+1:] in interval]  # création des noms des fichiers
all_text_file_names = [x.replace(DEST_TEXT_DIR, '').replace(
    '.txt', '') for x in all_text_files]
# reconstitution des noms des fichiers
all_text_files = [DEST_TEXT_DIR+x+'.txt' for x in all_text_file_names]


VLLM_URL = os.getenv("VLLM_URL")
API_KEY = os.getenv("VLLM_API_KEY")

MODEL = os.getenv("VLLM_MODEL", "")


OUTPUT_FORMAT = {
    "name": "person_list",
    "schema":
    {
        'type': 'object',
        'properties': {
            'result': {
                "type": "array",
                "items":
                {
                    "type": "object",
                    "properties": {
                        "NOM": {"type": "string"},
                        "PRENOM": {"type": "string"},
                        "PROFESSION": {"type": "string"},
                        "COMPLEMENT DE PROFESSION": {"type": "string"},
                        "NOM DE VOIE": {"type": "string"},
                        "NUMERO DE VOIE": {"type": "string"},
                        "FAUBOURG": {"type": "string"},
                        "PROPRIETAIRE": {"type": "string"},
                        "NOM DE JEUNE FILLE": {"type": "string"},
                        "SITUATION": {"type": "string"},
                        "ADRESSE PROFESSIONNELLE": {"type": "string"}
                    },
                    "required": [
                        "NOM",
                        "PRENOM",
                        "PROFESSION",
                        "COMPLEMENT DE PROFESSION",
                        "NOM DE VOIE",
                        "NUMERO DE VOIE",
                        "FAUBOURG",
                        "PROPRIETAIRE",
                        "NOM DE JEUNE FILLE",
                        "SITUATION",
                        "ADRESSE PROFESSIONNELLE"
                    ]
                },
            },
        },
        'required': ['result']
    }
}


PROMPT = """"
Tu es un extracteur de données, spécialisé dans les annuaires allemands
Voici un annuaire allemand. Extraire EXACTEMENT toutes les champs suivants sous format CSV : NOM,PRENOM,PROFESSION,COMPLEMENT DE PROFESSION,NOM DE VOIE,NUMERO DE VOIE,FAUBOURG,PROPRIETAIRE,NOM DE JEUNE FILLE,SITUATION,ADRESSE PROFESSIONNELLE

champ NOM: 
- lorsque la ligne débute par un tiret, caractère '-' ou '–', le NOM est celui de la ligne précédente.
- 2 entrées sur la même ligne ont le même NOM si le nom de la seconde entrée n'est pas précisé ;
- parfois précédé ou suivi de titre 'von', 'de', 'v.', 'Vater', 'senior', 'junior', 'freiherr', 'frhr.', 'freih.', à déplacer à la fin du champ NOM
PRENOM: 
- il s'agit du prénom de la personne
- peut être absent
- sans virgule
- '*' si absent
PROFESSION:  
- mettre à 'Sans' si absent
- Ne pas confondre avec le champ SITUATION ;
- conserver les valeurs 'Tagner','Tagnerin','Rentnerin','Rentner'.
COMPLEMENT DE PROFESSION
- contient toute précision sur la profession différente de l'adresse professionnelle
NOM DE VOIE: 
- adresse réelle en Allemagne
- il est uniquement constitué de mots, qui peuvent être séparés par des caractères '-' ou '—' ;
- la valeur du champ peut être préfixée par 'a.' ou 'n.' ; ces préfixes doivent être ajoutés au champ;
- ne pas confondre avec le FAUBOURG
- ne doit pas être ignorer
- parfois précisé avec les clés Wohnung ou Wohn.
NUMERO DE VOIE: 
- il est alphanumérique;
- copier VERBATIM
FAUBOURG
- Faubourg dans lequel se trouve la personne
- valeurs possibles : Neudorf, Neuhof, Ruprechtsau, Grüneberg, Königshofen, Kronenburg, Königsh., Rupr., Neuhof
- ne pas confondre avec le NOM DE VOIE
- uniquement les valeurs possibles
NOM DE JEUNE FILLE: 
- il apparait après la clé 'geb.' ou 'g.' ou 'gb.'. 
- Si la clé est absente la valeur est '*'
SITUATION: 
- indique si veuve ou célibataire
- le champ est mis EXACTEMENT à la valeur 'Ww.' UNIQUEMENT si la clé 'Wittve' ou 'Ww.' ou 'Wtw.' ou 'Wwe' apparait ou ledig.
PROPRIETAIRE: 
- Les valeurs possibles sont: E., I., II., III. IV., V., D., H. 
ADRESSE PROFESSIONNELLE: 
- adresse réelle en Allemagne
- adresse du travail de la personne, boutique ou magazin
- précisée avec les clés Geschäft ou Geshäftslokal ou Laden ou ldn.
- elle est présente lorsque la clé 'laden' ou 'ldn' apparaît ou elle est précisée entre parenthèses dans le champ PROFESSION ou après le champ VOIE ou le champ NUMERO DE VOIE. 
- peut être précisé comme adresse dans le champ PROFESSION  avec 'in der'
REMARQUES GENERALES
- garder le NUMÉRO DE VOIE si le NOM DE VOIE est absent
- garder le NUMÉRO DE VOIE si le FAUBOURG est présent
- certaines entrées s'étendent sur 2 lignes
- 2 entrées sur la même ligne ont le même NOM si le nom de la seconde entrée n'est pas précisé ;
- ne pas confondre FAUBOURG et NOM DE VOIE 
- ne pas ignorer le NUMÉRO DE VOIE si NOM DE VOIE est absent
- retirer les virgules à l'intérieur des champs
- chaque valeur de champ doit correspondre au sens défini dans les règles
- extraire OBLIGATOIREMENT le FAUBOURG si présent
- ne pas ignorer le NUMÉRO DE VOIE en présence du FAUBOURG
- les valeurs de chaque champ doivent correspondre à la définition du champ donnée ci-dessus ;
- si le champ PROFRESSION est absent, sa valeur est 'Sans' ;
- si le premier champ d'une ligne ne correspond pas à un nom, alors il faut concaténer cette ligne avec la ligne précédente pour formuler une seule entrée à extraire.
- les valeurs doivent être recopiées VERBATIM sans corrections ou développement
- si un champ est absent, sa valeur est mise à '*', sauf le champ PROFESSION qui est mis à 'Sans'
- Neuhof n'est pas un NOM DE VOIE mais un FAUBOURG
- Rupr. est un FAUBOURG
- ne pas garder les clés d'extraction dans les valeurs finales
- ne pas confondre SITUATION ET PROFESSION
- ne jamais ignorer le NUMÉRO DE VOIE
- 'Wittve' ou 'Ww.' ou 'Wtw.' ou 'Wwe' ne sont pas des valeurs de PROFESSION
LIGNES ENTREPRISE
- commence par '*' et représente une entrée d'entreprise.
- le premier champ le nom de l'entrepise. Il doit être ajouté au champ COMPLÉMENT PROFESSIONNEL
- le champ suivant mentionne le ser=cteur d'activité de l'entreprise et est ajouté à COMPLÉMENT PROFESSIONNEL
- la clé Inh ou Inhaber précise les informations du gérant. 
- si le gérant n'est pas précisé, le premier champ est dans l'ordre PRENOM NOM
- si l'entrée contient plusieurs personnes, créer une ligne pour chacune d'elles et garder le nom de l'entreprise en COMPLÉMENT D'ADRESSE
- la fonction de chaque personne de l'entreprise doit être précisée dans COMPLÉMENT D'ADRESSE
- 'Wittve' ou 'Ww.' ou 'Wtw.' ou 'Wwe' ne sont pas des professions 
Voici un exemple complet contenant des lignes et les sorties CSV attendues auquel tu dois te référer \n:

"""

# "


def get_model_name() -> str:
    """Récupère le nom du modèle chargé sur le serveur."""
    req = urllib.request.Request(
        f"{VLLM_URL}/v1/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
        return data["data"][0]["id"]


def evaluate(true_df, pred_df):
    """Comparer les valeurs réelles et les valeurs extraites à partir de la fonction match"""

    # Compteur des erreurs en général
    global_error = 0
    # Compteur des erreurs sur chaque champ
    errors = np.zeros(9)

    cpt = 0
    errors_lines = []

    # Parcours les df par ligne
    for true_row, pred_row in zip(true_df.itertuples(), pred_df.itertuples()):
        # retirer le champ de l'index
        true_row = true_row[1:]
        pred_row = pred_row[1:]

        res = []
        for pos, (t_, p_) in enumerate(zip(true_row, pred_row)):
            if pos == 0:
                pred_value = normalize(pred_row[0]+' '+pred_row[1])
                true_value = normalize(true_row[0]+' '+true_row[1])

                ratio = fuzz.token_set_ratio(pred_value, true_value)

                if ratio >= 90 and field_match(values=(normalize(true_value), normalize(pred_value)), tol=2):
                    res.append(True)
                else:
                    res.append(False)
            elif pos == 2:
                pred_value = normalize(pred_row[2]+' '+pred_row[3])
                true_value = normalize(true_row[2]+' '+true_row[3])

                ratio = fuzz.token_set_ratio(pred_value, true_value)

                if ratio >= 90 or field_match(values=(normalize(t_), normalize(p_)), tol=1):
                    res.append(True)
                else:
                    res.append(False)

            elif pos == 4:
                pred_value = normalize(p_)
                true_value = normalize(t_)

                ratio = fuzz.token_set_ratio(pred_value, true_value)

                if ratio >= 90 or field_match(values=(normalize(t_), normalize(p_)), tol=2):
                    res.append(True)
                else:
                    res.append(False)

            elif pos == 3 or pos == 1:
                continue
            else:
                res.append(field_match(values=(normalize(t_), normalize(p_))))

        # incrémenter les erreurs globales
        if False in res:
            global_error += 1

        # incrémentrer les compteurs d'erreur sur chaque champ correspondant
        for i, r in enumerate(res):
            if not r:
                errors_lines.append((cpt, i))
                errors[i] += 1

        cpt += 1

    print(f"\nErreurs présentes sur {global_error} entrées")
    print(f"\nLe nombre d'erreurs sur chaque champ est : \n{errors}")

    # calculer les taux en pourcentage
    errors = (1 - errors/len(true_df))*100
    global_error = (1 - global_error/len(true_df))*100

    return global_error, errors, errors_lines


def results_display(global_error, errors, nbr_entries):
    """Fonction pour afficher les résultats"""
    print(f"\nNombre total d'exemples : {nbr_entries}")

    print(f"Le total de bonne extraction globale est de {global_error:.2f} %")
    print("\nTaux de reconnaissance sur chaque champ :")
    print(f"- champ NOM & PRÉNOMS : {errors[0]:.2f} %")
    print(f"- champ PROFESSION & COMPLÉMENTS : {errors[1]:.2f} %")
    print(f"- champ NOM DE VOIE : {errors[2]:.2f} %")
    print(f"- champ NUMERO DE VOIE : {errors[3]:.2f} %")
    print(f"- champ FAUBOURG : {errors[4]:.2f} %")
    print(f"- champ PROPRIETAIRE : {errors[5]:.2f} %")
    print(f"- champ NOM DE JEUNE FILLE : {errors[6]:.2f} %")
    print(f"- champ SITUATION : {errors[7]:.2f} %")
    print(f"- champ ADRESSE PROFESSIONNELLE : {errors[8]:.2f} %")


def chunks(iterable, size):
    """function to yield chunks of a specified size from an iterable"""

    # Create an iterator from the input iterable
    iterator = iter(iterable)

    # Loop over the iterator, taking the first element in each iteration
    for first in iterator:
        # Yield a list consisting of the first element and the next 'size-1' elements from the iterator
        yield [first] + list(islice(iterator, size - 1))


def field_match(values, tol=1):
    """Permet de dire si un champ extrait est identique à la vraie valeur, 
    en tolérant des erreurs minimales, grâce à la distance de Levenshtein"""

    true_val, pred_val = values

    true_val = str(true_val)
    pred_val = str(pred_val)

    # Vérifier si la valeur est une valeur numérique ; si oui la comparaison est stricte
    try:
        float(true_val)
        return true_val == pred_val
    except:
        if true_val == '*' or true_val == 'Sans':
            return normalize(true_val) == normalize(pred_val)
        d = distance(true_val, pred_val)
        if d <= tol:
            return True
        return False


def col_sep(cols, n=4):
    """Séparer les colonnes en n parties afin de permettre des sorties entières"""

    final_cols = []

    for col in cols:
        col_lines = col.splitlines()
        # col_lines = correct_names_pre(col_lines)
        if len(col_lines) >= 4:
            total = len(col_lines)
            parts = []

            for k in range(1, n):
                mid = (k * total) // n
                # Décaler si on tombe sur un séparateur
                while mid < total and col_lines[mid].split() and col_lines[mid].split()[0] in ('—', '-'):
                    mid += 1
                parts.append(mid)

            # Découper selon les indices trouvés
            indices = [0] + parts + [total]
            for i in range(len(indices) - 1):
                chunk = col_lines[indices[i]:indices[i+1]]
                final_cols.append('\n'.join(chunk))
        else:
            final_cols.append(col)

    columns = []

    for col in final_cols:
        if len(col) != 0:
            columns.append(col)

    return columns


def get_columns(example_file):
    """Récupérer les colonnes de données du fichier texte"""

    # Lecture du fichier
    with open(example_file, "r") as fichier:
        lines = fichier.readlines()

    # Concaténer les lignes du fichier en une seule chaîne de caractères
    lines = ''.join(lines)

    # Définition des délimiteurs pour extraire le contenu de chaque colonne
    start_column = r"<column>\n"
    end_column = r"\n</column>"

    # Création du pattern regex pour extraire le contenu des colonnes
    column_pattern = rf"(?<={start_column})(.*?)(?={end_column})"

    # Extraction du contenu de chaque colonne
    cols = re.findall(column_pattern, lines, flags=re.DOTALL)

    # Diviser chaque colonne en parties correctes
    # final_cols = col_sep(cols)
    
    # final_cols = chunks(cols, 2)

    return cols


def get_columns_no_tag(example_file):
    """Récupérer les colonnes de données du fichier texte sans balises"""

    # Lecture du fichier
    with open(example_file, "r") as fichier:
        lines = fichier.readlines()

    cols = list(chunks(lines, len(lines)//4))

    n_cols = []
    for col in cols:
        n_cols.append(''.join(col))

    # Diviser chaque colonne en parties correctes
    final_cols = col_sep(n_cols)

    return final_cols


def correct_names_pre(lines):
    """Attribuer le bon nom aux lignes sans nom de famille afin de permettre un traitement par blocs plus petits"""
    p_name = ''
    new_lines = []
    for line in lines:

        name = line.split(',')[0]

        if len(name) > 0:
            name = line.split()[0]

        if name == '—' or name == '-':
            name = p_name
            parts = line.split()
            parts[0] = name
            line = ' '.join(parts)
        elif len(name) >= 2 and (not name.endswith('.')) and (not name.endswith('platz')) and (not name.endswith('gasse')) and name != 'von':
            p_name = name
        new_lines.append(line)
    return new_lines


def correct_names_post(name):
    """Remettre le titre de la personne après le nom de famille"""

    titles = ['von', 'de', 'v.', 'vater', 'senior',
              'junior', 'freiherr', 'frhr.', 'freih.']
    parts = name.split()

    if len(parts) > 1:
        temp_parts = []
        if parts[0].lower() in titles:
            temp_parts.append(parts[1])
            temp_parts.append(parts[0])
            name = ' '.join(temp_parts)
    return name


def normalize(s):
    """Normaliser une chaîne de caratère en supprimant les ponctuations et les espaces inutiles"""

    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


async def process_column(session, semaphore, i, column, prompt, exemple_csv, model, vllm_url, api_key, output_format):
    async with semaphore:
        messages = [
            {"role": "system", "content": prompt + exemple_csv},
            {"role": "user", "content": f"Extraire les données des lignes suivantes selon les règles fournies : \n {column}"},
        ]
        try:
            async with session.post(
                f"{vllm_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.0,
                    "response_format": {"type": "json_schema", "json_schema": output_format}
                }
            ) as resp:
                text = await resp.json()
                data = json.loads(text["choices"][0]["message"]["content"])
                return i, data['result']
        except Exception as e:
            print(f"\nErreur colonne {i} : {e}")
            return i, []

async def process_all_columns(columns, prompt, exemple_csv, model, vllm_url, api_key, output_format, max_concurrent=4):
    semaphore = asyncio.Semaphore(max_concurrent)
    async with aiohttp.ClientSession() as session:
        tasks = [
            process_column(session, semaphore, i, col, prompt, exemple_csv, model, vllm_url, api_key, output_format)
            for i, col in enumerate(columns)
        ]
        responses = await tqdm_asyncio.gather(*tasks, desc="Colonnes en cours", ncols=60, unit="col")
    
    result = []
    for i, data in sorted(responses, key=lambda x: x[0]):
        result.extend(data)
    return result
