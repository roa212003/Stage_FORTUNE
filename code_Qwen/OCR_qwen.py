#!/usr/bin/env python
# coding: utf-8

# In[40]:


import json
import urllib.request
import urllib.error
import os
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import glob


# In[41]:


VLLM_URL  = os.getenv("VLLM_URL", "https://litis-vllm.univ-rouen.fr")
API_KEY   = os.getenv("VLLM_KEY", "b5e611b779da8e6757a0860783af83d0afb148b9997bd898a7959c8c9fe52a64")
MODEL     = os.getenv("VLLM_MODEL", "") 


# In[44]:


def get_model_name() -> str:
    """Récupère le nom du modèle chargé sur le serveur."""
    req = urllib.request.Request(
        f"{VLLM_URL}/v1/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
        return data["data"][0]["id"]

OUTPUT_FORMAT = {
                    "name": "person_list",
                    "schema": 
                    {
                        'type':'object',
                        'properties':{
                            'result':{
                                "type": "array",
                                "items": 
                                        {
                                            "type": "string",
                                        },
                                    },
                        },
                        'required': ['result']
                    }
                }


# # Extration des entrées par nom

# In[43]:


import base64
all_images = sorted(glob.glob('images/'+'*.jpg'))

for image_path in tqdm(all_images[0:1]):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    # Détecte l'extension
    ext = image_path.split(".")[-1].lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")

    from PIL import Image
    import base64, io

    def resize_image(image_path, max_size=1024):
        img = Image.open(image_path)
        img.thumbnail((max_size, max_size), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    image_url = resize_image(image_path, max_size=1024*2.0)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    }
                },
                {
                    "type": "text",
                    "text": (
                        "Extrais toutes les lignes de données de manière fidèle. "
                        "Certaines entrées s'étendent sur 2 lignes, si une entrée ne commence pas par un nom, associe la à l'entrée précédente dans une seule ligne"
                        "si une ligne commence par '—' ou un autre tiret, donne lui le dernier nom de famille rencontré dans les lignes précédentes"
                        "retire toutes les lignes qui correspondent à des titres de section"
                        "retour à la ligne après chaque ligne"
                        "pas de numéro de page"
                        "chaque entrée complète sur une seule ligne"
                    )
                }
            ]
        }
    ]

    model = MODEL or get_model_name()
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "response_format" :{
                                   "type": "json_schema",
                                   "json_schema": OUTPUT_FORMAT
                               }
    }).encode()

    t_0 = time.time()
    req = urllib.request.Request(
            f"{VLLM_URL}/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            method="POST",
        )
    try:
        with urllib.request.urlopen(req) as resp:
                text = json.loads(resp.read())

                data = text["choices"][0]["message"]["content"]
                data = json.loads(data)

                txt_path = image_path.split(".")[0]+'.txt'
                txt_path = txt_path.split("/")[1]

                for line in data['result']:
                    with open('text/'+txt_path, mode="a", encoding="utf-8") as fichier:
                        fichier.write(line+'\n')

    except urllib.error.HTTPError as e:
        print("Status:", e.code)
        print("Body:", e.read().decode())
    t_1 = time.time()


# # Extraction des entrées par rue

# In[ ]:


import base64
all_images = glob.glob('images/'+'*.jpg')
all_images.sort()

for image_path in tqdm(all_images[8:]):
    with open('text/1873_all_entries_by_street.txt', mode="a", encoding="utf-8") as fichier:
        fichier.write(image_path.split('_')[-1].split('.')[0]+'\n')

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    # Détecte l'extension
    ext = image_path.split(".")[-1].lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")

    from PIL import Image
    import base64, io

    def resize_image(image_path, max_size=1024):
        img = Image.open(image_path)
        img.thumbnail((max_size, max_size), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    image_url = resize_image(image_path, max_size=1024*2.0)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    }
                },
                {
                    "type": "text",
                    "text": (
                        "Extrais toutes les lignes de données de manière fidèle. "
                        "l'image a 2 page et chaque page à 2 colonnes"
                        "ignore tout ce qui ne concerne pas les entrées de noms par rue"
                        "ignore la plage des noms de rue en haut de chaque page sous la forme 2 noms de rue séparés par un caractère '–'"
                        "ajouter au début de chaque entrée, le nom de la rue correspondante"
                        "chaque colonne contient plusieurs paragraphe"
                        "chaque paragraphe commence par le nom d'une rue et contient les personnes habitant cette rue et leur numéro"
                        "chaque entrée de numéro par colonne doit être écrite sur une ligne"
                        "parcourir l'image en fonction des colonnes"
                        "effectuer l'extraction colonne après colonne, de gauche à droite"
                        "un retour à la ligne après chaque ligne"
                        "l'extraction doit respecter les colonnes de l'image avec une ligne par entrée"
                        "si une ligne ne commence pas par un numéro, elle doit être mise sur la même ligne que l'entrée précédente"
                    )
                }
            ]
        }
    ]

    model = MODEL or get_model_name()
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "response_format" :{
                                   "type": "json_schema",
                                   "json_schema": OUTPUT_FORMAT
                               }
    }).encode()

    req = urllib.request.Request(
            f"{VLLM_URL}/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            method="POST",
        )
    try:
        with urllib.request.urlopen(req) as resp:
                text = json.loads(resp.read())

                data = text["choices"][0]["message"]["content"]
                data = json.loads(data)

                txt_path = image_path.split(".")[0]+'.txt'
                txt_path = txt_path.split("/")[1]

                for line in data['result']:
                    with open('text/1873_all_entries_by_street.txt', mode="a", encoding="utf-8") as fichier:
                        fichier.write(line+'\n')

    except urllib.error.HTTPError as e:
        print("Status:", e.code)
        print("Body:", e.read().decode())

    with open('text/1873_all_entries_by_street.txt', mode="a", encoding="utf-8") as fichier:
        fichier.write('\n\n')
