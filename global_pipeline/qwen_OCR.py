#!/usr/bin/env python
# coding: utf-8

# In[40]:
import json
import urllib.request
import urllib.error
import os
from tqdm import tqdm
import glob
from pathlib import Path
from dotenv import load_dotenv
import base64
from PIL import Image
import base64, io

load_dotenv()

VLLM_URL = os.getenv("VLLM_URL")
API_KEY = os.getenv("VLLM_API_KEY")
MODEL = os.getenv("VLLM_MODEL")
MODEL_RES_LOG_FILE = "log.txt"


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
    "schema": {
        "type": "object",
        "properties": {
            "result": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
        },
        "required": ["result"],
    },
}

# all_images = sorted(glob.glob("images/" + "*.jpg"))

# for image_path in tqdm(all_images):
#     with open(image_path, "rb") as f:
#         b64 = base64.b64encode(f.read()).decode()

#     # Détecte l'extension
#     ext = image_path.split(".")[-1].lower()
#     mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
#         ext, "image/jpeg"
#     )

#     def resize_image(image_path, max_size=1024):
#         img = Image.open(image_path)
#         img.thumbnail((max_size, max_size), Image.LANCZOS)

#         buf = io.BytesIO()
#         img.save(buf, format="JPEG", quality=85)
#         b64 = base64.b64encode(buf.getvalue()).decode()
#         return f"data:image/jpeg;base64,{b64}"

#     image_url = resize_image(image_path, max_size=1024 * 2.0)

#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": image_url,
#                     },
#                 },
#                 {
#                     "type": "text",
#                     "text": (
#                         "Extrais toutes les lignes de manière fidèle. "
#                         "CONSERVER OBLIGATOIREMENT et ABSOLUMENT VERBATIM le tiret en début de ligne"
#                         "Certaines entrées s'étendent sur 2 lignes, si une entrée ne commence pas par un nom, associe la à l'entrée précédente dans une seule ligne"
#                         "retire toutes les lignes qui correspondent à des titres de section"
#                         "retour à la ligne après chaque ligne"
#                         "pas de numéro de page"
#                         "chaque entrée complète sur une seule ligne"
#                         "NE PAS IGNORER LES TIRETS '—' NI LES '*' en début de ligne"
#                     ),
#                 },
#             ],
#         }
#     ]

#     model = MODEL or get_model_name()
#     payload = json.dumps(
#         {
#             "model": model,
#             "messages": messages,
#             "temperature": 0.0,
#             "response_format": {"type": "json_schema", "json_schema": OUTPUT_FORMAT},
#         }
#     ).encode()

#     req = urllib.request.Request(
#         f"{VLLM_URL}/v1/chat/completions",
#         data=payload,
#         headers={
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {API_KEY}",
#         },
#         method="POST",
#     )
#     try:
#         with urllib.request.urlopen(req) as resp:
#             text = json.loads(resp.read())

#             data = text["choices"][0]["message"]["content"]
#             data = json.loads(data)

#             txt_path = image_path.split(".")[0] + ".txt"
#             txt_path = txt_path.split("/")[1]

#             year = image_path.split("_")[2][2:]
#             output_path = "text/" + year + "/"

#             output_dir = Path(output_path)
#             output_dir.mkdir(exist_ok=True)

#             for line in data["result"]:
#                 with open(
#                     output_path + txt_path, mode="a", encoding="utf-8"
#                 ) as fichier:
#                     fichier.write(line + "\n")

#     except urllib.error.HTTPError as e:
#         print("Status:", e.code)
#         print("Body:", e.read().decode())


def worker_ocr(queue_in, queue_out_extraction):
    print("[Step 3] Module de correction OCR prêt et à l'écoute...")

    while True:
        # all_images = sorted(glob.glob("images/" + "*.jpg"))

        # for image_path in tqdm(all_images):
        image_path = queue_in.get()
        with open(MODEL_RES_LOG_FILE, "a") as log_file:
            log_file.write(f"[Step 3] Fichier reçu pour extraction OCR : {image_path}\n")
        try:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
    
            # Détecte l'extension
            ext = image_path.split(".")[-1].lower()
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
                ext, "image/jpeg"
            )
    
            def resize_image(image_path, max_size=1024):
                img = Image.open(image_path)
                img.thumbnail((max_size, max_size), Image.LANCZOS)
    
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                b64 = base64.b64encode(buf.getvalue()).decode()
                return f"data:image/jpeg;base64,{b64}"
    
            image_url = resize_image(image_path, max_size=1024 * 1.95)
    
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Extrais toutes les lignes de manière fidèle. "
                                "CONSERVER OBLIGATOIREMENT et ABSOLUMENT VERBATIM le tiret en début de ligne"
                                "Certaines entrées s'étendent sur 2 lignes, si une entrée ne commence pas par un nom, associe  la à l'entrée précédente dans une seule ligne"
                                "retire toutes les lignes qui correspondent à des titres de section"
                                "retour à la ligne après chaque ligne"
                                "pas de numéro de page"
                                "chaque entrée complète sur une seule ligne"
                                "NE PAS IGNORER LES TIRETS '—' NI LES '*' en début de ligne"
                                "Ignorer la page qui ne concerne pas les entrées de noms et toutes les entêtes"
                            ),
                        },
                    ],
                }
            ]
    
            model = MODEL or get_model_name()
            payload = json.dumps(
                {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.0,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": OUTPUT_FORMAT,
                    },
                }
            ).encode()
    
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
    
                    txt_path = image_path.split(".")[0] + ".txt"
                    txt_path = txt_path.split("/")[-1]
    
                    # year = image_path.split("_")[2][2:]
                    output_path = "echantillon/text/"
    
                    output_dir = Path(output_path)
                    output_dir.mkdir(exist_ok=True)
    
                    for line in data["result"]:
                        with open(
                            output_path + txt_path, mode="a", encoding="utf-8"
                        ) as fichier:
                            fichier.write(line + "\n")
                    queue_out_extraction.put(output_path + txt_path)
    
            except urllib.error.HTTPError as e:
                print("Status:", e.code)
                print("Body:", e.read().decode())
            
            os.remove(image_path)
            queue_in.task_done()
    
            with open(MODEL_RES_LOG_FILE, "a") as log_file:
                log_file.write(f"[Step 3] Fin extraction OCR : {image_path}\n")
        except:
            with open(MODEL_RES_LOG_FILE, "a") as log_file:
                log_file.write(f"[ERREUR] Erreur sur fichier : {image_path}\n")