import threading
from queue import Queue
from utils import *
import time
from tqdm import tqdm
from control_script import worker_validation
from Qwen36_parse import worker_extraction
from qwen_OCR import worker_ocr
from pathlib import Path

MODEL_RES_LOG_FILE = "log.txt"
VAL_QUEUE_FOLDER = "echantillon/extract"
OCR_QUEUE_FOLDER = "ctrl_files/redo/"

class CompteurPipeline:
    def __init__(self, total_initial):
        self.lock = threading.Lock()
        self.actifs = total_initial

    def terminer_un(self):
        with self.lock:
            self.actifs -= 1

    def est_vide(self):
        with self.lock:
            return self.actifs <= 0

def main():
    # 1. Création des files d'attente (Queues) de communication
    queue_extraction = Queue()
    queue_validation = Queue()
    queue_correction = Queue()

    compteur = CompteurPipeline(total_initial=len(all_text_files))

    # 2. Lancement de la Step 1 : Extraction (1 seul thread suffit souvent car séquentiel ou géré par l'API)
    t1 = threading.Thread(target=worker_extraction, args=(queue_extraction, queue_validation), daemon=True)
    t1.start()

    # 3. Lancement de la Step 2 : Validation (MULTI-THREADÉ : on lance par ex. 4 threads)
    t2 = threading.Thread(target=worker_validation, args=(queue_validation, queue_correction, compteur), daemon=True)
    t2.start()

    # 4. Lancement de la Step 3 : Correction (1 thread dédié)
    t3 = threading.Thread(target=worker_ocr, args=(queue_correction, queue_extraction), daemon=True)
    t3.start()

    # --- Initialisation du pipeline ---
    # On simule l'arrivée de 3 fichiers bruts dans la première file
    
    for file in all_text_files:
        queue_extraction.put(file)

    csv_files = [
        str(f) for f in Path(VAL_QUEUE_FOLDER).rglob("*.csv") 
        if ".ipynb_checkpoints" not in f.parts
    ]

    jpg_files = [
        str(f) for f in Path(OCR_QUEUE_FOLDER).rglob("*.jpg") 
        if ".ipynb_checkpoints" not in f.parts
    ]

    csv_files = sorted(csv_files)

    jpg_files = sorted(jpg_files)

    print(f"Nombre de fichiers en Step 1 : {len(all_text_files)}")
    print(f"Nombre de fichiers en Step 2 : {len(csv_files)}")
    print(f"Nombre de fichiers en Step 3 : {len(jpg_files)}")

    # Assurer la reprise après un arrêt prématuré
    for file in csv_files:
        queue_validation.put(file)

    for file in jpg_files:
        queue_correction.put(file)

    with open(MODEL_RES_LOG_FILE, "a") as log_file:
            log_file.write("[Orchestrateur] Début de l'extraction générale\n\n")
    
    with tqdm(total=len(all_text_files), desc="Traitement du pipeline", unit="fichier") as pbar:
        dernier_compte = len(all_text_files)
        
        while not compteur.est_vide():
            time.sleep(0.5)
            # Supposons que compteur.get_valeur() donne le nombre d'éléments restants
            valeur_actuelle = compteur.actifs 
            
            # On calcule combien de fichiers ont été validés pendant le dernier intervalle
            traites = dernier_compte - valeur_actuelle
            if traites > 0:
                pbar.update(traites)
                dernier_compte = valeur_actuelle

    with open(MODEL_RES_LOG_FILE, "a") as log_file:
        log_file.write("\n\n[Orchestrateur] Fin de l'extraction générale")
            

if __name__ == "__main__":
    main()