import glob
import os
from tqdm.contrib.concurrent import thread_map
from control_utils import *

# files = sorted(glob.glob(CSV_FOLDER + "*.csv"))


def validate_file(file, queue_ocr):
    """Fonction principale pour la validation des fichiers"""
        
    YEAR = file.split("_")[-2][2:]
    OUTPUT_PATH = DEST_FOLDER + YEAR + "/"
    IMAGES_FOLDER = IMAGES_FOLDER_BASE + YEAR

    OUTPUT_DIR = Path(OUTPUT_PATH)
    OUTPUT_DIR.mkdir(exist_ok=True)
    file_df = pd.read_csv(file, index_col=False).reset_index(drop=True)
    errors_mat = np.empty((file_df.shape[0], 16), dtype=object)

    names = file_df["NOM"]

    file_df["PRENOM"] = file_df["PRENOM"].apply(clean_first_name)

    file_name = file.split("/")[-1]
    RESULT_FILE = file_name.split(".")[0] + ".csv"

    ERROR_DICT = {
        "file": file_name,
        "LINES_CPT": 0,
        "ERROR_1": 0,
        "ERROR_2": 0,
        "ERROR_3": 0,
        "ENTR": 0,
    }

    # TEST_ENTRP
    errors_mat = test_entr(
        file_df=file_df, errors_mat=errors_mat, error_dict=ERROR_DICT
    )

    # TEST_1
    errors_mat, ERROR_DICT = test_1(
        names, errors_mat, ERROR_DICT, file_name=file_name, image_folder=IMAGES_FOLDER, queue_ocr=queue_ocr
    )
    # TEST 2
    text_file = TEXT_FOLDER + YEAR + '/' + 'txt/' + file_name.split(".")[0] + ".txt"

    text_lines = open(text_file, "r")
    text_lines = list(text_lines)
    errors_mat, ERROR_DICT = test_2(
        text_lines, list(file_df.iterrows()), errors_mat, ERROR_DICT
    )

    # TEST 3

    file_df = compute_predictions(file_df=file_df)

    errors_mat, ERROR_DICT = test_3(file_df, errors_mat, ERROR_DICT)

    file_df.drop(
        columns=[
            "proba_nom_jf",
            "proba_prenom",
            "proba_profession",
            "proba_comp_profession",
            "proba_voie",
            "proba_adr_pro",
        ],
        inplace=True,
    )

    errors_df = pd.DataFrame(
        errors_mat,
        columns=[
            "TEST_ENTR",
            "TEST_1",
            "TEST_2",
            "TEXTE",
            "TEST_3",
            "D_NOM",
            "D_PREN",
            "D_PROF",
            "D_COMP",
            "D_NOM_V",
            "D_NUM_V",
            "D_FAUB",
            "D_PROPR",
            "D_NOM_JF",
            "D_SITU",
            "D_ADR_PRO",
        ],
    )

    result_df = pd.concat([file_df, errors_df], axis=1)
    result_df.to_csv(f"{OUTPUT_DIR}/{RESULT_FILE}", index=False)

    return ERROR_DICT


######

# resultats = thread_map(validate_file, files, max_workers=64)

# resultats = []
# for file in files[600:601]:
#     resultats.append(validate_file(file))

# log_results(resultats, "validation_log.txt")


def worker_validation(queue_extraction, queue_ocr, compteur):
    print("[Step 2] Module de validation prêt et à l'écoute...")

    while True:
        file = queue_extraction.get()

        with open(MODEL_RES_LOG_FILE, "a") as log_file:
            log_file.write(f"[Step 2] : Début traitement du fichier {file}\n")
        try:
            resultat = validate_file(file, queue_ocr=queue_ocr)
            # log_results(resultats=resultat, filename="validation_log.txt")
            with open(MODEL_RES_LOG_FILE, "a") as log_file:
                log_file.write(f"[Step 2] : Fin traitement du fichier {file}\n")
            os.remove(file)
            if resultat['ERROR_1'] / resultat['LINES_CPT'] < 0.05:
                compteur.terminer_un()
        except Exception as e:
            with open(MODEL_RES_LOG_FILE, "a") as log_file:
                log_file.write(f"[ERREUR] : Erreur sur fichier {file}\n")
                log_file.write(f"{e}\n")
