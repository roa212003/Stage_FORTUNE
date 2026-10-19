from pathlib import Path
import unicodedata
import pandas as pd
import re
import glob
from rapidfuzz import fuzz, process
import numpy as np
from unidecode import unidecode
from weighted_levenshtein import lev
import shutil
import joblib
import statistics
from tqdm.contrib.concurrent import thread_map
import matplotlib.pyplot as plt
import numpy as np

KEYS = ["geb.", "gb.", ""]
PARTICULES = r"\b(d|l|j|m|t|s|n|qu|lorsqu|puisqu|quoiqu|jusqu|quelqu)['\u2019]"


def normalize_name(s: str) -> str:
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = re.sub(PARTICULES, "", s, flags=re.IGNORECASE)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = unidecode(s)
    return s.strip().lower()


def normalize(s):
    s = s.lower()
    s = unidecode(s)
    s = re.sub(r"[\"']", "", s)       
    s = re.sub(r"[^\w\s]", " ", s)   
    s = re.sub(r"\s+", " ", s).strip()
    return s


def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    similarity = float(intersection) / union
    return similarity


def weigthed_lev(ref, pred):
    insert_costs = np.ones(128, dtype=np.float64) * 0.5
    delete_costs = np.ones(128, dtype=np.float64) * 1
    substitute_costs = np.ones((128, 128), dtype=np.float64) * 0.2
    return lev(
        ref,
        pred,
        substitute_costs=substitute_costs,
        insert_costs=insert_costs,
        delete_costs=delete_costs,
    )


def check_num_voie(val):
    if len(re.findall(r"\d+", str(val))) or val == "*":
        return True, val
    else:
        return False, val


def check_faubourg(val):
    FAUBOURG_LIST = [
        "Grüneb.",
        "Neudorf",
        "Neuhof",
        "Ruprechtsau",
        "Grüneberg",
        "Königshofen",
        "Kronenburg",
        "Königsh",
        "Rupr",
        "*",
        "Kronenb",
        "Königshof",
        "Kronenb",
        "Nd",
        "Rup",
        "Ru",
        "Kr"
    ]
    return str(val).replace(".", "") in FAUBOURG_LIST, val


def check_propr(val):
    PROPR_LIST = ["E", "I", "II", "III", "IV", "V", "D", "H", "*"]
    return str(val).replace(".", "") in PROPR_LIST, val


def check_situation(val):
    SITUATION_LIST = ["Ww", "ledig", "*", "Wittve", "Wtw", "Wwe"]
    return str(val).replace(".", "") in SITUATION_LIST, val

LAST_NAME_MODEL = joblib.load("evaluation_script/ckpt/svc_rbf_nom.pkl")
LAST_NAME_SCALER = joblib.load("evaluation_script/ckpt/scaler_nom.pkl")
LAST_NAME_EMB = joblib.load("evaluation_script/ckpt/tfidf_nom.pkl")
NAME_MODEL = joblib.load("evaluation_script/ckpt/svc_rbf_prenom.pkl")
NAME_SCALER = joblib.load("evaluation_script/ckpt/scaler_prenom.pkl")
NAME_EMB = joblib.load("evaluation_script/ckpt/tfidf_prenom.pkl")
PROFESSION_MODEL = joblib.load("evaluation_script/ckpt/svc_rbf_profession.pkl")
PROFESSION_SCALER = joblib.load("evaluation_script/ckpt/scaler_profession.pkl")
PROFESSION_EMB = joblib.load("evaluation_script/ckpt/tfidf_profession.pkl")
VOIE_MODEL = joblib.load("evaluation_script/ckpt/svc_rbf_voie.pkl")
VOIE_SCALER = joblib.load("evaluation_script/ckpt/scaler_voie.pkl")
VOIE_EMB = joblib.load("evaluation_script/ckpt/tfidf_voie.pkl")
ADR_PRO_MODEL = joblib.load("evaluation_script/ckpt/svc_rbf_adresse_pro.pkl")
ADR_PRO_SCALER = joblib.load("evaluation_script/ckpt/scaler_adresse_pro.pkl")
ADR_PRO_EMB = joblib.load("evaluation_script/ckpt/tfidf_adresse_pro.pkl")
ENTRP_EMB = joblib.load("evaluation_script/ckpt/tfidf_entrp.pkl")
ENTRP_SCALER = joblib.load("evaluation_script/ckpt/scaler_entrp.pkl")
ENTRP_MODEL = joblib.load("evaluation_script/ckpt/svc_entrp.pkl")

def check_last_name(index, val, file_df):
    if normalize(val) in NAMES_SET:
        return True, val
    else:
        proba = file_df["proba_nom_jf"][index]
        # log.write(f"{val} : {proba}\n")
        dec = ["Faux", "Vrai"]
        return proba[0] == 1 and proba[1] > 0.52, (
            val,
            f"{dec[proba[0]]} à {proba[1]*100:.2f}%",
        )


def check_first_name(index, val, file_df):
    if normalize(val) in FIRST_NAMES_SET or val in ["*", "Sans"]:
        return True, val
    else:
        proba = file_df["proba_prenom"][index]
        dec = ["Faux", "Vrai"]
        return proba[0] == 1 and proba[1] > 0.52, (
            val,
            f"{dec[proba[0]]} à {proba[1]*100:.2f}%",
        )


def check_prof(index, val, file_df):
    if normalize(val) in PROF_SET or val in ["*", "Sans"]:
        return True, index
    else:
        proba = file_df["proba_profession"][index]
        # log.write(f"{val} : {proba}\n")
        dec = ["Faux", "Vrai"]
        return proba[0] == 1 and proba[1] > 0.52, (
            val,
            f"{dec[proba[0]]} à {proba[1]*100:.2f}%",
        )


def check_comp_prof(index, val, file_df):
    if normalize(val) in PROF_SET or val in ["*", "Sans"]:
        return True, val
    else:
        proba = file_df["proba_comp_profession"][index]
        # log.write(f"{val} : {proba}\n")
        dec = ["Faux", "Vrai"]
        return proba[0] == 1 and proba[1] > 0.52, (
            val,
            f"{dec[proba[0]]} à {proba[1]*100:.2f}%",
        )


def check_voie(index, val, file_df):
    if normalize(val) in VOIE_SET or val in ["*", "Sans"]:
        return True, val
    else:
        proba = file_df["proba_voie"][index]
        # log.write(f"{val} : {proba}\n")
        dec = ["Faux", "Vrai"]
        return proba[0] == 1 and proba[1] > 0.52, (
            val,
            f"{dec[proba[0]]} à {proba[1]*100:.2f}%",
        )


def check_adr_pro(index, val, file_df):
    if normalize(val) in ADRESSE_PRO_SET or val in ["*", "Sans"]:
        return True, val
    else:
        proba = file_df["proba_adr_pro"][index]
        # log.write(f"{val} : {proba}\n")
        dec = ["Faux", "Vrai"]
        return proba[0] == 1 and proba[1] > 0.52, (
            val,
            f"{dec[proba[0]]} à {proba[1]*100:.2f}%",
        )


def compute_predictions(file_df):
    mask = ~file_df["NOM DE JEUNE FILLE"].isin(["*", "Sans"])
    if mask.any():
        nom_jf_values = LAST_NAME_EMB.transform(file_df.loc[mask, "NOM DE JEUNE FILLE"])
        nom_jf_values = LAST_NAME_SCALER.transform(nom_jf_values)
        nom_jf_values_probas = LAST_NAME_MODEL.predict_proba(nom_jf_values)
        file_df["proba_nom_jf"] = None
        file_df.loc[mask, "proba_nom_jf"] = pd.Series(
            [(row.argmax(), row[row.argmax()]) for row in nom_jf_values_probas],
            index=file_df.index[mask],
        )
    else:
        file_df["proba_nom_jf"] = None

    mask = ~file_df["PRENOM"].isin(["*", "Sans", ""])
    if mask.any():
        values = NAME_EMB.transform(file_df.loc[mask, "PRENOM"])
        values = NAME_SCALER.transform(values)
        values_probas = NAME_MODEL._predict_proba_lr(values)
        file_df["proba_prenom"] = None
        file_df.loc[mask, "proba_prenom"] = pd.Series(
            [(row.argmax(), row[row.argmax()]) for row in values_probas],
            index=file_df.index[mask],
        )
    else:
        file_df["proba_prenom"] = None

    mask = ~file_df["NOM DE VOIE"].isin(["*", "Sans"])
    if mask.any():
        values = VOIE_EMB.transform(file_df.loc[mask, "NOM DE VOIE"])
        values = VOIE_SCALER.transform(values)
        values_probas = VOIE_MODEL.predict_proba(values)
        file_df["proba_voie"] = None
        file_df.loc[mask, "proba_voie"] = pd.Series(
            [(row.argmax(), row[row.argmax()]) for row in values_probas],
            index=file_df.index[mask],
        )
    else:
        file_df["proba_voie"] = None

    mask = ~file_df["PROFESSION"].isin(["*", "Sans"])
    if mask.any():
        values = PROFESSION_EMB.transform(file_df.loc[mask, "PROFESSION"])
        values = PROFESSION_SCALER.transform(values)
        values_probas = PROFESSION_MODEL._predict_proba_lr(values)
        file_df["proba_profession"] = None
        file_df.loc[mask, "proba_profession"] = pd.Series(
            [(row.argmax(), row[row.argmax()]) for row in values_probas],
            index=file_df.index[mask],
        )
    else:
        file_df["proba_profession"] = None

    mask = ~file_df["COMPLEMENT DE PROFESSION"].isin(["*", "Sans"])
    if mask.any():
        values = PROFESSION_EMB.transform(file_df.loc[mask, "COMPLEMENT DE PROFESSION"])
        values = PROFESSION_SCALER.transform(values)
        values_probas = PROFESSION_MODEL._predict_proba_lr(values)
        file_df["proba_comp_profession"] = None
        file_df.loc[mask, "proba_comp_profession"] = pd.Series(
            [(row.argmax(), row[row.argmax()]) for row in values_probas],
            index=file_df.index[mask],
        )
    else:
        file_df["proba_comp_profession"] = None

    mask = ~file_df["ADRESSE PROFESSIONNELLE"].isin(["*", "Sans"])
    if mask.any():
        values = ADR_PRO_EMB.transform(file_df.loc[mask, "ADRESSE PROFESSIONNELLE"])
        values = ADR_PRO_SCALER.transform(values)
        values_probas = ADR_PRO_MODEL.predict_proba(values)
        file_df["proba_adr_pro"] = None
        file_df.loc[mask, "proba_adr_pro"] = pd.Series(
            [(row.argmax(), row[row.argmax()]) for row in values_probas],
            index=file_df.index[mask],
        )
    else:
        file_df["proba_adr_pro"] = None

    return file_df



def clean_first_name(name):
    pattern = re.compile(r"^.*?(?:ge)?b\.\s*[A-ZÄÖÜ][a-zäöüß]+,\s*")
    return pattern.sub("", name).strip()


def preprocess_csv(row):
    vals = [str(v) for v in row.tolist() if v not in ["*", "Sans"]]
    line = " ".join(vals).replace(".", " ")
    return normalize(line)


def preprocess_text(line):
    line = line.replace(",", " ").replace(".", " ")
    line = normalize(line)
    line = (
        line.replace("wwe", "ww")
        .replace("geb", "")
        .replace("ledig", "")
        .replace("gb", "")
    )
    line = (
        line.replace("geschaft", "")
        .replace("geschaft-lokal", "")
        .replace("geschaftslokal", "")
        .replace("gesch.-local", "")
        .replace("wohn", "")
        .replace("laden", "")
        .replace("wohnung", "")
    )
    return line


def similarity_score(text_raw, csv_proc):
    t = preprocess_text(text_raw)
    j = jaccard(t.split(), csv_proc.split())
    f = fuzz.token_sort_ratio(t, csv_proc)
    w = weigthed_lev(t, csv_proc)
    return j, f, w, t


def test_entr(file_df, errors_mat, error_dict):
    processed = file_df.apply(
        lambda row: ' '.join(str(v) for v in row.values if v not in ["Sans", '*', None]),
        axis=1
    )

    values = ENTRP_EMB.transform(processed)
    values = ENTRP_SCALER.transform(values)
    preds = ENTRP_MODEL.predict(values)

    for idx, pred in zip(file_df.index, preds):
        if pred == '1':
            errors_mat[idx, 0] = 'ENTRP'

    return errors_mat


def determiner_zones_domination(votre_serie, fenetre_tolerance=3):
    """Détecter les initiaux domainant parmi les noms et leurs indices afin de détecter les mauvais noms"""
    
    if votre_serie.empty:
        return []

    # ignorer toutes les lignes d'entreprise
    motifs_entreprise = r"&|\b(cie|gmbh|co|company|et fils|soc|inc)\b"
    est_valide = (
        votre_serie.notna() & 
        (votre_serie.astype(str).str.strip() != "") & 
        ~votre_serie.astype(str).str.lower().str.contains(motifs_entreprise, regex=True)
    )

    serie_filtree = votre_serie[est_valide].astype(str).str.lower().str.strip()
    if serie_filtree.empty:
        return []

    # Extraction des initiales sur les lignes valides
    initiales = serie_filtree.str[0]
    
    # Détection des zones par balayage (Multi-zone)
    zones = []
    index_reels = initiales.index.tolist()
    dernier_index_global = len(votre_serie) - 1
    
    idx_debut_zone_reelle = index_reels[0]
    lettre_courante = initiales.iloc[0]
    
    for i in range(1, len(initiales)):
        lettre_actuelle = initiales.iloc[i]
        idx_actuel_reel = index_reels[i]
        
        # Si la lettre change, on regarde si ce changement se confirme sur les lignes suivantes
        # (Évite de casser la zone à cause d'une seule coquille OCR)
        if lettre_actuelle != lettre_courante:
            suivantes = initiales.iloc[i : i + fenetre_tolerance]
            # Si la nouvelle lettre devient majoritaire dans la fenêtre, on bascule
            if (suivantes == lettre_actuelle).sum() >= (fenetre_tolerance / 2):
                # On ferme la zone précédente à l'index juste avant la bascule
                idx_fin_zone_reelle = index_reels[i - 1]
                zones.append({
                    "car": lettre_courante, 
                    "bgn": int(idx_debut_zone_reelle), 
                    "end": int(idx_fin_zone_reelle)
                })
                # On ouvre la nouvelle zone
                lettre_courante = lettre_actuelle
                idx_debut_zone_reelle = idx_actuel_reel

    # Fermeture de la toute dernière zone
    zones.append({
        "car": lettre_courante, 
        "bgn": int(idx_debut_zone_reelle), 
        "end": int(dernier_index_global)
    })
    
    # Ajustement cosmétique : s'il n'y a qu'une seule zone au final, on met end à -1 comme avant
    if len(zones) == 1:
        zones[0]["end"] = -1
    else:
        # On s'assure que les zones couvrent bien tout l'espace (y compris les entreprises aux frontières)
        for i in range(len(zones) - 1):
            zones[i]["end"] = zones[i+1]["bgn"] - 1
        zones[-1]["end"] = dernier_index_global

    return zones


def verif_alpha(names, errors_mat):
    errors = set()
    
    # On passe la matrice à la détection pour ignorer les ENTRP dès le calcul des zones
    order_zones = determiner_zones_domination(names)

    for zone in order_zones:
        car = zone['car']
        bgn = zone['bgn']
        end = zone['end']

        end_idx = len(names) if end == -1 else end + 1

        for global_idx in range(bgn, end_idx):
            # Pas d'alerte sur les lignes d'entreprise
            if errors_mat is not None and errors_mat[global_idx, 0] == "ENTRP":
                continue
                
            name = names.iloc[global_idx]
            
            # ALERT si le nom est vide
            if pd.isna(name) or str(name).strip() == "" or str(name) == "*":
                errors.add((global_idx, name, car))
                
            # Détection de tous les noms qui n'ont pas l'initial correspondant
            if not str(name).lower().startswith(car):
                errors.add((global_idx, name, car))

    score = len(errors) / len(names) if len(names) > 0 else 0
    return score, errors


def test_1(names, errors_mat, error_dict, file_name, image_folder, queue_ocr):
    # On passe errors_mat à la fonction pour qu'elle puisse identifier les "ENTRP"
    score, errors = verif_alpha(names, errors_mat)

    error_dict["ERROR_1"] += len(errors)

    for error in errors:
        intrus = error[0]
        errors_mat[intrus, 1] = "ALERT_1"

    if score > 0.05:
        image_file = image_folder + "/IMAGES/" + file_name.split(".")[0] + ".jpg"
        shutil.copy(image_file, REDO_FOLDER)
        new_file_path = str(REDO_FOLDER) + file_name.split(".")[0] + ".jpg"
        queue_ocr.put(new_file_path)

    return errors_mat, error_dict



def tokens_manquants(a, b, seuil=71):
    tokens_a = a.split()
    tokens_b = b.split()

    manquants = []

    for token in tokens_a:
        match = process.extractOne(token, tokens_b, scorer=fuzz.ratio)

        if match is None or match[1] < seuil:
            manquants.append(token)

    return manquants
    

def test_2(text_lines, csv_rows, errors_mat, error_dict):
    errors = []

    i_csv = 0
    i_text = 0

    while i_csv < len(csv_rows) and i_text < len(text_lines):
        idx, csv_row = csv_rows[i_csv]
        text_line_ = text_lines[i_text]

        if errors_mat[idx, 0] == 'ENTRP':
            i_csv += 1
            i_text += 1
            continue

        csv_proc = preprocess_csv(csv_row)

        j, f, w, text_proc = similarity_score(text_line_, csv_proc)

        # Éviter les décalages parasites en raison d'un texte mal rendu. Si le score entre
        # le csv et le text est trop petit, on recherche dans le voisinage un texte qui a un score plus fiable
        if j < 0.2:
            # Score mauvais → cherche dans la fenêtre [i_text-1, i_text+1]
            WINDOW = 2  # nb de lignes text à regarder autour
            best_score = (j, f, -w)  # on maximise j et f, minimise w
            best_offset = 0  # offset sur i_text

            for offset in range(-WINDOW, WINDOW + 1):
                if offset == 0:
                    continue
                candidate_idx = i_text + offset
                if candidate_idx < 0 or candidate_idx >= len(text_lines):
                    continue
                jc, fc, wc, _ = similarity_score(text_lines[candidate_idx], csv_proc)
                if (jc, fc, -wc) > best_score:
                    best_score = (jc, fc, -wc)
                    best_offset = offset
            if best_offset != 0 and best_score[0] >= 0.5:
                i_text += best_offset
                continue

            text_to_csv_miss = tokens_manquants(preprocess_text(text_line_), csv_proc)
            csv_to_text_miss = tokens_manquants(csv_proc, preprocess_text(text_line_))

            csv_to_text_miss = [token for token in csv_to_text_miss if token != csv_proc.split()[0]]

            if len(text_to_csv_miss) > 0 and len(csv_to_text_miss) > 0:
                errors.append((idx + 1, text_line_.replace("\n", ""), csv_proc, f"MISS & ADD ({' '.join(text_to_csv_miss)} & {' '.join(csv_to_text_miss)})"))
            if len(text_to_csv_miss) > 0:
                errors.append((idx + 1, text_line_.replace("\n", ""), csv_proc, f"MISS ({' '.join(text_to_csv_miss)})"))
            elif len(csv_to_text_miss) > 0:
                errors.append((idx + 1, text_line_.replace("\n", ""), csv_proc, f"ADD ({' '.join(csv_to_text_miss)})"))
            else:
                pass

            i_csv += 1
            i_text += 1
        elif (j < 0.6 and f < 85) and w > 10:
            # Vérification des valeurs manquantes de l'un à l'autre

            text_to_csv_miss = tokens_manquants(preprocess_text(text_line_), csv_proc)
            csv_to_text_miss = tokens_manquants(csv_proc, preprocess_text(text_line_))

            csv_to_text_miss = [token for token in csv_to_text_miss if token != csv_proc.split()[0]]

            if len(text_to_csv_miss) > 0 and len(csv_to_text_miss) > 0:
                errors.append((idx + 1, text_line_.replace("\n", ""), csv_proc, f"MISS & ADD ({' '.join(text_to_csv_miss)} & {' '.join(csv_to_text_miss)})"))
            if len(text_to_csv_miss) > 0:
                errors.append((idx + 1, text_line_.replace("\n", ""), csv_proc, f"MISS ({' '.join(text_to_csv_miss)})"))
            elif len(csv_to_text_miss) > 0:
                errors.append((idx + 1, text_line_.replace("\n", ""), csv_proc, f"ADD ({' '.join(csv_to_text_miss)})"))
            else:
                pass
            i_csv += 1
            i_text += 1
        else:
            i_csv += 1
            i_text += 1
            pass

    error_dict["LINES_CPT"] += len(csv_rows)
    error_dict["ERROR_2"] += len(errors)

    for error in errors:
        errors_mat[error[0] - 1, 2] = f"ALERT_2 {error[3]}"
        errors_mat[error[0] - 1, 3] = error[1]

    return errors_mat, error_dict


def test_3(file_df, errors_mat, error_dict):
    nums_check = file_df["NUMERO DE VOIE"].apply(check_num_voie).tolist()
    faubs_check = file_df["FAUBOURG"].apply(check_faubourg).tolist()
    proprs_check = file_df["PROPRIETAIRE"].apply(check_propr).tolist()
    situs_check = file_df["SITUATION"].apply(check_situation).tolist()

    last_names_check = (
        file_df["NOM DE JEUNE FILLE"]
        .to_frame()
        .apply(
            lambda row: check_last_name(row.name, row["NOM DE JEUNE FILLE"], file_df),
            axis=1,
        )
        .tolist()
    )
    first_names_check = (
        file_df["PRENOM"]
        .to_frame()
        .apply(lambda row: check_first_name(row.name, row["PRENOM"], file_df), axis=1)
        .tolist()
    )
    profs_check = (
        file_df["PROFESSION"]
        .to_frame()
        .apply(lambda row: check_prof(row.name, row["PROFESSION"], file_df), axis=1)
        .tolist()
    )
    comp_profs_check = (
        file_df["COMPLEMENT DE PROFESSION"]
        .to_frame()
        .apply(
            lambda row: check_comp_prof(
                row.name, row["COMPLEMENT DE PROFESSION"], file_df
            ),
            axis=1,
        )
        .tolist()
    )
    voies_check = (
        file_df["NOM DE VOIE"]
        .to_frame()
        .apply(lambda row: check_voie(row.name, row["NOM DE VOIE"], file_df), axis=1)
        .tolist()
    )
    adr_pros_check = (
        file_df["ADRESSE PROFESSIONNELLE"]
        .to_frame()
        .apply(
            lambda row: check_adr_pro(
                row.name, row["ADRESSE PROFESSIONNELLE"], file_df
            ),
            axis=1,
        )
        .tolist()
    )

    error_dict["ERROR_3"] += sum(1 for err in nums_check if not err[0])
    error_dict["ERROR_3"] += sum(1 for err in faubs_check if not err[0])
    error_dict["ERROR_3"] += sum(1 for err in proprs_check if not err[0])
    error_dict["ERROR_3"] += sum(1 for err in situs_check if not err[0])
    error_dict["ERROR_3"] += sum(1 for err in last_names_check if not err[0])
    error_dict["ERROR_3"] += sum(1 for err in first_names_check if not err[0])
    error_dict["ERROR_3"] += sum(1 for err in profs_check if not err[0])
    error_dict["ERROR_3"] += sum(1 for err in comp_profs_check if not err[0])
    error_dict["ERROR_3"] += sum(1 for err in voies_check if not err[0])
    error_dict["ERROR_3"] += sum(1 for err in adr_pros_check if not err[0])

    for i, err in enumerate(nums_check):
        if err[0] == False:
            errors_mat[i, 4] = "ALERT_3"
            errors_mat[i, 10] = f"E_num : {err[1]}"
    for i, err in enumerate(faubs_check):
        if err[0] == False:
            errors_mat[i, 4] = "ALERT_3"
            errors_mat[i, 11] = f"E_faub : {err[1]}"
    for i, err in enumerate(proprs_check):
        if err[0] == False:
            errors_mat[i, 4] = "ALERT_3"
            errors_mat[i, 12] = f"E_propr : {err[1]}"
    for i, err in enumerate(situs_check):
        if err[0] == False:
            errors_mat[i, 4] = "ALERT_3"
            errors_mat[i, 14] = f"E_situa : {err[1]}"
    for i, err in enumerate(last_names_check):
        if err[0] == False:
            errors_mat[i, 4] = "ALERT_3"
            errors_mat[i, 13] = f"E_nom_jf : {err[1]}"
    for i, err in enumerate(first_names_check):
        if err[0] == False:
            errors_mat[i, 4] = "ALERT_3"
            errors_mat[i, 6] = f"E_prenom : {err[1]}"
    for i, err in enumerate(profs_check):
        if err[0] == False:
            errors_mat[i, 4] = "ALERT_3"
            errors_mat[i, 7] = f"E_prof : {err[1]}"
    for i, err in enumerate(comp_profs_check):
        if err[0] == False:
            errors_mat[i, 4] = "ALERT_3"
            errors_mat[i, 8] = f"E_prof : {err[1]}"
    for i, err in enumerate(voies_check):
        if err[0] == False:
            errors_mat[i, 4] = "ALERT_3"
            errors_mat[i, 5] = f"E_nom_v : {err[1]}"
    for i, err in enumerate(adr_pros_check):
        if err[0] == False:
            errors_mat[i, 4] = "ALERT_3"
            errors_mat[i, 15] = f"E_adr_pro : {err[1]}"

    return errors_mat, error_dict


def log_results(resultats, filename):
    with open(filename, "a", encoding="utf-8") as summary_file:

        seuil = 0.05  # 5%

        # Filtrer les fichiers valides (évite division par zéro)
        fichiers_valides = [d for d in resultats if d["LINES_CPT"] > 0]
        nb_fichiers = len(resultats)
        nb_fichiers_ignores = nb_fichiers - len(fichiers_valides)

        total_lines = sum(d["LINES_CPT"] for d in fichiers_valides)

        taux_err1 = [d["ERROR_1"] / d["LINES_CPT"] for d in fichiers_valides]
        taux_err2 = [d["ERROR_2"] / d["LINES_CPT"] for d in fichiers_valides]
        taux_err3 = [d["ERROR_3"] / d["LINES_CPT"] for d in fichiers_valides]

        nb_fichiers_err1_seuil = sum(t > seuil for t in taux_err1)

        global_error1 = sum(d["ERROR_1"] for d in fichiers_valides) / total_lines
        global_error2 = sum(d["ERROR_2"] for d in fichiers_valides) / total_lines
        global_error3 = sum(d["ERROR_3"] for d in fichiers_valides) / total_lines

        mediane_err1 = statistics.median(taux_err1)
        mediane_err2 = statistics.median(taux_err2)
        mediane_err3 = statistics.median(taux_err3)

        prop_err1 = sum(d["ERROR_1"] > 0 for d in fichiers_valides) / len(
            fichiers_valides
        )
        prop_err2 = sum(d["ERROR_2"] > 0 for d in fichiers_valides) / len(
            fichiers_valides
        )
        prop_err3 = sum(d["ERROR_3"] > 0 for d in fichiers_valides) / len(
            fichiers_valides
        )

        # ---------- KPI GLOBAUX ----------
        summary_file.write("=" * 50 + "\n")
        summary_file.write("KPI GLOBAUX\n")
        summary_file.write("=" * 50 + "\n\n")

        if nb_fichiers_ignores:
            summary_file.write(
                f"⚠ {nb_fichiers_ignores} fichier(s) ignoré(s) (LINES_CPT = 0)\n\n"
            )

        summary_file.write(
            f"Fichiers analysés : {len(fichiers_valides)} / {nb_fichiers}\n"
        )
        summary_file.write(f"Nombre total de lignes traitées : {total_lines}\n")
        summary_file.write(
            f"Fichiers avec taux ERROR_1 > {seuil:.0%} : {nb_fichiers_err1_seuil} / {len(fichiers_valides)}\n\n"
        )

        summary_file.write(f"{'':20}{'ERROR_1':>12}{'ERROR_2':>12}{'ERROR_3':>12}\n")
        summary_file.write(
            f"{'Taux global':20}{global_error1:>11.4%}{global_error2:>12.4%}{global_error3:>12.4%}\n"
        )
        summary_file.write(
            f"{'Médiane (taux)':20}{mediane_err1:>11.4%}{mediane_err2:>12.4%}{mediane_err3:>12.4%}\n"
        )
        summary_file.write(
            f"{'% fichiers > 0':20}{prop_err1:>11.2%}{prop_err2:>12.2%}{prop_err3:>12.2%}\n"
        )

        # ---------- DETAILS PAR FICHIER ----------
        summary_file.write("\n" + "=" * 50 + "\n")
        summary_file.write("DÉTAILS PAR FICHIER\n")
        summary_file.write("=" * 50 + "\n\n")

        for i, d in enumerate(resultats, 1):
            summary_file.write(f"[{i}] {d['file']}\n")
            if d["LINES_CPT"] == 0:
                summary_file.write(
                    "    Lignes traitées : 0 (fichier ignoré des stats globales)\n\n"
                )
                continue
            summary_file.write(f"    Lignes traitées : {d['LINES_CPT']}\n")
            summary_file.write(
                f"    ERROR_1 : {d['ERROR_1']:>4} ({d['ERROR_1']/d['LINES_CPT']:.4%})\n"
            )
            summary_file.write(
                f"    ERROR_2 : {d['ERROR_2']:>4} ({d['ERROR_2']/d['LINES_CPT']:.4%})\n"
            )
            summary_file.write(
                f"    ERROR_3 : {d['ERROR_3']:>4} ({d['ERROR_3']/d['LINES_CPT']:.4%})\n\n"
            )


##########################################

# CSV_FOLDER = "ctrl_files/csv/"
TEXT_FOLDER = "echantillon/extract/"
DEST_FOLDER = "ctrl_files/results/"
REDO_FOLDER = "ctrl_files/redo/"
IMAGES_FOLDER_BASE = "Annuaires/AMC67482_1BA_00"
MODEL_RES_LOG_FILE = "log.txt"

LAST_NAMES_FILE = "evaluation_script/data/nom_.txt"
FIRST_NAME_FILE = "evaluation_script/data/prenom_.txt"
PROF_FILE = "evaluation_script/data/profession_.txt"
VOIE_FILE = "evaluation_script/data/nom_voie_.txt"
ADRESSE_PRO_FILE = "evaluation_script/data/adresse_pro_.txt"

NAMES_SET = set()
FIRST_NAMES_SET = set()
PROF_SET = set()
VOIE_SET = set()
ADRESSE_PRO_SET = set()

with open(LAST_NAMES_FILE, "r", encoding="utf-8") as f:
    for line in f:
        NAMES_SET.add(normalize(line))

with open(FIRST_NAME_FILE, "r", encoding="utf-8") as f:
    for line in f:
        FIRST_NAMES_SET.add(normalize(line))

with open(PROF_FILE, "r", encoding="utf-8") as f:
    for line in f:
        PROF_SET.add(normalize(line))

with open(VOIE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        VOIE_SET.add(normalize(line))

with open(ADRESSE_PRO_FILE, "r", encoding="utf-8") as f:
    for line in f:
        ADRESSE_PRO_SET.add(normalize(line))

# files = sorted(glob.glob(CSV_FOLDER + "*.csv"))

LINES_CPT = 0
# FILES_CPT = len(files)
ERROR_1_CPT = 0
ERROR_2_CPT = 0
ERROR_3_CPT = 0
ERROR_1_FILES_CPT = 0
ERROR_2_FILES_CPT = 0
ERROR_3_FILES_CPT = 0
