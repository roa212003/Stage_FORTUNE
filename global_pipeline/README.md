# Pipeline d'extraction — Documentation

## Vue d'ensemble

Le pipeline orchestre trois processus parallèles (extraction, validation, OCR) tournant sur des threads distincts.

## Fonctionnement

1. Les fichiers `.txt` sont chargés dans la file d'extraction ;
2. Après extraction, le fichier CSV primaire (sauvegardé dans `echantillon/extract/csv/` et les fichiers .txt utilisés sont dans `echantillon/extract/txt/`) est envoyé dans la file de validation ;
3. Si le fichier est **validé**, le CSV est supprimé de `echantillon/extract/csv/` et la version finale avec alertes est sauvegardée dans `ctrl_files/results/` ;
4. Si le fichier **n'est pas validé**, l'image est copiée depuis `Annuaire/` vers `ctrl_files/redo/` et ajoutée dans la file d'extraction OCR ;
5. Une fois l'OCR effectué, le fichier est remis dans la file d'extraction texte-CSV à l'étape 1.

## Commandes

**Lancer le pipeline :**
```bash
nohup python global_pipeline.py > pipeline_log.txt 2>&1 &
```

**Consulter les logs :**
```bash
tail -f log.txt
```

**Consulter l'évolution et le temps restant :**
```bash
tail -f pipeline_log.txt
```

## Accès au serveur

Les images des annuaires sont chargées sur le serveur LITIS. Accès via :

[https://litis-jupyterhub.univ-rouen.fr/](https://litis-jupyterhub.univ-rouen.fr/)
