# Datensätze – Übersicht und Reproduzierbarkeit

Dieses Verzeichnis enthält die **Konfigurationsdateien und Metadaten** der für die Thesis verwendeten Datensätze.
Die Rohdaten sind aufgrund ihrer Größe (> 30 GB) nicht enthalten, können aber aus den unten genannten Quellen bezogen werden.

---

## 1. Textur-Klassifikationsdatensatz (`texture/`)

### Quelle und Aufbereitung
Der Datensatz wird aus zwei öffentlich verfügbaren Quellen zusammengesetzt:

| Quelle | Beschreibung | Lizenz |
|--------|-------------|--------|
| **RSCD** (Road Surface Condition Dataset) | ~1 Mio. 75×75 px Patches, 27 Klassen | [Zhao et al., 2022/2023] |
| **RoadSaW** | Multi-Resolution-Patches (75/150/225 px), 12 Klassen | [Cordes et al., CVPRW 2022] |

Das Skript `prepare_texture_dataset.py` (→ `code/utils/`) führt folgende Schritte durch:
1. Canonical Mapping: 27 RSCD-Klassen + 12 RoadSaW-Klassen → 4 Zielklassen (`asphalt`, `cobblestone`, `gravel`, `sand`)
2. Zufälliger 80/10/10-Split (seed=42)
3. Generierung von `meta.json` mit exakten Counts pro Split und Klasse

### Trainierter Datensatz: `texture_data_final`
- **Gesamtzahl:** 524.998 Bilder
- **Split:** Train 419.997 / Val 52.498 / Test 52.503
- **Datei:** [`texture/meta.json`](texture/meta.json)

| Klasse | Train | Val | Test | Gesamt |
|--------|-------|-----|------|--------|
| asphalt | 245.780 | 30.722 | 30.723 | 307.225 |
| cobblestone | 4.185 | 523 | 524 | 5.232 |
| gravel | 85.083 | 10.635 | 10.636 | 106.354 |
| sand | 84.949 | 10.618 | 10.620 | 106.187 |

### Robustness-Evaluation
Für die Robustheitsanalyse (Kapitel 6.6) wurde ein separater Evaluationspfad verwendet.
Dabei kam ein gesondert vorbereiteter Texturdatensatz (`texture_data_thesis`, 424.093 Bilder)
zum Einsatz, aus dessen Val-Split ein stratifiziertes Subset von 400 Bildern
(100 pro Klasse, seed=42) gezogen wurde.

**Hinweis:** Training und Offline-Test-Evaluation erfolgten auf `texture_data_final` (524.998).
Die Robustheitsanalyse verwendet denselben trainierten Checkpoint, evaluiert aber auf dem
oben genannten Subset aus `texture_data_thesis`. Diese Auswertung dient dem relativen
Robustheitsvergleich innerhalb desselben Setups und ist nicht direkt mit der
kanonischen Offline-Patch-Baseline vergleichbar.

### Reproduktion
> **Hinweis:** Die CLI der im Repository enthaltenen Skripte entspricht dem aktuellen Stand
> und kann von den in der Thesis verkürzt dargestellten Beispielaufrufen (Abschnitt 4.7) abweichen.

```bash
python prepare_texture_dataset.py \
    --rscd-root <pfad>/RSCD \
    --roadsaw-root <pfad>/RoadSaW \
    --out texture_data_final \
    --seed 42 --split 0.8 0.1 0.1
```

---

## 2. EVCS-Detektionsdatensatz (`evcs/`)

### Quelle
| Quelle | Version | Bilder | Annotationen |
|--------|---------|--------|-------------|
| VISCODA EVCS Dataset Part A | V1.1 | 3.645 | 6.979 |
| VISCODA EVCS Dataset Part B | V1.1 | 759 | 759 |
| **Gesamt** | | **4.404** | **7.738** |

> **Zur Thesis:** In Abschnitt 4.2 wird vereinfachend formuliert, Part A umfasse 4.404 Bilder.
> Gemeint ist der Gesamt-Datensatz (Part A + Part B). Die korrekten Teilmengen sind oben aufgeschlüsselt.

### Aufbereitung
Das Skript `prepare_evcs_dataset.py` (→ `code/utils/`) konvertiert die VISCODA-Annotationen
ins Ultralytics-YOLO-Format und führt einen **site-basierten Split** durch:
- Bilder desselben Standorts bleiben im selben Split (kein Data Leakage)
- Resultat: Train 3.512 / Val 495 / Test 397

### Konfigurationsdateien
- [`evcs/data.yaml`](evcs/data.yaml) — Ultralytics-Konfiguration (site-basiert, für Training verwendet)
- [`evcs/meta.json`](evcs/meta.json) — Datensatz-Metadaten (Counts, Splits, Quellinfo)

### Reproduktion
> **Hinweis:** Die CLI kann von den in der Thesis (Abschnitt 4.7) gezeigten Beispielaufrufen abweichen.

```bash
python prepare_evcs_dataset.py \
    --part-a <pfad>/EVCSDataset_VISCODA_V1.1_PartA \
    --part-b <pfad>/EVCSDataset_VISCODA_V1.1_PartB \
    --out evcs_yolo_site_based \
    --split-mode site
```

---

## 3. Zuordnung Datensatz → Modell

| Modell | Datensatz | data_root (aus train_log) |
|--------|-----------|--------------------------|
| 01_baseline (MNV3-Small) | texture_data_final | ✓ Verifiziert |
| 02_class_weighted (MNV3-Small) | texture_data_final | ✓ Verifiziert |
| 03_robust_augmentation (MNV3-Small) | texture_data_final | ✓ Verifiziert |
| 04_large_baseline (MNV3-Large) | texture_data_final | ✓ Verifiziert |
| yolo_baseline (YOLO12m) | evcs_yolo_site_based | ✓ Verifiziert (args.yaml) |
| yolo_robust (YOLO12m) | evcs_yolo_site_based | ✓ Verifiziert (args.yaml) |

Alle `data_root`-Pfade wurden gegen die `train_log.json` bzw. `args.yaml` der jeweiligen
Modellvarianten verifiziert.
