# Latenz-Messungen – Simulator vs. Reales Gerät

## ⚠️ Wichtiger Hinweis

Die in diesem Ordner enthaltenen Latenz-Daten stammen aus dem **iOS-Simulator**
(macOS-Host), NICHT von einem realen iPhone 14.

Die **realen Messdaten vom iPhone 14 (A15 Bionic)** sind als Konsolenprotokolle archiviert in:
- `../results/metrics/batch_evaluation_iphone14_fallback.txt` (Fallback-ROI, 100 Bilder)
- `../results/metrics/batch_evaluation_iphone14_car_det_roi.txt` (Car-Detection-ROI, 100 Bilder)

Diese Protokolle enthalten die vollständigen Per-Image-Ergebnisse (GT, Prediction, OK/Fail)
sowie die Stage-Latenzen und belegen alle Werte in Thesis-Tabellen 15–17 exakt.

## Dateien

| Datei | Gerät (Simulator-UUID) | Pipeline |
|-------|----------------------|----------|
| `device_79B1_5stage_stats.json` | 79B1A960 (iPhone 16 Pro Max) | 5-Stufen Car-Det → ROI |
| `device_79B1_5stage_per_image.csv` | 79B1A960 | Per-Image-Breakdown (n=100) |
| `device_1A3C_fallback_stats.json` | 1A3C4727 (iPhone 16 Pro Max) | 3-Stufen Fallback |
| `device_1A3C_fallback_per_image.csv` | 1A3C4727 | Per-Image-Breakdown (n=100) |

## Vergleich: Simulator vs. Thesis (Reales Gerät)

### Car-Det → ROI Pipeline (5 Stufen)

| Stufe | Simulator (mean, ms) | Thesis / iPhone 14 (ms) | Faktor |
|-------|---------------------|------------------------|--------|
| YOLO Car Detection | 78.63 | 21.66 | ~3.6× |
| ROI Extraction | 0.01 | 0.01 | ~1× |
| MobileNet Classification | 14.94 | 5.34 | ~2.8× |
| YOLO EV Detection | 196.98 | 35.95 | ~5.5× |
| **Total E2E** | **290.56** | **62.95** | **~4.6×** |

### Fallback Pipeline (3 Stufen)

| Stufe | Simulator (mean, ms) | Thesis / iPhone 14 (ms) | Faktor |
|-------|---------------------|------------------------|--------|
| MobileNet Classification | 16.10 | 5.34* | ~3.0× |
| YOLO EV Detection | 210.15 | 35.95* | ~5.8× |
| **Total E2E** | **226.25** | **41.93** | **~5.4×** |

*Thesis-Werte für MNV3-Large und YOLO12m aus Tabelle 5.2

### Einordnung
Der Simulator emuliert die ARM-Architektur auf x86/ARM-Mac-Hardware ohne
Neural-Engine-Beschleunigung. Deshalb liegen die Simulator-Werte systematisch
3–6× höher als die Messungen auf dem realen A15 Bionic mit ANE-Inferenz.
Die relativen Verhältnisse zwischen den Stufen sind jedoch konsistent.
