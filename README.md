# Person Tracking with Kalman Filter and ResNet Appearance Embeddings

This project explores practical multi-person tracking using motion-based prediction (Kalman filter) combined with appearance-based Re-ID using ResNet embeddings. The main objective is to observe how well the system can **maintain identity consistency (tracking and re-identification)** even when detections intermittently drop or targets cross paths.

The experiment demonstrates that **the tracking and Re-ID pipeline performs reliably when detections are available**. Identity switches remain low, and tracks remain stable. The main bottleneck is **detection recall**, not the tracking or matching pipeline.

---

## Overview

**Core Components**
- Detector: FRCNN (MOT17 public detections)
- Motion Model: Kalman Filter (constant velocity)
- Appearance Matching: ResNet-50 embeddings + cosine distance
- Association: Hungarian algorithm (cost = motion + embedding similarity)

**Project Goals**
- Maintain identity across occlusions, exits, re-entries, and crossings  
- Minimize ID switching  
- Test viability of combining Kalman + CNN embeddings for Re-ID  
- Provide interpretable metrics and analysis  

---

## Benchmark Evaluation

**Dataset**: MOT17 (train split, first 100 frames per sequence)  
**Sequences tested**: 02, 04, 05, 09, 10, 11, 13  
**Detection source**: MOT17 FRCNN public detections  
**Hardware**: GPU recommended (CPU works at reduced FPS)

### Overall Metrics

| Metric | Value |
|--------|------:|
| **MOTA** | 27.31% |
| **MOTP** | 10.69 px |
| **Precision** | 94.07% |
| **Recall** | 29.67% |
| **F1 Score** | 42.17% |
| **ID Switches** | 17 total |
| **Fragmentations** | 119 total |
| **Average FPS** | 1.63 |

High precision (94%) and very low ID switch count confirm that **when detections exist, identities remain stable and rarely get confused.** Recall is low because the detector misses many targets, but tracking quality is strong when detections are present.

---

### Per-Sequence Results

| Seq | MOTA (%) | MOTP (px) | Precision (%) | Recall (%) | F1 (%) | ID Sw | Frag | FPS |
|----:|---------:|----------:|--------------:|-----------:|-------:|------:|-----:|----:|
| 02 | 17.72 | 6.93 | 98.30 | 18.03 | 30.47 | 0 | 17 | 2.04 |
| 04 | 9.77 | 9.84 | 85.74 | 11.77 | 20.70 | 2 | 15 | 1.34 |
| 05 | 47.83 | 22.43 | 90.16 | 54.13 | 67.65 | 2 | 11 | 2.06 |
| 09 | 50.07 | 11.45 | 91.54 | 55.17 | 68.85 | 0 | 6 | 0.75 |
| 10 | 24.36 | 7.15 | 96.58 | 25.64 | 40.52 | 6 | 32 | 1.27 |
| 11 | 32.27 | 13.11 | 97.05 | 33.58 | 49.89 | 4 | 10 | 1.26 |
| 13 | 9.14 | 3.90 | 99.14 | 9.34 | 17.07 | 3 | 28 | 2.64 |

**Highlight:** Sequences 05 and 09 achieve both high Recall (~55%) and excellent Tracking Quality (F1 ~68%) with very low identity confusion (0–2 ID switches).  
This is the clearest support that the **tracking + Re-ID pipeline works well when detections are present.**

---

## Interpretation

**What works well**
- Strong identity preservation due to ResNet embeddings and effective assignment logic
- Very low ID switch count across all sequences
- Consistent precision above 90% (correct detections are almost always assigned correctly)
- Kalman motion prediction maintains track continuity across short occlusions

**What limits performance**
- Low recall from FRCNN public detections (only ~29%) causes track dropouts
- Fragmentations rise mainly when detections disappear temporarily
- FPS is limited due to per-frame embedding computation

**Core takeaway**  
➡ The Kalman + ResNet-based tracking pipeline is effective at **identity stability and re-identification**, but overall performance is detection-limited.  
➡ When it sees, it tracks.

---
## How to run

1. Clone the repository
```bash
git clone https://github.com/giosanchez0208/Person-Tracking-with-Kalman-Filter-and-Resnet-Embeddings.git
cd Person-Tracking-with-Kalman-Filter-and-Resnet-Embeddings
```

2. Create and activate a Python virtual environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
```

3. Install dependencies and download models
```bash
pip install -r requirements.txt
python setup.py        # runs repository setup and downloads pretrained models
```

4. Run on a demo video or MOT sequence
```bash
python main.py --video path/to/video.mp4
```

5. Evaluate on MOT17 (example)
```bash
python eval_mot.py \
    --dataset /path/to/MOT17/train \
    --frames 100 \
    --detection-format frcnn
```

## What I learned
- Combining motion prediction (Kalman) with deep appearance embeddings improves identity consistency.  
- Kalman alone struggles with occlusions; visual Re-ID reduces ID switches during crossings and re-entries.  
- Detector recall is the primary bottleneck for end-to-end tracking performance.

## Future improvements
- Improve recall — use a stronger detector (e.g., fine-tuned YOLO variant).  
- Faster embeddings — batch crop processing on GPU.  
- Reduce fragmentation — increase track age, refine gating thresholds.  
- Better metrics — add IDF1, HOTA, and TA-ID evaluations.  
- Visualization — save annotated videos and sample GIFs for inspection.

## Notes
- Activate the virtual environment before running scripts.  
- `setup.py` in this repo handles model downloads; adjust paths in scripts if using custom data locations.

## Keywords
multi-object-tracking kalman-filter resnet deep-reid identity-tracking opencv feature-embeddings tracking-by-detection MOT17 computer-vision
