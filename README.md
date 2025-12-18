# THESIS_Cellpainting

This repository contains code developed for my thesis project on supervised morphological profiling
using large-scale Cell Painting mitochondrial images.

## Contents

- `supervised/`
  - `cp_pipeline_v3_virtualtiles_noleak_wellsplit_maandag.py`
  - Supervised pipeline for leakage-free well-level splitting, virtual tiling, training, and evaluation.

- `analysis/`
  - `disulfiram_vector_similarity_analysis.py`
  - Downstream similarity analysis based on learned representations.

- `preprocessing/`
  - `convert_tiffs_to_pngs.py`
  - Utility script to convert TIFF microscopy images to normalized PNGs for lightweight storage and
    downstream tiling/training.

## Preprocessing: TIFF → PNG

Convert a folder of TIFF images to normalized PNGs (percentile normalization per channel):

```bash
python preprocessing/convert_tiffs_to_pngs.py \
  --in_root  /scratch-shared/$USER/cellpainting/images \
  --out_root /scratch-shared/$USER/cellpainting/preprocessed_png \
  --pattern  "**/*.tif*" \
  --skip_existing
