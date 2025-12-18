#!/usr/bin/env python3
"""
cp_pipeline_v3_virtualtiles_noleak_wellsplit_maandag.py
=======================================================

Zelfde functionaliteit als cp_pipeline_v3_virtualtiles.py, maar MET
image-/well-niveau splits om data leakage te voorkomen.

Belangrijkste punten:
- Voor elke drug nemen we de unieke src_image's voor treated (label=1)
  en MOCK (label=0).
- We leiden hieruit een 'well' af op basis van de bestandsnaam.
- Die wells worden per label gesplitst in ongeveer 70% train, 20% val,
  10% test.
- ALLE tiles van één well gaan naar dezelfde split (dus niet over
  train/val/test gemengd).
- Drugs waarvoor we niet minimaal 1 well in train/val/test hebben
  voor zowel treated als mock, worden overgeslagen.
- Daarna wordt zoals eerder met virtuele tiles gewerkt.

CLI-voorbeeld:

    python cp_pipeline_v3_virtualtiles_noleak_wellsplit_maandag.py \
      --manifest_csv /scratch-shared/$USER/cellpainting/combined_manifest_from_images.csv \
      --out_root     /scratch-shared/$USER/cellpainting/v3_virtualtiles_run_noleak \
      --models       resnetv2_101 \
      --rows         3 \
      --cols         4 \
      --img_size     224 \
      --batch        64 \
      --lr           1e-3 \
      --epochs       20 \
      --max_drugs    100
"""

import os
import re
import time
import math
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

import cv2

# OpenCV-warnings dempen (zoals "can't open/read file")
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

try:
    import tifffile as tiff
except Exception:
    tiff = None

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    import timm  # needed for ResNetV2 models
except Exception:
    timm = None

# ---------- GLOBALS / HELPERS ----------

USER = os.environ.get("USER", "")
HOME = Path.home()
THESIS = HOME / "THESIS"


def now_tag():
    return time.strftime("%Y%m%d_%H%M%S")


BRD_RE = re.compile(r"(BRD-[A-Z0-9]+)")


def msg(s: str):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm_id(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    m = BRD_RE.search(s)
    return m.group(1) if m else s


def extract_well_id(path: str) -> str:
    """
    Probeer een 'well-id' uit de bestandsnaam te halen.

    Voorbeelden die hiermee werken:
      ..._w05d3_...
      ..._wA01_...
      ..._wB12_s3...

    We zoeken eerst een '_w...' patroon en nemen alles tot de volgende '_'.
    Als dat niet lukt, geven we de hele bestandsnaam terug (fall-back).
    """
    name = Path(path).stem
    m = re.search(r"_w([^_]+)", name)
    if m:
        return m.group(1)
    return name


def imread_retry(path: str, retries: int = 5, delay: float = 0.2):
    for _ in range(retries):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
        time.sleep(delay)
    return None


# ---------- MODEL (GEKOPIEERD UIT v3) ----------

def _normalize_model_name(name: str) -> str:
    s = name.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    if s == "resnett18":
        s = "resnet18"  # alias
    return s


def build_model(name, num_classes=2):
    """
    Zelfde implementatie als in cp_pipeline_v3:
    - Torchvision ResNet V1
    - ResNetV2 via timm (resnetv2_50, resnetv2_101, etc.).
    """
    s = _normalize_model_name(name)
    # Torchvision ResNets
    tv_names = {"resnet18", "resnet34", "resnet50", "resnet101", "resnet152"}
    if s in tv_names:
        from torchvision import models
        ctor = getattr(models, s)
        weights_attr = f"{s.capitalize()}_Weights"
        weights = getattr(models, weights_attr).IMAGENET1K_V1 if hasattr(models, weights_attr) else None
        m = ctor(weights=weights)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m

    # ResNetV2 via timm
    if "v2" in s:
        if timm is None:
            raise ValueError("Model requires timm. Please install: pip install timm")
        depth = s.replace("resnet", "").replace("v2", "")
        timm_id = f"resnetv2_{depth}"
        try:
            return timm.create_model(timm_id, pretrained=True, num_classes=num_classes)
        except RuntimeError as e:
            msg(f"[WARN] {e} — falling back to pretrained=False for {timm_id}.")
            return timm.create_model(timm_id, pretrained=False, num_classes=num_classes)

    raise ValueError(f"Unknown model: {name}. For *V2 models install timm.")


# ---------- METRICS / EVAL (GEKOPIEERD UIT v3, UITGEBREID MET CONFUSION) ----------

def _acc_f1_from_counts(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    acc = (tp + tn) / max(1, total)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)
    return acc, f1


@torch.no_grad()
def evaluate_metrics(model, loader, device):
    """
    Geeft terug:
      loss, acc, f1, tp, fp, tn, fn
    """
    model.eval()
    crit = nn.CrossEntropyLoss()
    loss_sum = 0.0
    nseen = 0
    tp = fp = tn = fn = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            out = model(x)
            loss = crit(out, y)
        loss_sum += float(loss) * x.size(0)
        pred = out.argmax(1)
        tp += int(((pred == 1) & (y == 1)).sum().item())
        tn += int(((pred == 0) & (y == 0)).sum().item())
        fp += int(((pred == 1) & (y == 0)).sum().item())
        fn += int(((pred == 0) & (y == 1)).sum().item())
        nseen += x.size(0)
    acc, f1 = _acc_f1_from_counts(tp, fp, tn, fn)
    return loss_sum / max(1, nseen), acc, f1, tp, fp, tn, fn


# ---------- TILE-CROP LOGICA ----------

def crop_tile(img: np.ndarray, row: int, col: int, rows: int, cols: int):
    """
    Snij één tile uit:
      - hoogte en breedte integer verdeeld
      - laatste rij/kolom pakt 'de rest' mee (zodat randen niet wegvallen).
    """
    H, W = img.shape[:2]
    th, tw = H // rows, W // cols
    y0 = row * th
    y1 = (row + 1) * th if row < rows - 1 else H
    x0 = col * tw
    x1 = (col + 1) * tw if col < cols - 1 else W
    return img[y0:y1, x0:x1].copy()


# ---------- DATASET: VIRTUELE TILES ----------

class VirtualTilesDataset(Dataset):
    """
    Dataset die een virtuele tile-manifest gebruikt.

    Verwacht DataFrame met ten minste:
      - src_image : pad naar de grote PNG
      - label     : 0 of 1
      - row, col  : 0-based tile-coördinaten
    """
    def __init__(self, df: pd.DataFrame, img_size: int, rows: int, cols: int):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.rows = rows
        self.cols = cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        path = r["src_image"]
        y = int(r["label"])

        img = imread_retry(path, retries=3, delay=0.1)
        if img is None:
            # fallback: zwart beeld (zou nu zeldzaam moeten zijn,
            # omdat we in main al niet-bestaande paden filteren)
            img = np.zeros((self.img_size, self.img_size, 3), np.uint8)

        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, 2)

        tile = crop_tile(img, int(r["row"]), int(r["col"]), self.rows, self.cols)
        tile = cv2.resize(tile, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        # HWC -> CHW, [0..255] -> [0..1] float
        tile = torch.from_numpy(tile.transpose(2, 0, 1)).float() / 255.0
        return tile, torch.tensor(y, dtype=torch.long)


# ---------- BALANCED VIRTUAL TILE-MANIFEST ----------

def build_balanced_virtual_manifest(df_images: pd.DataFrame, rows: int, cols: int):
    """
    Bouwt een balanced virtual tiles manifest.

    Input:
      df_images met kolommen:
        - 'broad_id' (drug-id; MOCK voor controles)
        - 'src_image' (pad naar PNG)

    Output:
      full_df: DataFrame met per tile:
        - src_image
        - broad_id
        - group   (drug-id; ook voor MOCK tiles van die drug)
        - label   (1=treated, 0=MOCK)
        - row, col, tile_id
      per_drug_summary: DataFrame met aantallen per drug.
    """
    df = df_images.copy()
    df["broad_id"] = df["broad_id"].astype(str).map(norm_id)
    df["broad_id"] = df["broad_id"].replace({"MOCK": "MOCK", "mock": "MOCK", "Mock": "MOCK"})

    treated = df[df["broad_id"] != "MOCK"].copy()
    mock = df[df["broad_id"] == "MOCK"].copy()

    if treated.empty or mock.empty:
        raise SystemExit("[ERROR] Need both treated and MOCK images in manifest.")

    treated_by_img = treated.groupby(["broad_id", "src_image"]).size()
    mock_by_img = mock.groupby("src_image").size()

    rng = np.random.default_rng(42)
    rows_out = []
    per_drug_summary = []

    mock_imgs = list(mock_by_img.index.unique())
    total_tiles_per_image = rows * cols

    for bid in sorted(treated["broad_id"].unique()):
        # alle treated images voor deze drug
        try:
            t_sub = treated_by_img.loc[bid]
        except KeyError:
            continue

        treated_imgs = sorted(
            t_sub.index.unique().tolist()
        ) if isinstance(t_sub, pd.Series) else [t_sub.index]

        n_images = len(treated_imgs)
        if n_images == 0:
            continue

        # sample n_images MOCK-afbeeldingen (zoals v3)
        replace = len(mock_imgs) < n_images
        pick_idx = rng.choice(len(mock_imgs), size=n_images, replace=replace)
        picked_mock_imgs = [mock_imgs[i] for i in pick_idx]

        treated_tiles_all = 0
        mock_tiles_all = 0

        # treated tiles
        for src in treated_imgs:
            for r in range(rows):
                for c in range(cols):
                    tile_id = r * cols + c + 1
                    rows_out.append(
                        dict(
                            src_image=src,
                            broad_id=bid,
                            group=bid,
                            label=1,
                            row=r,
                            col=c,
                            tile_id=tile_id,
                        )
                    )
                    treated_tiles_all += 1

        # mock tiles
        for src in picked_mock_imgs:
            for r in range(rows):
                for c in range(cols):
                    tile_id = r * cols + c + 1
                    rows_out.append(
                        dict(
                            src_image=src,
                            broad_id="MOCK",
                            group=bid,
                            label=0,
                            row=r,
                            col=c,
                            tile_id=tile_id,
                        )
                    )
                    mock_tiles_all += 1

        per_drug_summary.append(
            dict(
                broad_id=bid,
                treated_images=n_images,
                mock_images=n_images,
                treated_tiles=treated_tiles_all,
                mock_tiles=mock_tiles_all,
                tiles_per_image=total_tiles_per_image,
            )
        )

    full_df = pd.DataFrame(rows_out)
    sum_df = pd.DataFrame(per_drug_summary)
    return full_df, sum_df


# ---------- WELL-LEVEL SPLIT ----------

def split_wells_70_20_10(wells, rng):
    """
    Verdeel een lijst well-id's in ongeveer 70% train, 20% val, 10% test,
    met zoveel mogelijk:
      - minimaal 1 train / 1 val / 1 test.
    """
    wells = list(wells)
    rng.shuffle(wells)
    n = len(wells)

    if n == 0:
        return [], [], []
    if n == 1:
        return wells, [], []          # 1 well → alleen train
    if n == 2:
        return wells[:1], [], wells[1:]    # 1 train, 1 test
    if n == 3:
        return wells[:1], wells[1:2], wells[2:]  # 1/1/1

    # n >= 4 → echte 70/20/10
    n_train = max(1, int(round(0.70 * n)))
    n_val   = max(1, int(round(0.20 * n)))
    n_test  = n - n_train - n_val

    if n_test < 1:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        else:
            n_train -= 1

    train = wells[:n_train]
    val   = wells[n_train:n_train + n_val]
    test  = wells[n_train + n_val:]

    return train, val, test


# ---------- TRAIN PER-DRUG (MET WELL-LEVEL SPLITS) ----------

def train_per_drug_virtual(
    full_df: pd.DataFrame,
    out_root: Path,
    models: list,
    img_size: int,
    rows: int,
    cols: int,
    batch: int,
    lr: float,
    epochs: int,
    max_drugs=None,
):
    """
    Zelfde logica als train_per_drug in cp_pipeline_v3, maar met virtuele tiles
    en WELL-level splits:
      - per drug: 70/20/10 split op well-niveau voor label=1 en label=0
      - alle tiles van wells in train/val/test gaan naar de betreffende split.
      - als we niet minimaal 1 well in train/val/test hebben voor zowel
        treated als mock, wordt de drug overgeslagen.
    """
    out_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    msg(f"Using device: {device}")

    drugs = sorted([g for g in full_df["group"].unique() if g != "MOCK"])
    if max_drugs is not None:
        drugs = drugs[:max_drugs]

    msg(f"Training per drug: {len(drugs)} drugs, models: {models}")

    results_long_rows = []

    for drug in drugs:
        msg("-" * 60)
        msg(f"Drug {drug}")
        sub = full_df[full_df["group"] == drug].reset_index(drop=True)

        if sub.empty:
            msg("[WARN] No tiles for this drug; skipping.")
            continue

        # WELL-kolom (afgeleid uit src_image)
        sub = sub.copy()
        sub["well"] = sub["src_image"].apply(extract_well_id)

        rng = np.random.default_rng(42)

        pos_wells = sorted(sub[sub["label"] == 1]["well"].unique().tolist())
        neg_wells = sorted(sub[sub["label"] == 0]["well"].unique().tolist())

        if len(pos_wells) < 3 or len(neg_wells) < 3:
            msg(
                f"[WARN] Not enough wells for proper 70/20/10 split "
                f"(pos_wells={len(pos_wells)}, neg_wells={len(neg_wells)}); skipping drug."
            )
            continue

        tr_pos_w, val_pos_w, te_pos_w = split_wells_70_20_10(pos_wells, rng)
        tr_neg_w, val_neg_w, te_neg_w = split_wells_70_20_10(neg_wells, rng)

        # check minimaal 1 well in iedere split voor zowel pos als neg
        if (len(tr_pos_w) == 0 or len(val_pos_w) == 0 or len(te_pos_w) == 0 or
            len(tr_neg_w) == 0 or len(val_neg_w) == 0 or len(te_neg_w) == 0):
            msg(
                "[WARN] After split, one of the splits has zero wells for pos/neg; "
                "skipping this drug."
            )
            continue

        def pick(df, wells):
            return df[df["well"].isin(wells)]

        train_df = pd.concat(
            [
                pick(sub[sub["label"] == 1], tr_pos_w),
                pick(sub[sub["label"] == 0], tr_neg_w),
            ],
            ignore_index=True,
        ).sample(frac=1.0, random_state=42)

        val_df = pd.concat(
            [
                pick(sub[sub["label"] == 1], val_pos_w),
                pick(sub[sub["label"] == 0], val_neg_w),
            ],
            ignore_index=True,
        ).sample(frac=1.0, random_state=42)

        test_df = pd.concat(
            [
                pick(sub[sub["label"] == 1], te_pos_w),
                pick(sub[sub["label"] == 0], te_neg_w),
            ],
            ignore_index=True,
        ).sample(frac=1.0, random_state=42)

        msg(
            f"Well counts (pos/neg): "
            f"train={len(tr_pos_w)}/{len(tr_neg_w)}, "
            f"val={len(val_pos_w)}/{len(val_neg_w)}, "
            f"test={len(te_pos_w)}/{len(te_neg_w)}"
        )
        msg(
            f"Tile counts: train={len(train_df)} val={len(val_df)} test={len(test_df)}"
        )

        # ========= DATA LOADERS =========
        train_loader = DataLoader(
            VirtualTilesDataset(train_df, img_size, rows, cols),
            batch_size=batch,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            VirtualTilesDataset(val_df, img_size, rows, cols),
            batch_size=batch,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        test_loader = DataLoader(
            VirtualTilesDataset(test_df, img_size, rows, cols),
            batch_size=batch,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # ========= TRAINING PER MODEL =========
        for model_name in models:
            msg(f"[{drug}] Model {model_name}")
            set_seed(42)
            model = build_model(model_name, num_classes=2).to(device)
            opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            crit = nn.CrossEntropyLoss()
            scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

            best_val_acc, best_val_f1, best_state = -1.0, 0.0, None
            best_val_tp = best_val_fp = best_val_tn = best_val_fn = 0

            for epoch in range(1, epochs + 1):
                # ---- TRAIN ----
                model.train()
                loss_sum = 0.0
                correct = 0
                total = 0
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                        out = model(xb)
                        loss = crit(out, yb)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    loss_sum += float(loss) * xb.size(0)
                    correct += (out.argmax(1) == yb).sum().item()
                    total += xb.size(0)

                tr_loss = loss_sum / max(1, total)
                tr_acc = correct / max(1, total)

                # ---- VAL ----
                val_loss, val_acc, val_f1, val_tp, val_fp, val_tn, val_fn = evaluate_metrics(
                    model, val_loader, device
                )

                msg(
                    f"{model_name} | {drug} | epoch {epoch}/{epochs} | "
                    f"train {tr_loss:.4f}/{tr_acc:.3f} | "
                    f"val {val_loss:.4f}/{val_acc:.3f} (F1={val_f1:.3f}) "
                    f"[VAL TP={val_tp} FP={val_fp} TN={val_tn} FN={val_fn}]"
                )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_f1 = val_f1
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    best_val_tp, best_val_fp, best_val_tn, best_val_fn = (
                        val_tp, val_fp, val_tn, val_fn
                    )

            # ---- TEST METRICS MET BESTE STATE ----
            if best_state is not None:
                model.load_state_dict(best_state)
            test_loss, test_acc, test_f1, test_tp, test_fp, test_tn, test_fn = evaluate_metrics(
                model, test_loader, device
            )
            msg(
                f"{model_name} | {drug} | BEST val_acc={best_val_acc:.3f} "
                f"(val_f1={best_val_f1:.3f}) | test_acc={test_acc:.3f} "
                f"(test_f1={test_f1:.3f})"
            )
            msg(
                f"[CONFUSION] VAL:  TP={best_val_tp} FP={best_val_fp} "
                f"TN={best_val_tn} FN={best_val_fn}"
            )
            msg(
                f"[CONFUSION] TEST: TP={test_tp} FP={test_fp} "
                f"TN={test_tn} FN={test_fn}"
            )

            # save model onder out_root/models/<drug>/
            out_dir = out_root / "models" / drug
            out_dir.mkdir(parents=True, exist_ok=True)
            model_path = out_dir / f"{_normalize_model_name(model_name)}_best.pt"
            try:
                torch.save(model.state_dict(), model_path)
            except Exception as e:
                msg(f"[WARN] Failed to save model for {drug}: {e}")

            results_long_rows.append(
                dict(
                    drug=drug,
                    model=model_name,
                    val_best_acc=round(float(best_val_acc), 4),
                    val_best_f1=round(float(best_val_f1), 4),
                    test_acc=round(float(test_acc), 4),
                    test_f1=round(float(test_f1), 4),
                    n_train=len(train_df),
                    n_val=len(val_df),
                    n_test=len(test_df),
                    val_tp=best_val_tp,
                    val_fp=best_val_fp,
                    val_tn=best_val_tn,
                    val_fn=best_val_fn,
                    test_tp=test_tp,
                    test_fp=test_fp,
                    test_tn=test_tn,
                    test_fn=test_fn,
                )
            )

            # klein beetje opruimen
            import gc
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    if results_long_rows:
        long_df = pd.DataFrame(results_long_rows).sort_values(["drug", "model"])
        long_csv = out_root / "results_long_virtual.csv"
        long_df.to_csv(long_csv, index=False)
        msg(f"Wrote per-drug results (long) → {long_csv}")

        wide_acc = long_df.pivot(
            index="drug", columns="model", values="val_best_acc"
        ).reset_index()
        wide_f1 = long_df.pivot(
            index="drug", columns="model", values="val_best_f1"
        ).reset_index()
        wide_acc.to_csv(
            out_root / "results_per_drug_val_acc_virtual.csv", index=False
        )
        wide_f1.to_csv(
            out_root / "results_per_drug_val_f1_virtual.csv", index=False
        )
        msg("Wrote per-drug VAL accuracy/F1 (virtual) CSVs.")


# ---------- MAIN / CLI ----------

def main():
    ap = argparse.ArgumentParser(
        description="cp_pipeline_v3 style per-drug training met VIRTUELE tiles (geen PNG-tiles op schijf) en well-level splits."
    )
    ap.add_argument(
        "--manifest_csv",
        required=True,
        help="CSV met minimaal 'broad_id' of 'drug' en 'image_path'/out_path/path/...",
    )
    ap.add_argument(
        "--out_root",
        required=True,
        help="Output-root (bijv. /scratch-shared/$USER/cellpainting/v3_virtualtiles_run_noleak)",
    )
    ap.add_argument("--models", default="resnetv2_101")
    ap.add_argument("--rows", type=int, default=3, help="Aantal tile-rijen per image.")
    ap.add_argument("--cols", type=int, default=4, help="Aantal tile-kolommen per image.")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max_drugs", type=int, default=None)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---------- MANIFEST LADEN + NORMALISEREN ----------
    msg(f"Reading manifest: {args.manifest_csv}")
    df = pd.read_csv(args.manifest_csv)

    # drug kolom
    if "broad_id" in df.columns:
        df["broad_id"] = df["broad_id"]
    elif "drug" in df.columns:
        df = df.rename(columns={"drug": "broad_id"})
    else:
        raise SystemExit(
            "Manifest must contain 'broad_id' or 'drug' column."
        )

    # path kolom
    if "image_path" in df.columns:
        df = df.rename(columns={"image_path": "src_image"})
    else:
        for cand in ["out_path", "path", "filepath", "file_path"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "src_image"})
                break

    if "src_image" not in df.columns:
        raise SystemExit(
            "Manifest must contain an image path column: "
            "image_path, out_path, path, filepath, or file_path."
        )

    df_images = df[["broad_id", "src_image"]].drop_duplicates().reset_index(drop=True)
    msg(f"Unique images in manifest (voor filter): {len(df_images)}")

    # ---- NIEUW: manifest opschonen → alleen bestaande PNG's ----
    df_images["exists"] = df_images["src_image"].apply(lambda p: os.path.exists(p))
    missing = (~df_images["exists"]).sum()
    if missing > 0:
        msg(f"[WARN] {missing} image paths do not exist on disk – these rows will be dropped.")
        df_images = df_images[df_images["exists"]].drop(columns="exists")
    else:
        df_images = df_images.drop(columns="exists", errors="ignore")

    msg(f"Unique images after filtering non-existing files: {len(df_images)}")

    # ---------- BALANCED VIRTUAL TILE-MANIFEST ----------
    msg("Building balanced virtual tile manifest (v3-style)…")
    full_tiles_df, per_drug_summary = build_balanced_virtual_manifest(
        df_images, rows=args.rows, cols=args.cols
    )
    msg(
        f"Balanced tile manifest built: {len(full_tiles_df)} tiles over "
        f"{len(per_drug_summary)} drugs."
    )

    tiles_csv = out_root / "train_manifest_balanced_virtual.csv"
    full_tiles_df.to_csv(tiles_csv, index=False)
    msg(f"Saved virtual tile manifest → {tiles_csv}")

    summary_csv = out_root / "per_drug_balance_summary_virtual.csv"
    per_drug_summary.to_csv(summary_csv, index=False)
    msg(f"Saved per-drug balance summary → {summary_csv}")

    # ---------- TRAIN PER-DRUG ----------
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    train_per_drug_virtual(
        full_df=full_tiles_df,
        out_root=out_root,
        models=models,
        img_size=args.img_size,
        rows=args.rows,
        cols=args.cols,
        batch=args.batch,
        lr=args.lr,
        epochs=args.epochs,
        max_drugs=args.max_drugs,
    )

    msg("DONE.")


if __name__ == "__main__":
    main()

