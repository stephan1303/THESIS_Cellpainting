#!/usr/bin/env python3
"""
disulfiram_vector_similarity_analysis.py

Supervisor-style analysis (vector-based, no arbitrary thresholds):
- Use a trained tile model (base vs MOCK/DMSO) and extract FEATURE vectors (embeddings).
- Build a prototype vector for the base drug (mean of base-image embeddings).
- Compute cosine similarity of every image embedding to the base prototype.
- Provide distributions for:
    * Base drug images (positive-like)
    * MOCK/DMSO images (negative-like)
    * Treated drug images (screening population)
- Provide ROC/AUC using base as positives and mock as negatives (threshold-free evaluation).

Outputs:
- image_similarity_scores_<base>.csv      (per image score + metadata)
- drug_similarity_summary_<base>.csv     (per drug summary stats of similarity)
- similarity_distributions_<base>.png
- roc_curve_<base>.png

Works with Python 3.9 and timm models (e.g., resnetv2_101).
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import timm
except Exception:
    timm = None


# ---------------------------
# Image utils
# ---------------------------

def imread_png(path: str) -> np.ndarray:
    import imageio.v3 as iio
    return iio.imread(path)

def crop_tile(img: np.ndarray, row: int, col: int, rows: int, cols: int) -> np.ndarray:
    H, W = img.shape[:2]
    th, tw = H // rows, W // cols
    y0 = row * th
    y1 = (row + 1) * th if row < rows - 1 else H
    x0 = col * tw
    x1 = (col + 1) * tw if col < cols - 1 else W
    return img[y0:y1, x0:x1]

def resize_to(img: np.ndarray, size: int) -> np.ndarray:
    try:
        import cv2
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image
        im = Image.fromarray(img)
        im = im.resize((size, size))
        return np.asarray(im)

def to_3ch(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


# ---------------------------
# Manifest helpers
# ---------------------------

def read_manifest(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"plate", "drug", "well", "role", "image_path"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit("[ERROR] Manifest mist kolommen: {}. Jij hebt: {}"
                         .format(missing, list(df.columns)))
    return df

def is_mock_row(role: str, drug: str, mock_roles: set, mock_drugs: set) -> bool:
    r = str(role).upper()
    d = str(drug).upper()
    return (r in mock_roles) or (d in mock_drugs)


# ---------------------------
# Dataset: Per IMAGE -> 12 tiles
# ---------------------------

class ScreenImagesDataset(Dataset):
    """
    Each item: one image -> tensor [T,3,H,W] where T=rows*cols + metadata.
    If missing file and skip_missing=True -> return None
    """
    def __init__(self, df_images: pd.DataFrame, img_size: int, rows: int, cols: int, skip_missing: bool = True):
        self.df = df_images.reset_index(drop=True)
        self.img_size = img_size
        self.rows = rows
        self.cols = cols
        self.skip_missing = skip_missing

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        path = str(r["image_path"])
        if self.skip_missing and (not os.path.exists(path)):
            return None

        img = imread_png(path)
        tiles = []
        for rr in range(self.rows):
            for cc in range(self.cols):
                t = crop_tile(img, rr, cc, self.rows, self.cols)
                t = resize_to(t, self.img_size)
                t = to_3ch(t)
                x = torch.from_numpy(t.transpose(2, 0, 1)).float() / 255.0
                tiles.append(x)

        tiles = torch.stack(tiles, 0)  # [T,3,H,W]

        plate = str(r["plate"])
        well = str(r["well"])
        drug = str(r["drug"])
        role = str(r["role"])
        image_id = "{}__{}__{}".format(plate, well, Path(path).stem)
        well_id = "{}__{}".format(plate, well)

        meta = {
            "drug": drug,
            "plate": plate,
            "well": well,
            "well_id": well_id,
            "role": role,
            "image_id": image_id,
            "image_path": path,
        }
        return tiles, meta

def collate_screen_drop_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    tiles, metas = zip(*batch)
    tiles = torch.stack(tiles, 0)  # [B,T,3,H,W]
    out_meta = {k: [m[k] for m in metas] for k in metas[0].keys()}
    return tiles, out_meta


# ---------------------------
# Model + feature extractor
# ---------------------------

def build_model(model_name: str, num_classes: int = 2) -> nn.Module:
    if timm is None:
        raise SystemExit("[ERROR] timm is not installed in this env.")
    return timm.create_model(model_name, pretrained=True, num_classes=num_classes)

class TimmFeatureWrapper(nn.Module):
    """
    Wrap a timm classification model to output a feature vector (embedding).
    Uses forward_features + pooling if available.
    Falls back to removing classifier head if needed.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # Remove classifier head where possible (safe for many timm models)
        # We still keep forward_features path if available.
        if hasattr(self.model, "reset_classifier"):
            try:
                self.model.reset_classifier(0)  # num_classes=0 => output features in many timm models
            except Exception:
                pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Try timm standard: forward_features + pool/head
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)

            # feats can be [B,C,H,W] or [B,N,C] or [B,C]
            if feats.ndim == 4:
                feats = feats.mean(dim=(2, 3))
            elif feats.ndim == 3:
                feats = feats.mean(dim=1)

            return feats

        # Fallback: use model(x) directly (if classifier reset worked, this may be features)
        out = self.model(x)
        if out.ndim > 2:
            out = out.view(out.size(0), -1)
        return out


def find_checkpoint(out_dir: str, base_drug: str) -> str:
    """
    Find base_<base_drug>_best.pt under out_dir (or out_dir itself).
    """
    out_dir = os.path.expanduser(os.path.expandvars(out_dir))
    p = Path(out_dir)
    if not p.exists():
        raise FileNotFoundError("out_dir does not exist: {}".format(out_dir))

    cand = p / "base_{}_best.pt".format(base_drug)
    if cand.exists():
        return str(cand)

    # search
    hits = list(p.rglob("base_{}_best.pt".format(base_drug)))
    if hits:
        hits.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(hits[0])

    raise FileNotFoundError(
        "Could not find checkpoint base_{}_best.pt under {}".format(base_drug, out_dir)
    )


@torch.no_grad()
def compute_image_embeddings(df_images: pd.DataFrame,
                             feature_model: nn.Module,
                             device: str,
                             img_size: int,
                             rows: int,
                             cols: int,
                             batch_images: int,
                             num_workers: int,
                             skip_missing: bool,
                             log_every: int) -> pd.DataFrame:
    """
    Returns dataframe with one row per image:
    drug, role, plate, well, image_id, image_path, embedding (as np array), + cosine later.
    """
    ds = ScreenImagesDataset(df_images, img_size, rows, cols, skip_missing=skip_missing)
    ld = DataLoader(ds, batch_size=batch_images, shuffle=False, num_workers=num_workers,
                    pin_memory=True, collate_fn=collate_screen_drop_none)

    tiles_per_image = rows * cols

    rows_out = []
    processed = 0

    feature_model.eval()

    for step, batch in enumerate(ld, start=1):
        if batch is None:
            continue
        tiles, meta = batch  # [B,T,3,H,W]
        B = tiles.size(0)

        tiles_f = tiles.view(B * tiles_per_image, 3, img_size, img_size).to(device, non_blocking=True)
        feats = feature_model(tiles_f)  # [B*T, D]
        feats = feats.detach().cpu().numpy().reshape(B, tiles_per_image, -1)

        # image embedding = mean over tiles
        img_emb = feats.mean(axis=1)  # [B, D]

        for i in range(B):
            rows_out.append({
                "drug": meta["drug"][i],
                "role": meta["role"][i],
                "plate": meta["plate"][i],
                "well": meta["well"][i],
                "well_id": meta["well_id"][i],
                "image_id": meta["image_id"][i],
                "image_path": meta["image_path"][i],
                "embedding": img_emb[i],
            })

        processed += B
        if log_every and (step % log_every == 0):
            print("[EMB] step={} images_done={:,}".format(step, processed))

    return pd.DataFrame(rows_out)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def roc_auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve and AUC without sklearn (pure numpy).
    Returns fpr, tpr, auc.
    """
    y = np.concatenate([np.ones_like(scores_pos), np.zeros_like(scores_neg)])
    s = np.concatenate([scores_pos, scores_neg])

    # sort by descending score
    order = np.argsort(-s)
    y = y[order]
    s = s[order]

    P = float((y == 1).sum())
    N = float((y == 0).sum())
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    tps = 0.0
    fps = 0.0
    tpr = [0.0]
    fpr = [0.0]

    last_score = None
    for i in range(len(s)):
        if last_score is None:
            last_score = s[i]
        # update counts
        if y[i] == 1:
            tps += 1.0
        else:
            fps += 1.0

        # add point when score changes (stepwise ROC)
        if i == len(s) - 1 or s[i + 1] != s[i]:
            tpr.append(tps / P)
            fpr.append(fps / N)

    fpr = np.array(fpr, dtype=float)
    tpr = np.array(tpr, dtype=float)

    # AUC via trapezoid rule over FPR
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--base_drug", default="BRD-K32744045")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=224)

    ap.add_argument("--model_name", type=str, default="resnetv2_101")

    ap.add_argument("--checkpoint", default=None,
                    help="Path to base_<base_drug>_best.pt. If omitted, will search in --out_dir.")
    ap.add_argument("--mock_roles", type=str, default="MOCK,DMSO")
    ap.add_argument("--mock_drugs", type=str, default="MOCK,DMSO")

    ap.add_argument("--batch_images", type=int, default=8, help="Images per batch for embedding extraction.")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--skip_missing", action="store_true")
    ap.add_argument("--log_every", type=int, default=50)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    df = read_manifest(args.manifest_csv)
    df["drug"] = df["drug"].astype(str)
    df["role"] = df["role"].astype(str)

    mock_roles = set([x.strip().upper() for x in args.mock_roles.split(",") if x.strip()])
    mock_drugs = set([x.strip().upper() for x in args.mock_drugs.split(",") if x.strip()])

    base_drug = str(args.base_drug)

    # Split into base, mock, treated
    is_mock = df.apply(lambda r: is_mock_row(r["role"], r["drug"], mock_roles, mock_drugs), axis=1)
    df_mock = df[is_mock].drop_duplicates(subset=["image_path"]).copy()
    df_base = df[df["drug"] == base_drug].drop_duplicates(subset=["image_path"]).copy()
    df_treated = df[~is_mock].drop_duplicates(subset=["image_path"]).copy()

    if df_base.empty:
        raise SystemExit("[ERROR] No images found for base_drug={}".format(base_drug))
    if df_mock.empty:
        raise SystemExit("[ERROR] No MOCK/DMSO images found. Check mock_roles/mock_drugs settings.")

    # checkpoint
    if args.checkpoint is None:
        ckpt_path = find_checkpoint(str(out_dir), base_drug)
    else:
        ckpt_path = os.path.expanduser(os.path.expandvars(args.checkpoint))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError("checkpoint not found: {}".format(ckpt_path))

    print("[INFO] base_drug   : {}".format(base_drug))
    print("[INFO] checkpoint : {}".format(ckpt_path))
    print("[INFO] base_images: {:,}".format(len(df_base)))
    print("[INFO] mock_images: {:,}".format(len(df_mock)))
    print("[INFO] treated_images (incl base): {:,}".format(len(df_treated)))

    # Build model and load checkpoint weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = build_model(args.model_name, num_classes=2)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    clf.load_state_dict(ckpt["model"], strict=True)
    clf.to(device)
    clf.eval()

    feat_model = TimmFeatureWrapper(clf).to(device).eval()

    # Compute embeddings for base, mock, and treated
    print("[STEP] Extract embeddings: base")
    emb_base = compute_image_embeddings(df_base, feat_model, device,
                                        args.img_size, args.rows, args.cols,
                                        args.batch_images, args.num_workers,
                                        args.skip_missing, args.log_every)

    print("[STEP] Extract embeddings: mock")
    emb_mock = compute_image_embeddings(df_mock, feat_model, device,
                                        args.img_size, args.rows, args.cols,
                                        args.batch_images, args.num_workers,
                                        args.skip_missing, args.log_every)

    print("[STEP] Extract embeddings: treated (this can be big)")
    emb_treated = compute_image_embeddings(df_treated, feat_model, device,
                                           args.img_size, args.rows, args.cols,
                                           args.batch_images, args.num_workers,
                                           args.skip_missing, args.log_every)

    # Prototype vector for base = mean base embeddings
    base_proto = np.vstack(emb_base["embedding"].values).mean(axis=0)

    # Compute cosine similarity for each table
    def add_scores(df_emb: pd.DataFrame, proto: np.ndarray) -> pd.DataFrame:
        sims = []
        for v in df_emb["embedding"].values:
            sims.append(cosine_sim(v, proto))
        out = df_emb.drop(columns=["embedding"]).copy()
        out["cosine_to_base"] = np.array(sims, dtype=float)
        return out

    scores_base = add_scores(emb_base, base_proto)
    scores_mock = add_scores(emb_mock, base_proto)
    scores_all = add_scores(emb_treated, base_proto)

    # Save per-image scores
    out_img_csv = out_dir / "image_similarity_scores_{}.csv".format(base_drug)
    scores_all.to_csv(out_img_csv, index=False)
    print("[SAVED] {}".format(out_img_csv))

    # Per-drug summary (no threshold): mean/median/max + n_images
    summ = scores_all.groupby("drug").agg(
        n_images=("image_id", "count"),
        mean_sim=("cosine_to_base", "mean"),
        median_sim=("cosine_to_base", "median"),
        max_sim=("cosine_to_base", "max"),
        p95_sim=("cosine_to_base", lambda x: float(np.quantile(x, 0.95))),
    ).reset_index().sort_values(["mean_sim", "max_sim"], ascending=False)

    out_drug_csv = out_dir / "drug_similarity_summary_{}.csv".format(base_drug)
    summ.to_csv(out_drug_csv, index=False)
    print("[SAVED] {}".format(out_drug_csv))

    # Plot distributions
    fig1 = plt.figure(figsize=(9.5, 5.2))
    plt.hist(scores_mock["cosine_to_base"].values, bins=50, alpha=0.6, label="MOCK/DMSO (negative)")
    plt.hist(scores_base["cosine_to_base"].values, bins=50, alpha=0.6, label="Base drug (positive)")
    plt.title("Cosine similarity to base prototype ({}): distributions".format(base_drug), fontsize=13)
    plt.xlabel("Cosine similarity to base vector", fontsize=12)
    plt.ylabel("Number of images", fontsize=12)
    plt.legend()
    plt.tight_layout()
    out_dist = out_dir / "similarity_distributions_{}.png".format(base_drug)
    plt.savefig(out_dist, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print("[SAVED] {}".format(out_dist))

    # ROC/AUC (threshold-free summary of separability base vs mock)
    fpr, tpr, auc = roc_auc(scores_base["cosine_to_base"].values, scores_mock["cosine_to_base"].values)

    fig2 = plt.figure(figsize=(6.2, 6.0))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.title("ROC: base vs mock (AUC = {:.3f})".format(auc), fontsize=13)
    plt.xlabel("False Positive Rate (mock)", fontsize=12)
    plt.ylabel("True Positive Rate (base)", fontsize=12)
    plt.tight_layout()
    out_roc = out_dir / "roc_curve_{}.png".format(base_drug)
    plt.savefig(out_roc, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print("[SAVED] {}".format(out_roc))

    # Print top 20 drugs (by mean similarity) excluding the base drug itself
    top20 = summ[summ["drug"] != base_drug].head(20)
    print("\n[TOP-20] Drugs by mean cosine similarity to {} (no threshold):".format(base_drug))
    print(top20[["drug", "n_images", "mean_sim", "median_sim", "max_sim", "p95_sim"]].to_string(index=False))


if __name__ == "__main__":
    main()
