"""VGG-16 distances and similarity/stability tables for LA, PA and SA."""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import rankdata
import torch
import torchvision
from torchvision import models, transforms

PARTS = ("FL", "CL", "UG", "LG", "BD", "RT", "TL", "RRR", "RVM")
CARS = ("LA", "PA", "SA")


def calculate_tables(features, parts=PARTS, cars=CARS, weight=0.7):
    """Use unrounded float32 feature differences; sample SD; ascending ranks."""
    if not 0 <= weight <= 1:
        raise ValueError("weight must lie between zero and one")
    pairs = list(itertools.combinations(cars, 2))
    rows = []
    for part in parts:
        row = {"Feature Curve": part}
        distances = []
        for first, second in pairs:
            value = float(np.linalg.norm(features[(part, first)] - features[(part, second)]))
            row[f"{first}-{second}"] = value
            distances.append(value)
        values = np.asarray(distances, dtype=np.float64)
        mu, sigma = float(values.mean()), float(values.std(ddof=1))
        if mu <= 0 or not np.isfinite(values).all():
            raise ValueError(f"{part}: CV requires a finite positive mean distance")
        row.update(mu=mu, sigma=sigma, CV=sigma / mu)
        rows.append(row)
    for key, target in (("mu", "rank_mu"), ("CV", "rank_CV")):
        ranks = rankdata([row[key] for row in rows], method="average")
        for row, rank in zip(rows, ranks):
            row[target] = float(rank)
    for row in rows:
        row["S"] = weight * row["rank_mu"] + (1 - weight) * row["rank_CV"]
    return sorted(rows, key=lambda row: row["rank_mu"]), pairs


def write_table(path, rows, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(input_dir, output_dir, weight=0.7):
    source, output = Path(input_dir), Path(output_dir)
    expected = [source / f"{part}_{car}.png" for part in PARTS for car in CARS]
    missing = [path.name for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing component inputs: {missing}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "features").mkdir(exist_ok=True)
    torch.set_num_threads(4)
    weights = models.VGG16_Weights.IMAGENET1K_V1
    model = models.vgg16(weights=weights).features.eval()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    features, inputs = {}, []
    for path in expected:
        part, car = path.stem.split("_")
        with Image.open(path) as image:
            rgba = np.asarray(image.convert("RGBA"))
            # Preserve the supplied notebook: drop alpha, do not composite or recolor.
            tensor = preprocess(image.convert("RGB")).unsqueeze(0)
            size, mode = image.size, image.mode
        with torch.no_grad():
            feature_map = model(tensor)
        if tuple(feature_map.shape) != (1, 512, 7, 7):
            raise RuntimeError("Unexpected VGG feature-map shape")
        vector = feature_map.flatten().numpy()
        if not np.isfinite(vector).all():
            raise RuntimeError(f"Nonfinite features: {path.name}")
        features[(part, car)] = vector
        np.save(output / "features" / f"{path.stem}.npy", vector)
        inputs.append({"file": path.name, "size": size, "mode": mode,
                       "transparent_pixels": int((rgba[:, :, 3] < 255).sum()),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        print(path.name, flush=True)
    rows, pairs = calculate_tables(features, weight=weight)
    table2_columns = ["Feature Curve"] + [f"{a}-{b}" for a, b in pairs]
    table3_columns = ["Feature Curve", "mu", "sigma", "CV", "rank_mu", "rank_CV", "S"]
    write_table(output / "Table2.csv", rows, table2_columns)
    write_table(output / "Table3.csv", rows, table3_columns)
    lines = []
    for title, columns in (("Euclidean Distance Calculation Results", table2_columns),
                           ("Comprehensive Evaluation Indicators of Similarity and Stability", table3_columns)):
        lines += [f"## {title}", "", "| " + " | ".join(columns) + " |",
                  "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in rows:
            cells = [str(row[key]) if isinstance(row[key], str) else f"{row[key]:.4f}" for key in columns]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["SD uses ddof=1; CV=SD/mean; ranks ascend, with average ties.",
              f"S={weight}*rank_mu+{1-weight:.10g}*rank_CV. Rows are ordered by rank_mu.",
              "Scores are quantitative references. Final BGF/BDF labels require expert judgment."]
    (output / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    report = {
        "settings": {"weights": "IMAGENET1K_V1", "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "cars": CARS, "parts": PARTS,
                     "feature_layer": "features including final maxpool", "feature_shape": [512, 7, 7],
                     "dimension": 25088, "feature_L2_normalization": False,
                     "input": "RGB; resize224 bilinear; ImageNet normalization; no realignment",
                     "sample_sd_ddof": 1, "rank_ties": "average", "weight": weight},
        "inputs": inputs, "rows": rows,
        "ignored_pngs": sorted(path.name for path in source.glob("*.png") if path not in expected),
    }
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/euclidean")
    parser.add_argument("--weight", type=float, default=0.7)
    parser.add_argument("--torch-home", help="Optional existing Torch cache directory")
    args = parser.parse_args()
    if args.torch_home:
        os.environ["TORCH_HOME"] = str(Path(args.torch_home).resolve())
    run(args.input, args.output, args.weight)
