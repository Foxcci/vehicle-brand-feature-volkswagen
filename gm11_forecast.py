"""11_24 standard GM(1,1): five observed SA styling stages -> next stage."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, splprep, splev

YEARS = (2015, 2018, 2020, 2023, 2026)
PARTS = ("LG", "CL", "TL")


def extract_red_contour(path):
    """Original 11_24 thresholds, closing and largest-perimeter external contour."""
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(str(path))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255])),
                         cv2.inRange(hsv, np.array([160, 80, 80]), np.array([180, 255, 255])))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError(f"No red contour in {Path(path).name}")
    contour = max(contours, key=lambda item: cv2.arcLength(item, True))[:, 0, :].astype(np.float64)
    if len(contour) < 3:
        raise ValueError("Contour has fewer than three points")
    return contour, mask


def resample_contour(points, num_points=100):
    closed = np.vstack([points, points[0]])
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    if lengths[-1] <= 0 or num_points < 4:
        raise ValueError("Degenerate contour or invalid point count")
    positions = np.linspace(0, lengths[-1], num_points, endpoint=False)
    return np.column_stack([np.interp(positions, lengths, closed[:, axis]) for axis in (0, 1)])


def gm11_predict_next(values, solver="stable"):
    """legacy preserves the notebook formula; stable evaluates the same model safely."""
    values = np.asarray(values, dtype=np.float64).flatten()
    if len(values) < 4 or not np.isfinite(values).all():
        raise ValueError("At least four finite observations are required")
    accumulated = np.cumsum(values)
    background = 0.5 * (accumulated[1:] + accumulated[:-1])
    design = np.column_stack([-background, np.ones(len(values)-1)])
    if solver == "legacy":
        normal = design.T @ design
        if np.linalg.det(normal) == 0:
            return float(values[-1])  # Original notebook's singular-fit fallback.
        a, b = (np.linalg.inv(normal) @ design.T @ values[1:].reshape(-1, 1)).ravel()
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            def response(k):
                return (values[0] - b/a) * np.exp(-a*(k-1)) + b/a
            prediction = response(len(values)+1) - response(len(values))
    elif solver == "stable":
        if np.ptp(values) < 1e-10:
            return float(values[-1])
        fitted, _, rank, _ = np.linalg.lstsq(design, values[1:], rcond=None)
        if rank < 2:
            raise ValueError("Rank-deficient GM fit")
        a, b = fitted
        factor = 1-a/2+a*a/6 if abs(a) < 1e-8 else -np.expm1(-a)/a
        with np.errstate(over="ignore", invalid="ignore"):
            prediction = (b-a*values[0]) * np.exp(-a*(len(values)-1)) * factor
    else:
        raise ValueError("Unknown solver")
    if not np.isfinite(prediction):
        raise FloatingPointError("Nonfinite GM prediction; audit inputs or explicitly use --solver stable")
    return float(prediction)


def display_curve(points, method="periodic", rms=0.8):
    """Return display coordinates only; never replace saved prediction coordinates."""
    closed = np.vstack([points, points[0]])
    if method == "none":
        return closed
    if method == "periodic":
        # Same index-parameterized periodic interpolation as cells 8/9 of 11_24.
        parameter = np.arange(len(closed))
        dense_parameter = np.linspace(0, len(points), 500, endpoint=True)
        return np.column_stack([CubicSpline(parameter, closed[:, axis], bc_type="periodic")(dense_parameter)
                                for axis in (0, 1)])
    increments = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    if np.any(increments <= 1e-10):
        raise ValueError("Light smoothing requires distinct consecutive contour points")
    parameter = np.r_[0., np.cumsum(increments)]
    parameter /= parameter[-1]
    spline, _ = splprep(closed.T, u=parameter, s=len(points)*rms*rms, per=True, k=3)
    return np.column_stack(splev(np.linspace(0, 1, 1001), spline))


def save_points(path, points):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["point", "x_pixels", "y_pixels"])
        writer.writerows((i+1, float(x), float(y)) for i, (x, y) in enumerate(points))


def run(input_dir, output_dir, solver="stable", smoothing="periodic", points=100,
        light_rms=0.8, legacy_part=None):
    source, output = Path(input_dir), Path(output_dir)
    selected = [legacy_part] if legacy_part else PARTS
    expected = {part: [source / (f"curve{i+1}.png" if legacy_part else f"{part}_SA_{year}.png")
                       for i, year in enumerate(YEARS)] for part in selected}
    missing = [path.name for paths in expected.values() for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing historical inputs: {missing}")
    for folder in ("coordinates", "masks", "point_plots", "display_curves", "comparisons"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
    inputs, reports = [], []
    for part, paths in expected.items():
        history, sizes = [], []
        for year, path in zip(YEARS, paths):
            contour, mask = extract_red_contour(path)
            sizes.append(mask.shape)
            sampled = resample_contour(contour, points)
            history.append(sampled)
            cv2.imencode(".png", mask)[1].tofile(output / "masks" / f"{part}_SA_{year}.png")
            inputs.append({"file": path.name, "part": part, "year_label": year,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        if len(set(sizes)) != 1:
            raise ValueError(f"{part}: use the same source canvas size for all stages")
        history = np.stack(history)
        prediction = np.array([[gm11_predict_next(history[:, j, axis], solver) for axis in (0, 1)]
                               for j in range(points)])
        np.save(output / "coordinates" / f"{part}_history.npy", history)
        np.save(output / "coordinates" / f"{part}_prediction.npy", prediction)
        series = [*history, prediction]
        labels = [*map(str, YEARS), "next_prediction"]
        displayed = [display_curve(item, smoothing, light_rms) for item in series]
        fig, ax = plt.subplots(figsize=(6, 6))
        for label, item in zip(labels[:-1], history):
            closed = np.vstack([item, item[0]])
            ax.plot(closed[:, 0], closed[:, 1], "--", lw=1, label=label)
        closed = np.vstack([prediction, prediction[0]])
        ax.plot(closed[:, 0], closed[:, 1], lw=2, label="Next stage")
        decorate(ax, f"{part}: historical contours and next-stage prediction", grid=False)
        ax.legend(fontsize=8)
        save_figure(fig, output / "comparisons" / part)
        for label, item, dense in zip(labels, series, displayed):
            stem = f"{part}_SA_{label}"
            save_points(output / "coordinates" / f"{stem}.csv", item)
            fig, ax = plt.subplots(figsize=(5, 5))
            # Draw the closing line separately so exactly N equally sized markers appear.
            closed = np.vstack([item, item[0]])
            ax.plot(closed[:, 0], closed[:, 1], lw=2)
            ax.plot(item[:, 0], item[:, 1], linestyle="none", marker="o", markersize=6,
                    markeredgewidth=0, color="#1f77b4")
            decorate(ax, f"{part}: {label}")
            save_figure(fig, output / "point_plots" / stem)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(dense[:, 0], dense[:, 1], lw=2)
            decorate(ax, f"{part}: {label} ({smoothing} display)")
            save_figure(fig, output / "display_curves" / stem)
            save_points(output / "display_curves" / f"{stem}_DISPLAY_ONLY.csv", dense)
        reports.append({"part": part, "canvas_height_width": sizes[0],
                        "historical_shape": list(history.shape),
                        "prediction_shape": list(prediction.shape),
                        "prediction_min": prediction.min(0).tolist(),
                        "prediction_max": prediction.max(0).tolist()})
        print(f"{part}: five observed stages -> next stage ({solver})", flush=True)
    report = {"solver": solver, "residual_correction": False, "years": YEARS,
              "num_points": points, "point_order": "native OpenCV; no phase/direction alignment",
              "time_axis": "styling-stage index, not calendar-year intervals",
              "smoothing": smoothing, "light_rms": light_rms if smoothing == "light" else None,
              "forecast_coordinates_modified_by_smoothing": False,
              "inputs": inputs, "parts": reports,
              "status": "extrapolation; no held-out generation evaluated by this command"}
    (output / "settings.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def decorate(ax, title, grid=True):
    ax.set_title(title)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.grid(grid, alpha=.5) if grid else ax.grid(False)


def save_figure(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/gm11")
    parser.add_argument("--solver", choices=("legacy", "stable"), default="stable")
    parser.add_argument("--smoothing", choices=("none", "periodic", "light"), default="periodic")
    parser.add_argument("--points", type=int, default=100)
    parser.add_argument("--light-rms", type=float, default=.8)
    parser.add_argument("--legacy-part", choices=PARTS,
                        help="Read curve1.png...curve5.png for this single component")
    args = parser.parse_args()
    if args.light_rms < 0:
        parser.error("--light-rms must be nonnegative")
    run(args.input, args.output, args.solver, args.smoothing, args.points, args.light_rms, args.legacy_part)
