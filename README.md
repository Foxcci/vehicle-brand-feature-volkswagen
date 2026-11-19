# Vehicle brand feature analysis for Volkswagen

Code for VGG-16/Grad-CAM visualization, VGG-16 Euclidean-distance calculation,
and standard GM(1,1) curve extrapolation in the Volkswagen case.
Release: **0.1.0**. The repository contains portable scripts and notebook entry
points. It does not perform expert annotation or replace expert BGF/BDF decisions.

## Modules

| Script | Input | Output |
| --- | --- | --- |
| `vgg_gradcam.py` | Six prepared vehicle PNGs | Individual overlays, heatmaps, foreground masks, raw activations, target-class log |
| `euclidean_distance.py` | 27 component PNGs: nine parts × LA/PA/SA | 25,088-dimensional vectors, pairwise distances, similarity/stability tables |
| `gm11_forecast.py` | Five historical contour PNGs per part: LG/CL/TL | Raw 100-point coordinates, next-stage predictions, point plots and display curves |

Manual stages connect these modules: experts interpret saliency maps, delineate
feature contours with Bézier curves, classify BGFs/BDFs, and select BDFs for
historical analysis. The case uses an expert panel of 15 people. There is no
automatic numerical threshold that assigns the final BGF/BDF labels here.

## Installation

Use Python **3.12** in a virtual environment. The pinned versions reflect the
verified local environment, not an asserted minimum compatible version.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux/macOS: use `python3.12 -m venv .venv` and `.venv/bin/python` in place of the
Windows interpreter path. CPU is the verified execution path. The official
ImageNet V1 VGG-16 weights are downloaded by torchvision on the first run unless
already cached. They are not bundled in this repository. `--torch-home PATH`
selects an existing Torch cache.

## Prepare the inputs

### Vehicle images

Place these files in `data/vehicles/`:

```text
I1_Lavida_front.png
I2_Lavida_back.png
I3_Passat_front.png
I4_Passat_back.png
I5_Sagitar_front.png
I6_Sagitar_back.png
```

Use prepared front/rear PNGs with a uniform gray background, normally RGB
128/128/128. The manuscript's plate/window preparation is manual; the script
does not locate or mask those regions automatically. Vehicle logos are retained.
The verified images are 448×448. Non-square inputs are padded with preserved
aspect ratio, not cropped. The display image is 448×448; the network image is
224×224. Grayscale is replicated into RGB and normalized with ImageNet mean/SD.

### Component images

Place `PART_MODEL.png` files in `data/components/`.
Models are `LA`, `PA`, `SA`; parts are `FL`, `CL`, `UG`, `LG`, `BD`, `RT`, `TL`,
`RRR`, `RVM`. For example: `CL_LA.png`, `CL_PA.png`, `CL_SA.png`.
Exactly 27 required combinations are processed. Extra PNGs, including abandoned
BO/MA samples, are ignored and listed in the output report.

The verified canvases are 340×340, with opaque white backgrounds and the original
part-specific color fills. Preserve the supplied canvas, placement and colors
when reproducing numerical outputs. The script uses RGB and does not recolor,
register, crop or normalize vectors to unit length. Alpha is dropped as in the
source notebook; use opaque inputs or understand the underlying RGB values.
These distances describe deep image features, not pure geometric contour errors.

### Historical SA contours

Place these fifteen files in `data/sa_history/`:

```text
LG_SA_2015.png  LG_SA_2018.png  LG_SA_2020.png  LG_SA_2023.png  LG_SA_2026.png
CL_SA_2015.png  CL_SA_2018.png  CL_SA_2020.png  CL_SA_2023.png  CL_SA_2026.png
TL_SA_2015.png  TL_SA_2018.png  TL_SA_2020.png  TL_SA_2023.png  TL_SA_2026.png
```

Use a red closed outline or red filled region on an opaque white canvas, following
the 11_24 notebook's red extraction convention. A single component must use
consistent canvas size, units, side/half-versus-full representation, and contour
definition across all five stages. This script does not mirror, resize or register
the curves. Display colors of final paper figures are not suitable inputs if
they fail the documented HSV thresholds.

For the original one-component names `curve1.png` ... `curve5.png`, use
`--legacy-part CL` (or LG/TL). Their order must correspond to the five stages.

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe vgg_gradcam.py --input data/vehicles --output outputs/gradcam
.\.venv\Scripts\python.exe euclidean_distance.py --input data/components --output outputs/euclidean
.\.venv\Scripts\python.exe gm11_forecast.py --input data/sa_history --output outputs/gm11
```

The three notebooks in `notebooks/` call these same scripts. They contain no
embedded local results or original photographs. On Colab upload/extract the
whole repository, install dependencies and set the working directory first.

## Exact computational settings

**Grad-CAM.** `VGG16_Weights.IMAGENET1K_V1`, eval mode, no training. Hook
`features[16]` and `features[23]`: **pool3 and pool4**, not conv3_3/conv4_3.
The target is the highest-logit ImageNet class for each image. Channel weights
are spatially averaged gradients. Positive CAMs are bilinearly upsampled,
scaled by each foreground maximum, fused with weights 0.55/0.45, and scaled
again within the foreground. A border-connected color mask (tolerance 3) limits
the visualization. Display mapping is CAM^0.65, with overlay alpha
0.35+0.45×display_CAM. The background pixels are preserved. Masking and gamma
do not change inference; ImageNet targets need not be vehicle/brand labels.
Check `targets_and_checks.csv` and mask outputs rather than treating the colors
as validated part segmentation or comparable absolute scores across images.

**Distances.** Final VGG feature output: 512×7×7, flattened to 25,088 values.
No L2 normalization. The three distances per part are LA–PA, LA–SA and PA–SA.
SD uses n−1 (ddof=1); CV=SD/mean. Ascending ranks use averaged ties.
S=0.7×rank(mean)+0.3×rank(CV). Both tables are ordered by rank(mean).
These shared-model pairwise distances are not independent experimental units.

**GM(1,1).** The default `--solver stable` uses least-squares fitting and an
expm1/near-zero response with constant-series handling. This numerical variant
preserves the standard GM model; it is not residual correction. The explicit
`--solver legacy` option retains the 11_24 normal equations and response-difference
formula for comparison. That original formula can fail or produce spikes for
constant/near-constant coordinates. Both preserve native OpenCV contour order. The optional
residual-corrected experiment in the original notebook is not used by this
standard command. Native contour order and equal arc-length sampling alone do
not guarantee semantic point correspondence across styling stages.

Stages are indexed in sequence; calendar-year gaps are not modeled. The
default 100 samples exclude the duplicated closure point. A forecast from the
five stages is a next-stage extrapolation, not held-out validation of 2026.

**Display smoothing.** `--smoothing periodic` retains the notebook's periodic
cubic interpolation; `none` draws the point polyline. The explicit `light`
option uses an arc-length-parameterized periodic smoothing spline with
s=N×0.8² (adjustable by `--light-rms`). These options never modify saved raw
history/prediction coordinates. Smoothing is display processing and cannot
establish prediction accuracy or guarantee the absence of geometric protrusions.

## Verification and available data

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe make_demo_inputs.py --output outputs/demo_inputs
.\.venv\Scripts\python.exe gm11_forecast.py --input outputs/demo_inputs/sa_history --output outputs/demo_gm11 --solver stable
```

Synthetic geometry is for software checks only. Located real-data checks and
original-notebook regression checks are documented in `docs/verification.md`.
The original website photographs and research contour inputs are not included.
The full Volkswagen historical input set was not located during packaging, so
the historical paper plots are not claimed to be reproduced by this release.
Source hashes and packaging changes are listed in `docs/provenance.json`.

For the Chinese input and upload instructions, see `README_中文.md` and
`docs/GITHUB_UPLOAD_中文.md`. No reuse license has been selected on behalf of
the authors; decide the code license separately before granting reuse rights.
