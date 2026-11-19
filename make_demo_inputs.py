"""Generate artificial geometry for software checks, not paper experiments."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw


def run(destination):
    root = Path(destination)
    for folder in ("vehicles", "components", "sa_history"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    names = [("Lavida", "LA"), ("Passat", "PA"), ("Sagitar", "SA")]
    parts = ["FL", "CL", "UG", "LG", "BD", "RT", "TL", "RRR", "RVM"]
    for j, (name, car) in enumerate(names):
        for k, view in enumerate(("front", "back")):
            image = Image.new("RGB", (448, 448), (128, 128, 128))
            drawing = ImageDraw.Draw(image)
            drawing.rounded_rectangle((70, 130+j*3, 378, 350), radius=25, fill="white", outline="black", width=3)
            drawing.polygon([(110, 130+j*3), (145, 70), (303, 70), (338, 130+j*3)], fill="white")
            drawing.rectangle((145, 85, 303, 140), fill=(165, 165, 165), outline="black")
            drawing.rectangle((90, 350, 135, 385), fill="black")
            drawing.rectangle((313, 350, 358, 385), fill="black")
            drawing.rectangle((95, 195+k*5, 170, 210+k*5), fill=(235, 50, 50) if k else (50, 50, 50))
            drawing.rectangle((278, 195+k*5, 353, 210+k*5), fill=(235, 50, 50) if k else (50, 50, 50))
            drawing.rectangle((130, 260, 318, 295), fill=(45, 45, 45))
            image.save(root / "vehicles" / f"I{j*2+k+1}_{name}_{view}.png")
        for index, part in enumerate(parts):
            image = Image.new("RGB", (340, 340), "white")
            drawing = ImageDraw.Draw(image)
            drawing.polygon([(55, 110+j*5), (270-index*2, 125), (285, 190+j*3), (65+index, 210)],
                            fill=(225-index*8, 190+index*3, 160+index*5), outline="black")
            image.save(root / "components" / f"{part}_{car}.png")
    for index, part in enumerate(("LG", "CL", "TL")):
        for stage, year in enumerate((2015, 2018, 2020, 2023, 2026)):
            image = np.full((340, 340, 3), 255, dtype=np.uint8)
            # The ellipse's width changes, with fixed origin and traversal behavior.
            cv2.ellipse(image, (170, 170), (100+stage*3, 35+index*13), 0, 0, 360, (0, 0, 255), -1)
            cv2.imencode(".png", image)[1].tofile(root / "sa_history" / f"{part}_SA_{year}.png")
    (root / "SYNTHETIC_ONLY.txt").write_text(
        "Artificial software-check inputs. No research samples, model accuracy claims or paper results.\n", encoding="utf-8")
    print("Created synthetic test inputs:", root.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/demo_inputs")
    run(parser.parse_args().output)
