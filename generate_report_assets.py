from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import diabetes_project as project
from sklearn.metrics import roc_curve


RESULTS_DIR = Path("results")
ASSETS_DIR = RESULTS_DIR / "report_assets"

WIDTH = 1200
HEIGHT = 760
BG = "#FAFAF7"
TEXT = "#1F2937"
GRID = "#D1D5DB"
BLUE = "#4E79A7"
RED = "#E15759"
TEAL = "#76B7B2"
GOLD = "#EDC948"


def font(size: int, bold: bool = False):
    try:
        path = "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"
        return ImageFont.truetype(path, size=size)
    except Exception:
        return ImageFont.load_default()


def base_canvas(title: str, subtitle: str | None = None) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((50, 35), title, fill=TEXT, font=font(32, bold=True))
    if subtitle:
        draw.text((50, 78), subtitle, fill="#475569", font=font(18))
    return image, draw


def draw_axes(draw: ImageDraw.ImageDraw, left: int, top: int, right: int, bottom: int, y_ticks: list[float], y_labels: list[str]) -> None:
    draw.line((left, top, left, bottom), fill=TEXT, width=3)
    draw.line((left, bottom, right, bottom), fill=TEXT, width=3)
    for tick, label in zip(y_ticks, y_labels):
        y = int(bottom - tick * (bottom - top))
        draw.line((left - 10, y, right, y), fill=GRID, width=1)
        draw.text((left - 55, y - 10), label, fill=TEXT, font=font(16))


def save_class_distribution(raw_df) -> None:
    counts = raw_df["Outcome"].value_counts().sort_index()
    image, draw = base_canvas("Sınıf Dağılımı", "Veri setindeki negatif ve pozitif gözlem sayıları")
    left, top, right, bottom = 120, 180, 1040, 620
    draw_axes(draw, left, top, right, bottom, [0, 0.25, 0.5, 0.75, 1.0], ["0", "125", "250", "375", "500"])

    labels = ["Negatif (0)", "Pozitif (1)"]
    colors = [BLUE, RED]
    max_count = max(counts.values)
    bar_width = 190
    xs = [300, 650]
    for x, label, value, color in zip(xs, labels, counts.values, colors):
        bar_height = int((value / max_count) * (bottom - top))
        draw.rounded_rectangle((x, bottom - bar_height, x + bar_width, bottom), radius=16, fill=color)
        draw.text((x + 40, bottom + 18), label, fill=TEXT, font=font(18))
        draw.text((x + 70, bottom - bar_height - 34), str(int(value)), fill=TEXT, font=font(20, bold=True))

    image.save(ASSETS_DIR / "class_distribution.png")


def save_metric_comparison(comparison_df) -> None:
    image, draw = base_canvas("Algoritma Performans Karşılaştırması", "Accuracy, F1-Score ve ROC-AUC değerleri")
    left, top, right, bottom = 110, 180, 1120, 620
    draw_axes(draw, left, top, right, bottom, [0.5, 0.6, 0.7, 0.8, 0.9], ["0.50", "0.60", "0.70", "0.80", "0.90"])

    models = comparison_df["model"].tolist()
    group_w = 90
    gap = 12
    start_x = 125
    scale = bottom - top
    for idx, row in comparison_df.iterrows():
        base_x = start_x + idx * 100
        vals = [row["accuracy"], row["f1_score"], row["roc_auc"]]
        cols = [BLUE, RED, TEAL]
        for j, (val, col) in enumerate(zip(vals, cols)):
            x1 = base_x + j * 24
            x2 = x1 + 18
            y1 = int(bottom - ((val - 0.5) / 0.4) * scale)
            draw.rounded_rectangle((x1, y1, x2, bottom), radius=4, fill=col)
        label = models[idx].replace("Regression", "Reg.").replace("Forest", "For.")
        draw.text((base_x - 6, bottom + 18), label[:10], fill=TEXT, font=font(12))

    legend_y = 120
    for x, color, label in [(720, BLUE, "Accuracy"), (860, RED, "F1-Score"), (995, TEAL, "ROC-AUC")]:
        draw.rounded_rectangle((x, legend_y, x + 22, legend_y + 22), radius=5, fill=color)
        draw.text((x + 32, legend_y - 1), label, fill=TEXT, font=font(16))

    image.save(ASSETS_DIR / "metric_comparison.png")


def best_model_payload():
    df = project.load_and_prepare_data()
    x = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    x_train_full, x_test, y_train_full, y_test = project.train_test_split(
        x, y, test_size=0.20, stratify=y, random_state=project.RANDOM_STATE
    )
    x_train, x_val, y_train, y_val = project.train_test_split(
        x_train_full, y_train_full, test_size=0.25, stratify=y_train_full, random_state=project.RANDOM_STATE
    )

    best = None
    best_name = None
    for model_name, pipeline in project.get_models(list(x.columns)).items():
        pipeline.fit(x_train, y_train)
        val_probabilities = pipeline.predict_proba(x_val)[:, 1]
        threshold = project.find_best_threshold(y_val, val_probabilities)
        test_probabilities = pipeline.predict_proba(x_test)[:, 1]
        test_predictions = (test_probabilities >= threshold).astype(int)
        payload = {
            "threshold": threshold,
            "probs": test_probabilities,
            "preds": test_predictions,
            "actual": y_test.to_numpy(),
            "f1": project.f1_score(y_test, test_predictions, zero_division=0),
            "accuracy": project.accuracy_score(y_test, test_predictions),
            "auc": project.roc_auc_score(y_test, test_probabilities),
        }
        if best is None or (payload["f1"], payload["accuracy"], payload["auc"]) > (best["f1"], best["accuracy"], best["auc"]):
            best = payload
            best_name = model_name
    return best_name, best


def save_confusion_matrix(best_name: str, payload: dict) -> None:
    matrix = project.confusion_matrix(payload["actual"], payload["preds"])
    image, draw = base_canvas("Confusion Matrix", f"En iyi model: {best_name}")

    start_x, start_y = 280, 190
    cell = 150
    labels = [["", "Tahmin 0", "Tahmin 1"], ["Gerçek 0", str(matrix[0][0]), str(matrix[0][1])], ["Gerçek 1", str(matrix[1][0]), str(matrix[1][1])]]
    colors = [[BG, BLUE, RED], [BLUE, "#DCEAF6", "#F8D7DA"], [BLUE, "#DCEAF6", "#F8D7DA"]]

    for r in range(3):
        for c in range(3):
            x1 = start_x + c * cell
            y1 = start_y + r * cell
            x2 = x1 + cell
            y2 = y1 + cell
            fill = BG if (r == 0 and c == 0) else (colors[r][c] if r < len(colors) and c < len(colors[r]) else BG)
            draw.rounded_rectangle((x1, y1, x2, y2), radius=12, outline="#64748B", width=2, fill=fill)
            text = labels[r][c]
            bbox = draw.textbbox((0, 0), text, font=font(22, bold=(r == 0 or c == 0)))
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((x1 + (cell - tw) / 2, y1 + (cell - th) / 2), text, fill=TEXT, font=font(22, bold=(r == 0 or c == 0)))

    draw.text((285, 655), "Yanlış negatiflerin düşük olması sağlık bağlamında önemli bir avantajdır.", fill="#475569", font=font(18))
    image.save(ASSETS_DIR / "best_confusion_matrix.png")


def save_roc_curve(best_name: str, payload: dict) -> None:
    fpr, tpr, _ = roc_curve(payload["actual"], payload["probs"])
    image, draw = base_canvas("ROC Eğrisi", f"En iyi model: {best_name}")
    left, top, right, bottom = 120, 170, 980, 620
    draw_axes(draw, left, top, right, bottom, [0, 0.25, 0.5, 0.75, 1.0], ["0.00", "0.25", "0.50", "0.75", "1.00"])

    def to_xy(xv: float, yv: float) -> tuple[int, int]:
        x = int(left + xv * (right - left))
        y = int(bottom - yv * (bottom - top))
        return x, y

    # diagonal
    draw.line([to_xy(0, 0), to_xy(1, 1)], fill="#94A3B8", width=2)

    points = [to_xy(float(xv), float(yv)) for xv, yv in zip(fpr, tpr)]
    for p1, p2 in zip(points[:-1], points[1:]):
        draw.line([p1, p2], fill=RED, width=5)

    draw.text((1020, 250), "Y ekseni: True Positive Rate", fill=TEXT, font=font(18))
    draw.text((1020, 285), "X ekseni: False Positive Rate", fill=TEXT, font=font(18))
    draw.text((1020, 350), f"ROC-AUC: {payload['auc']:.4f}", fill=TEXT, font=font(24, bold=True))
    image.save(ASSETS_DIR / "best_roc_curve.png")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    ASSETS_DIR.mkdir(exist_ok=True)
    raw_df = project.pd.read_csv(project.DATA_PATH)
    comparison_df = project.pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    save_class_distribution(raw_df)
    save_metric_comparison(comparison_df)
    best_name, payload = best_model_payload()
    save_confusion_matrix(best_name, payload)
    save_roc_curve(best_name, payload)


if __name__ == "__main__":
    main()
