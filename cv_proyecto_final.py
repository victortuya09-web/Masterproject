import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from PIL import Image, ImageSequence

warnings.filterwarnings("ignore")


# CONFIGURACIÓN
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
CSV_PATH = DATA_DIR / "catalog.csv"
FEATURES_PATH = DATA_DIR / "catalog_features.npy"
FILENAMES_PATH = DATA_DIR / "catalog_filenames.npy"
LABELS_PATH = DATA_DIR / "catalog_labels.npy"

IMAGE_SIZE = (224, 224)
TOP_K = 5


# PREPARAR CATÁLOGO AUTOMÁTICO DESDE NIKA.JPG
def prepare_demo_catalog_from_nika() -> Path:
    """
    Si solo existe nika.jpg en la carpeta principal del proyecto,
    crea automáticamente un catálogo de prueba reutilizando esa imagen.
    """
    source_image = BASE_DIR / "nika.jpg"

    if not source_image.exists():
        raise FileNotFoundError(
            f"No se encontró la imagen base: {source_image}"
        )

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    demo_files = [
        ("nika_1.jpg", "ropa"),
        ("nika_2.jpg", "ropa"),
        ("nika_3.jpg", "moda"),
        ("nika_4.jpg", "casual"),
        ("nika_5.jpg", "estilo"),
    ]

    for filename, _ in demo_files:
        target_path = IMAGES_DIR / filename
        shutil.copy2(source_image, target_path)

    catalog_df = pd.DataFrame(demo_files, columns=["filename", "label"])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    catalog_df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    print(f"[OK] Catálogo demo creado en: {CSV_PATH}")
    print(f"[OK] Imágenes demo creadas en: {IMAGES_DIR}")

    return source_image


# CARGA DEL MODELO PREENTRENADO
def load_feature_extractor():
    model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    return model


# PREPROCESAMIENTO DE IMÁGENES
def load_and_preprocess_image(img_path):
    img_path = Path(img_path)

    if not img_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {img_path}")

    img = Image.open(img_path)

    # 👉 Detectar GIF animado
    if getattr(img, "is_animated", False):
        img = ImageSequence.Iterator(img)[0]  # primer frame

    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE)

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array


# EXTRACCIÓN DE CARACTERÍSTICAS
def extract_features(model, img_path):
    preprocessed = load_and_preprocess_image(img_path)
    features = model.predict(preprocessed, verbose=0)
    features = normalize(features, norm="l2")
    return features.flatten()


# CARGA Y VALIDACIÓN DEL CATÁLOGO
def validate_catalog(df):
    if "filename" not in df.columns:
        raise ValueError(
            "El archivo catalog.csv debe contener al menos la columna 'filename'."
        )

    if "label" not in df.columns:
        df["label"] = "Sin etiqueta"

    return df


def load_catalog(csv_path, images_dir):
    csv_path = Path(csv_path)
    images_dir = Path(images_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el CSV del catálogo: {csv_path}")

    if not images_dir.exists():
        raise FileNotFoundError(f"No se encontró la carpeta de imágenes: {images_dir}")

    df = pd.read_csv(csv_path)
    df = validate_catalog(df)

    df["image_path"] = df["filename"].apply(lambda x: str(images_dir / str(x)))
    df["exists"] = df["image_path"].apply(lambda x: Path(x).exists())

    missing = df.loc[~df["exists"], "filename"].tolist()
    if missing:
        print("[AVISO] Las siguientes imágenes no existen y serán ignoradas:")
        for name in missing:
            print(f"   - {name}")

    df = df[df["exists"]].copy().reset_index(drop=True)

    if df.empty:
        raise ValueError("No hay imágenes válidas en el catálogo tras la validación.")

    return df[["filename", "label", "image_path"]]


# GENERACIÓN DE EMBEDDINGS DEL CATÁLOGO
def build_catalog_features(model, catalog_df):
    features_list = []
    filenames = []
    labels = []

    print("[INFO] Extrayendo características del catálogo...")

    for idx, row in catalog_df.iterrows():
        try:
            feat = extract_features(model, row["image_path"])
            features_list.append(feat)
            filenames.append(row["filename"])
            labels.append(row["label"])
            print(f"   [{idx + 1}/{len(catalog_df)}] OK -> {row['filename']}")
        except Exception as e:
            print(f"   [{idx + 1}/{len(catalog_df)}] ERROR -> {row['filename']}: {e}")

    if not features_list:
        raise ValueError("No se pudieron extraer características de ninguna imagen.")

    features_array = np.vstack(features_list)
    filenames = np.array(filenames)
    labels = np.array(labels)

    return features_array, filenames, labels


# GUARDADO / CARGA DE EMBEDDINGS
def save_catalog_features(features, filenames, labels):
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(FEATURES_PATH, features)
    np.save(FILENAMES_PATH, filenames)
    np.save(LABELS_PATH, labels)
    print(f"[OK] Embeddings guardados en: {FEATURES_PATH}")


def load_saved_catalog_features():
    if not FEATURES_PATH.exists() or not FILENAMES_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError("No existen embeddings guardados del catálogo.")

    features = np.load(FEATURES_PATH, allow_pickle=True)
    filenames = np.load(FILENAMES_PATH, allow_pickle=True)
    labels = np.load(LABELS_PATH, allow_pickle=True)

    return features, filenames, labels


# CÁLCULO DE SIMILITUD
def compute_similarity(query_features, catalog_features):
    query_features = query_features.reshape(1, -1)
    similarities = cosine_similarity(query_features, catalog_features)
    return similarities.flatten()


# RECOMENDACIÓN
def recommend_similar_products(
    query_img_path,
    model,
    catalog_features,
    catalog_filenames,
    catalog_labels,
    top_k=TOP_K,
):
    query_features = extract_features(model, query_img_path)
    similarities = compute_similarity(query_features, catalog_features)

    top_k = min(top_k, len(similarities))
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = pd.DataFrame(
        {
            "filename": catalog_filenames[top_indices],
            "label": catalog_labels[top_indices],
            "similarity_score": similarities[top_indices],
        }
    )

    return results.reset_index(drop=True)


# VISUALIZACIÓN
def open_image_rgb(img_path):
    return Image.open(img_path).convert("RGB")


def plot_recommendations(query_img_path, results_df, images_dir):
    images_dir = Path(images_dir)
    total_plots = len(results_df) + 1

    plt.figure(figsize=(4 * total_plots, 4))

    plt.subplot(1, total_plots, 1)
    plt.imshow(open_image_rgb(query_img_path))
    plt.title("Consulta")
    plt.axis("off")

    for i, row in results_df.iterrows():
        img_path = images_dir / row["filename"]
        plt.subplot(1, total_plots, i + 2)
        plt.imshow(open_image_rgb(img_path))
        plt.title(f"{row['label']}\nScore: {row['similarity_score']:.4f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# FUNCIÓN PRINCIPAL
def main(rebuild_features=True, top_k=5):
    query_image_path = BASE_DIR / "nika.jpg"

    print("[INFO] Ruta de consulta:", query_image_path)
    print("[INFO] Cargando extractor de características...")
    model = load_feature_extractor()

    if rebuild_features or not (
        FEATURES_PATH.exists() and FILENAMES_PATH.exists() and LABELS_PATH.exists()
    ):
        print("[INFO] Generando embeddings del catálogo desde cero...")
        catalog_df = load_catalog(CSV_PATH, IMAGES_DIR)
        catalog_features, catalog_filenames, catalog_labels = build_catalog_features(
            model, catalog_df
        )
        save_catalog_features(catalog_features, catalog_filenames, catalog_labels)
    else:
        print("[INFO] Cargando embeddings guardados del catálogo...")
        catalog_features, catalog_filenames, catalog_labels = load_saved_catalog_features()

    print("[INFO] Buscando productos similares...")
    results = recommend_similar_products(
        query_img_path=query_image_path,
        model=model,
        catalog_features=catalog_features,
        catalog_filenames=catalog_filenames,
        catalog_labels=catalog_labels,
        top_k=top_k,
    )

    print("\n TOP RECOMENDACIONES ")
    print(results)

    plot_recommendations(query_image_path, results, IMAGES_DIR)


# EJECUCIÓN
if __name__ == "__main__":
    main(rebuild_features=True, top_k=5)