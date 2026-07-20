#from ultralytics import YOLO

# Load a model
#model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

# Train the model
#results = model.train(data="coco.yaml", epochs=100, imgsz=640)
import urllib.request,os,sys,json,shutil,zipfile,random,subprocess,cv2, csv
import albumentations as A, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from ultralytics.data.converter import convert_coco
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO

URL={"train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"}

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "Files"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def download(url: str , out: Path):
    out.parent.mkdir(parents= True, exist_ok= True) 
    if out.exists():
        print(f"esta carpeta ya existe {out}")
        return out
    try:
        print(f"descargando... {url} en {out}")
        urllib.request.urlretrieve(url, out)
        print(f"descargado {out}")
        return out
    except Exception as e:
        print(f"no se pudo descargar {url}: {e}")
        return None

def unzip(path:Path):
    print(f"descomprimiendo {path.name}")
    dest_dir = path.with_suffix("")
    if dest_dir.exists():
        print(f"ya existe {dest_dir.name}, no se extrae")
        return None

    try:
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(path.parent)
        print("unzip completado")
        return dest_dir
    except Exception as e:
        print(f"no se pudo descomprimir {path}: {e}")
        return None

def write_coco_aug_yaml(yaml_path: Path, train_images: Path, val_images: Path, nc: int = 80):
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        "path: .\n"
        f"train: {train_images.as_posix()}\n"
        f"val: {val_images.as_posix()}\n"
        f"nc: {nc}\n"
        "names:\n" + "\n".join([f"  {i}: class_{i}" for i in range(nc)]) + "\n",
        encoding="utf-8"
    )

def yolo_read(txt: Path):
    if not txt.exists():
        return []
    out = []
    for ln in txt.read_text().splitlines():
        p = ln.split()
        if len(p) != 5:
            continue
        cls, x, y, w, h = p
        out.append([float(x), float(y), float(w), float(h), int(cls)])  # YOLO + cls
    return out

def yolo_write(txt: Path, bboxes):
    # bboxes: [x,y,w,h,cls] con coords normalizadas
    lines = [f"{b[4]} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for b in bboxes]
    txt.write_text("\n".join(lines) + ("\n" if lines else ""))

def denorm_to_uint8(img_norm):
    # img_norm está normalizada (float). La volvemos a uint8 para poder guardarla como imagen.
    img = (img_norm * STD + MEAN)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def save_augmented_examples(images_dir: str, out_dir: str, n: int = 50):
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tfm = A.Compose([
        A.Resize(640, 640),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Affine(translate_percent=(-0.05, 0.05), scale=(0.8, 1.2), rotate=(-20, 20),
                 mode=cv2.BORDER_CONSTANT, p=0.5),
        A.Blur(blur_limit=3, p=0.1),
        A.Normalize(mean=MEAN, std=STD),
    ])

    img_paths = list(images_dir.glob("*.jpg"))
    for i, p in enumerate(random.sample(img_paths, k=min(n, len(img_paths)))):
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = tfm(image=img)["image"]

        # desnormalizar para guardar
        aug = (aug * np.array(STD) + np.array(MEAN))
        aug = np.clip(aug * 255, 0, 255).astype(np.uint8)

        cv2.imwrite(str(out_dir / f"{p.stem}_aug.jpg"), cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))


def train_eval_infer(
    data_yaml: str,
    test_images: str,
    out_preds: str,
    load_path: str | None = None,
    resume: bool = False,
    epochs: int = 50,
    batch: int = 16,
    imgsz: int = 640,
    save_path: str = "yolov8_trained_model.pt",
    conf: float = 0.25,
    iou: float = 0.7,
    save_pred_csv: bool = True,
):
    # 1) Cargar modelo
    model = YOLO(load_path if load_path else "yolov8n.pt")

    # 2) Entrenar
    train_res = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        resume=resume,
        workers=0

    )
    # train_res suele incluir la ruta del run
    if hasattr(train_res, "save_dir"):
        print(f"[TRAIN] Resultados en: {train_res.save_dir}")

    # 3) Evaluar
    val_res = model.val(data=data_yaml, imgsz=imgsz)
    # Mostrar métricas clave de forma legible
    try:
        # En Ultralytics YOLOv8 val_res.box tiene métricas de detección
        mp = float(val_res.box.mp)     # mean precision
        mr = float(val_res.box.mr)     # mean recall
        map50 = float(val_res.box.map50)
        map5095 = float(val_res.box.map)
        print("\n[VAL] Métricas:")
        print(f"  Precision (mP):   {mp:.4f}")
        print(f"  Recall (mR):      {mr:.4f}")
        print(f"  mAP@0.50:         {map50:.4f}")
        print(f"  mAP@0.50:0.95:    {map5095:.4f}")
    except Exception:
        # fallback si cambia la API
        print("\n[VAL] Resultado completo:")
        print(val_res)

    if hasattr(val_res, "save_dir"):
        print(f"[VAL] Curvas/plots en: {val_res.save_dir}")

    # 4) Inferencia
    out_preds = Path(out_preds)
    out_preds.mkdir(parents=True, exist_ok=True)

    csv_path = out_preds / "preds.csv"
    csv_f = None
    writer = None
    if save_pred_csv:
        csv_f = open(csv_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_f)
        writer.writerow(["image", "class_id", "conf", "x1", "y1", "x2", "y2"])

    for img in sorted(Path(test_images).glob("*.jpg")):
        results = model.predict(
            source=str(img),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False
        )

        # Guarda imagen con cajas dibujadas
        results[0].save(filename=str(out_preds / f"{img.stem}_pred.jpg"))

        # Guarda detecciones en CSV
        if writer is not None:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, confs):
                    writer.writerow([img.name, int(c), float(cf), float(x1), float(y1), float(x2), float(y2)])

    if csv_f:
        csv_f.close()
        print(f"[PRED] CSV guardado en: {csv_path}")

    print(f"[PRED] Imágenes con predicciones en: {out_preds}")

    # 5) Guardar modelo final
    model.save(save_path)
    print(f"[MODEL] Guardado en: {save_path}")

    return YOLO(save_path)


def main():
    for filename, url in URL.items():
        out= OUT / filename
        zip_path = download(url,out)
        print
        if zip_path is not None: 
            unzip(zip_path)
    ann_dir = OUT / "annotations"

    instances_dir = OUT / "annotations_instances"
    instances_dir.mkdir(parents=True, exist_ok=True)

    for p in ann_dir.glob("instances_*.json"):
        shutil.copy2(p, instances_dir / p.name)

    convert_coco(
        labels_dir=str(instances_dir),
        save_dir=str(OUT),
        use_segments=False,
        use_keypoints=False,
        cls91to80=True,
    )

    train_images_dir = OUT / "train2017"          # imágenes reales
    val_images_dir   = OUT / "val2017"            # imágenes reales
    train_labels_dir = OUT / "labels" / "train2017"  # labels convertidas

    print("train images:", train_images_dir.exists(), "jpg:", len(list(train_images_dir.glob("*.jpg"))))
    print("train labels:", train_labels_dir.exists(), "txt:", len(list(train_labels_dir.glob("*.txt"))))
    print("val images:", val_images_dir.exists(), "jpg:", len(list(val_images_dir.glob("*.jpg"))))

    print("Generando ejemplos aumentados...")
    save_augmented_examples(
        images_dir=str(OUT / "train2017"),
        out_dir=str(OUT / "processed_examples"),
        n=50
    )
    print(f"Ejemplos aumentados guardados en: {OUT / 'processed_examples'}")

    yaml_path = OUT / "datasets" / "coco_aug.yaml"
    
    write_coco_aug_yaml(
        yaml_path=yaml_path,
         train_images=OUT / "train2017",
        val_images=OUT / "val2017",   
        nc=80
    )

    print(f"Archivo YAML para entrenamiento: {yaml_path}")
    print("Empezando entrenamiento con YOLOv8...")
    train_eval_infer(
    data_yaml=str(yaml_path),
    test_images=str(OUT / "inference_images"),
    out_preds=str(OUT / "inference_outputs")
    )
     
if __name__ == "__main__":
    main()

