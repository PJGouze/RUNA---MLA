## 1. Ce qu’il faut vraiment implémenter (vision réaliste)

Tu n’as pas besoin de réimplémenter :

- Faster R-CNN → Detectron2 le fait

- CLIP → déjà prêt

- Text encoder → déjà prêt

Ce qu’il faut coder :

1. extraction des régions (Detectron2)

2. extraction embeddings global + région (CLIP)

3. fusion embeddings

4. score OOD (max similarity)

5. fine-tuning projection layer (optionnel mais recommandé)

Donc 80% du travail = glue code.

## 2. Pipeline réel de RUNA

Le pipeline réel est :

```bash
Image
  ↓
Detectron2
  ↓
Bounding boxes

Pour chaque box:
    Crop → CLIP encoder (regional)
    Blur image → CLIP encoder (global)

Fusion embeddings
Similarity avec text embeddings
Score OOD
```

## 3. Installation propre (important)

```bash

conda create -n runa python=3.10
conda activate runa

pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
pip install detectron2 -f \
https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.1/index.html
``` 
et
```bash
pip install opencv-python
pip install tqdm
pip install matplotlib
``` 

## 4. Charger Detectron2 (detector gelé)

Code minimal :
```bash
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )
)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)
```


