# Hyperparameter Optimization Report

## Optimization Summary
- **Date**: 2025-07-01T15:45:30.701572
- **Number of Trials**: 15
- **Best mAP50 Score**: 0.8443
- **Best Trial Number**: 1

## Best Hyperparameters

| Parameter | Value |
|-----------|-------|
| epochs | 67 |
| batch | 16 |
| imgsz | 768 |
| lr0 | 0.004138040112561018 |
| lrf | 0.15926074689495165 |
| momentum | 0.8175809805211491 |
| weight_decay | 0.0007158097238609412 |
| warmup_epochs | 3 |
| box | 5.610191174223894 |
| cls | 0.498070764044508 |
| dfl | 1.0343885211152184 |
| patience | 19 |
| hsv_h | 0.02587799816000169 |
| hsv_s | 0.662522284353982 |
| hsv_v | 0.28053996848046986 |
| degrees | 23.403060953001486 |
| translate | 0.10934205586865593 |
| scale | 0.16636900997297435 |
| shear | 9.695846277645586 |
| perspective | 0.0007751328233611145 |
| flipud | 0.46974947078209456 |
| mosaic | 0.8948273504276488 |
| mixup | 0.17936999364332554 |
| copy_paste | 0.27656227050693505 |

## Training Command
```bash
yolo detect train \
  model=/home/ailaty3088@id.sdsu.edu/Ashter/runs/detect/yolov8n_finetune_old_dataset/weights/best.pt \
  data=/home/ailaty3088@id.sdsu.edu/Ashter/dataset/data.yaml \
  epochs=67 \
  batch=16 \
  imgsz=768 \
  lr0=0.004138040112561018 \
  lrf=0.15926074689495165 \
  momentum=0.8175809805211491 \
  weight_decay=0.0007158097238609412 \
  warmup_epochs=3 \
  box=5.610191174223894 \
  cls=0.498070764044508 \
  dfl=1.0343885211152184 \
  patience=19 \
  hsv_h=0.02587799816000169 \
  hsv_s=0.662522284353982 \
  hsv_v=0.28053996848046986 \
  degrees=23.403060953001486 \
  translate=0.10934205586865593 \
  scale=0.16636900997297435 \
  shear=9.695846277645586 \
  perspective=0.0007751328233611145 \
  flipud=0.46974947078209456 \
  mosaic=0.8948273504276488 \
  mixup=0.17936999364332554 \
  copy_paste=0.27656227050693505 \
  project=optimized_model \
  name=best_hyperparameters
```

## Trial Results Summary

| Trial | mAP50 | Status |
|-------|-------|--------|
| 0 | 0.7865 | ✅ |
| 1 | 0.8443 | ✅ |
| 2 | 0.7361 | ✅ |
| 3 | 0.7642 | ✅ |
| 4 | 0.7418 | ✅ |
| 5 | 0.6499 | ✅ |
| 6 | 0.7737 | ✅ |
| 7 | 0.8210 | ✅ |
| 8 | 0.7590 | ✅ |
| 9 | 0.7630 | ✅ |
| 10 | 0.7674 | ✅ |
| 11 | 0.8352 | ✅ |
| 12 | 0.7517 | ✅ |
| 13 | 0.8202 | ✅ |
| 14 | 0.7322 | ✅ |
