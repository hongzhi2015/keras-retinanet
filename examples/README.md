# Data collection

Double check the collected images to make sure orentation of the page is correct. Pay attention to OpenCV version, OpenCV 2.4 does not support EXIF, use OpenCV 3.

```
python dlib_xml_to_ssd_csv.py math5000.xml
tail -996 math5000.csv > math5000_testing.csv
head -7227 math5000.csv > math5000_training.csv
```

# Train

Keras (2.1.3)
tensorflow (1.3.0)

```
python3 keras_retinanet/bin/train.py --batch-size 4 --gpu 0 --epochs 30 --steps 700 --no-weights --snapshot-path examples/snapshots5_augment --image_dir 5k/sheetimages csv examples/math5000_training.csv examples/label.csv --val-annotations examples/math5000_testing.csv
```
Result:
```
Epoch 27/30
mAP: 0.8820

700/700 [==============================] - 667s 953ms/step - loss: 1.3088 - regression_loss: 1.1491 - classification_loss: 0.1597
```

# Evaluation

```
python3 keras_retinanet/bin/evaluate.py --iou-threshold 0.2 --save-path examples/prediction5 --output_metrics examples/prediction5/metrics_27.pkl --image_dir 5k/sheetimages csv examples/math5000_testing.csv examples/label.csv examples/snapshots5_augment/resnet18_csv_27.h5
```
Result is:
mAP: 0.8820

plot [roc curve](prediction5_metrics_27.jpg):
```
python examples/plot_roc.py examples/prediction5/metrics_27.pkl examples/prediction5_metrics_27.jpg
```


# Export pb

Run the section "Load RetinaNet Model" in ResNet50RetinaNet.ipynb to export protobuf

# Run Prediction
```
python examples/predict_retinanet.py examples/snapshots5_augment/resnet18_csv_27.pb 0.36 FpT2WxTUqy6cBBWtt9oG2oDpLobL.jpg
```
