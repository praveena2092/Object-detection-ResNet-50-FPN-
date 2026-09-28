# Data layout

Download the competition data and place it here (or point `--data-root` elsewhere):

```
data/
├── train/
│   ├── images/          # 7500 training images
│   └── labels/          # YOLO txt: class xc yc w h (normalized)
├── test/
│   └── images/          # 525 test images
└── sample_submission.csv
```

Classes: aegypti 0, albopictus 1, anopheles 2, culex 3, culiseta 4, japonicus/koreicus 5.
