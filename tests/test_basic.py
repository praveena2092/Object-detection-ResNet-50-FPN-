import pandas as pd
import pytest

from mosquito.config import CLASSES, load_config
from mosquito.dataset import split_train_val, yolo_to_xyxy
from mosquito.submission import finalize_submission, validate_submission


def test_yolo_to_xyxy():
    assert yolo_to_xyxy(0.5, 0.5, 0.2, 0.4, 100, 200) == pytest.approx([40, 60, 60, 140])


def test_split_is_deterministic():
    paths = [f"{i}.jpg" for i in range(100)]
    a = split_train_val(paths, 0.15, 42)
    b = split_train_val(paths, 0.15, 42)
    assert a == b and len(a[1]) == 15


def test_unknown_config_key():
    with pytest.raises(ValueError):
        load_config(None, not_a_key=1)


def test_submission_fill_and_order(tmp_path):
    sample = pd.DataFrame({"id": [0, 1], "ImageID": ["b.jpg", "a.jpg"], "LabelName": ["aegypti"] * 2,
                           "Conf": [1, 1], "xcenter": [.5, .5], "ycenter": [.5, .5],
                           "bbx_width": [.1, .1], "bbx_height": [.1, .1]})
    p = tmp_path / "s.csv"
    sample.to_csv(p, index=False)
    preds = pd.DataFrame([
        {"ImageID": "a.jpg", "class_id": 3, "Conf": 0.9, "xcenter": .4, "ycenter": .4, "bbx_width": .2, "bbx_height": .2},
        {"ImageID": "b.jpg", "class_id": None, "Conf": None, "xcenter": None, "ycenter": None,
         "bbx_width": None, "bbx_height": None},
    ])
    sub = finalize_submission(preds, str(p))
    assert list(sub["ImageID"]) == ["b.jpg", "a.jpg"]
    assert sub.loc[1, "LabelName"] == CLASSES[3]
    assert validate_submission(sub, str(p)) == []
