import os
import json
from collections import namedtuple

from .evaluate import CookedDiagnostic


# img_path  Image path under root
# fp    Number of false positives
# fn    Number of false negatives
Abstraction = namedtuple('Abstraction', ['img_path', 'fp', 'fn'])


def abstract(cooked_diag, out_dir):
    """
    Get abstraction from given CookedDiagnostic and dump to abstract.json
    for further analyzing.
    """
    assert isinstance(cooked_diag, CookedDiagnostic)

    obj = []
    for img_path in cooked_diag.iter_image_paths():
        fp = 0
        fn = 0
        for _, cdet in cooked_diag.get_image_detection(img_path).items():
            fp += len(cdet.false_positives)
            fn += len(cdet.false_negatives)

        obj.append(Abstraction(img_path, fp, fn)._asdict())

    json_path = os.path.join(out_dir, 'abstract.json')
    with open(json_path, 'wt') as f:
        json.dump(obj, f, indent=2)
