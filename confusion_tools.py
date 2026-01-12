# confusion_tools.py
from __future__ import annotations
from typing import Sequence, Optional
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------- metrics helpers ----------
def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def accuracy_from_cm(cm: np.ndarray):
    """
    Per-class accuracy (TP+TN)/All and Macro Accuracy (mean over classes).
    Works for any number of classes.
    """
    cm = np.asarray(cm)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    total = cm.sum().astype(np.float64)
    tn = total - (tp + fp + fn)

    denom = tp + tn + fp + fn  # this is just 'total' repeated per class
    per_class_acc = np.divide(tp + tn, denom, out=np.zeros_like(tp), where=denom > 0)
    macro_acc = float(np.mean(per_class_acc)) if per_class_acc.size > 0 else 0.0
    #overall_acc = float(tp.sum() / total) if total > 0 else 0.0
    overall_acc = np.trace(cm) / np.sum(cm)
    return per_class_acc, macro_acc,overall_acc




def precision_recall_f1_from_cm(cm: np.ndarray):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp   #axis = 0 (Columns)
    fn = cm.sum(axis=1) - tp   #axis=1 (Rows)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall    = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1        = np.divide(2 * precision * recall, precision + recall,
                          out=np.zeros_like(tp), where=(precision + recall) > 0)
    macro_f1  = float(np.mean(f1)) if f1.size > 0 else 0.0
    return precision, recall, f1, macro_f1

# ---------- plotting ----------
def _plot_cm(cm: np.ndarray, class_names: Sequence[str], title: str,
             save_path: Optional[str] = None, normalize: bool = False) -> None:
    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums

    plt.figure(figsize=(4.8, 4.2))
    im = plt.imshow(cm, interpolation="nearest", cmap="viridis")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.5
    fmt = ".2f" if normalize else "d"
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=9)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

class ConfusionTracker:
    """
    Save confusion-matrix images during training.

    Two saving modes:
      1) Folder mode:   ConfusionTracker(class_names, out_dir="cm_plots", keep_history=True/False)
      2) Prefix mode:   ConfusionTracker(class_names, save_prefix="runs/emotion_classifier", keep_history=False)

    If both out_dir and save_prefix are provided, save_prefix takes precedence.
    """
    def __init__(self,
                 class_names: Sequence[str],
                 out_dir: Optional[str] = None,
                 keep_history: bool = False,
                 *,
                 save_prefix: Optional[str] = None,
                 normalize: bool = False):
        self.class_names = list(class_names)
        self.n_classes = len(self.class_names)
        self.out_dir = out_dir
        self.keep_history = keep_history
        self.save_prefix = save_prefix
        self.normalize = normalize

        self.best_macro_f1 = -1.0
        self.best_epoch = None
        self.paths = {"baseline": None, "best": None, "final": None}

    # path builder that supports both modes
    def _path(self, stem: str) -> Optional[str]:
        if self.save_prefix:  # prefix mode
            base = f"{self.save_prefix}_{stem}.png"
            os.makedirs(os.path.dirname(base), exist_ok=True) if os.path.dirname(base) else None
            return base
        if self.out_dir:      # folder mode
            os.makedirs(self.out_dir, exist_ok=True)
            return os.path.join(self.out_dir, f"{stem}.png")
        return None  # nothing to save

    def save_baseline(self, cm: np.ndarray):
        p = self._path("cm_baseline")
        _plot_cm(cm, self.class_names, "Baseline (no training)", p, normalize=self.normalize)
        self.paths["baseline"] = p

    def update_if_best(self, cm: np.ndarray, macro_f1: float, epoch: int):
        if macro_f1 > self.best_macro_f1 + 1e-12:
            self.best_macro_f1 = macro_f1
            self.best_epoch = epoch

            if self.keep_history:
                stem = f"cm_best_epoch_{epoch:03d}"
            else:
                # overwrite one file; delete previous best if it existed
                stem = "cm_best"
                old = self.paths.get("best")
                if old and os.path.exists(old):
                    try: os.remove(old)
                    except OSError: pass

            p = self._path(stem)
            _plot_cm(cm, self.class_names,
                     f"Best so far (epoch {epoch}, macroF1={macro_f1:.3f})",
                     p, normalize=self.normalize)
            self.paths["best"] = p

    def save_final(self, cm: np.ndarray):
        p = self._path("cm_final")
        _plot_cm(cm, self.class_names, "Final", p, normalize=self.normalize)
        self.paths["final"] = p


"""
Small guide:
Rows = actual (true) classes
Columns = predicted classes

| Term                    | Meaning                                                     | Computation (row/col logic)                        |
| ----------------------- | ----------------------------------------------------------- | -------------------------------------------------- |
| **TP (True Positive)**  | Model correctly predicted class *i*                         | `CM[i, i]`                                         |
| **FP (False Positive)** | Model predicted class *i* but true class was something else | Sum of **column i** (predicted=i) minus `CM[i, i]` |
| **FN (False Negative)** | True class was *i*, but model predicted something else      | Sum of **row i** (true=i) minus `CM[i, i]`         |
| **TN (True Negative)**  | All other cells not involving class *i*                     | total − (TP + FP + FN)                             |


"""