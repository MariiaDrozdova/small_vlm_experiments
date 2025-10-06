import re
import numpy as np
from typing import List, Optional
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def extract_last_class(
    text: str,
    all_labels: List[str],
    *,
    case_sensitive: bool = True,
    word_boundary: bool = True
) -> Optional[str]:
    """
    Return the last occurrence of any label in `all_labels` found in `text`.
    If none is found, return None.

    Args:
        text: The input text to search.
        all_labels: List of label strings to match (e.g., ["FR-I", "FR-II"] or
                    ["edge_on_disk", "smooth_cigar", "smooth_round", "unbarred_spiral"]).
        case_sensitive: If False, matches case-insensitively but returns the canonical
                        casing from `all_labels`.
        word_boundary: If True, only match labels that are not embedded within a larger
                       alphanumeric token (uses (?<!\\w) ... (?!\\w)).

    Examples:
        extract_last_class("... FR-I ... FR-II", ["FR-I", "FR-II"]) -> "FR-II"
        extract_last_class("edge_on_disk then smooth_round", ["edge_on_disk","smooth_round"]) -> "smooth_round"
    """
    if not all_labels:
        return None

    escaped = [re.escape(lbl) for lbl in all_labels if lbl]
    if not escaped:
        return None
    escaped.sort(key=len, reverse=True)
    alternation = "|".join(escaped)

    pattern = rf"({alternation})"
    if word_boundary:
        pattern = rf"(?<!\w){pattern}(?!\w)"

    flags = 0 if case_sensitive else re.IGNORECASE

    last_match_text = None
    for m in re.finditer(pattern, text, flags):
        last_match_text = m.group(1)

    if last_match_text is None:
        return None

    if case_sensitive:
        return last_match_text

    canon_map = {lbl.lower(): lbl for lbl in all_labels}
    return canon_map.get(last_match_text.lower(), last_match_text)


def f1_from_confusion_matrix(cm: np.ndarray):
    """
    Compute per-class F1 and macro-F1 from a confusion matrix.

    Args:
        cm (np.ndarray): Confusion matrix of shape (C, C), where cm[i, j] 
                         is the count of true class i predicted as class j.

    Returns:
        per_class_f1 (np.ndarray): F1 score for each class.
        macro_f1 (float): Unweighted average of per-class F1 scores.
    """
    tp = np.diag(cm).astype(float)
    predicted_counts = cm.sum(axis=0).astype(float)
    actual_counts = cm.sum(axis=1).astype(float)

    precision = np.divide(tp, predicted_counts, out=np.zeros_like(tp), where=predicted_counts!=0)
    recall    = np.divide(tp, actual_counts,    out=np.zeros_like(tp), where=actual_counts!=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_f1 = 2 * (precision * recall) / (precision + recall)
        per_class_f1 = np.nan_to_num(per_class_f1)  # replace NaNs with 0

    macro_f1 = per_class_f1.mean()
    return per_class_f1, macro_f1

def report_results(y_true, y_pred, print_text=True, all_labels=["FR-I","FR-II",]):
    labels = all_labels
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if print_text:
        print(cm)

        total = cm.sum()
        correct = np.trace(cm)
        errors = total - correct

        error_rate = errors / total
        print("Error rate", error_rate )

        print("\nClassification report:\n")
        print(classification_report(y_true, y_pred, labels=labels))
    res = {}
    res["f1"] = f1_from_confusion_matrix(cm)[1]
    res["cm"] = cm
    res["errors"] = errors
    return res
