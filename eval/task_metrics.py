import numpy as np
from typing import Dict, List, Tuple

def concordance_index(
    predicted_risk: np.ndarray,
    survival_time: np.ndarray,
    event: np.ndarray,
) -> float:

    N = len(predicted_risk)
    concordant = 0
    discordant = 0
    tied_risk = 0

    for i in range(N):
        for j in range(i + 1, N):

            if event[i] == 1 and survival_time[i] < survival_time[j]:
                if predicted_risk[i] > predicted_risk[j]:
                    concordant += 1
                elif predicted_risk[i] < predicted_risk[j]:
                    discordant += 1
                else:
                    tied_risk += 0.5

            elif event[j] == 1 and survival_time[j] < survival_time[i]:
                if predicted_risk[j] > predicted_risk[i]:
                    concordant += 1
                elif predicted_risk[j] < predicted_risk[i]:
                    discordant += 1
                else:
                    tied_risk += 0.5

            elif event[i] == 1 and event[j] == 1 and survival_time[i] == survival_time[j]:
                tied_risk += 1

    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied_risk) / total

def concept_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    concept_names: List[str] = None,
) -> Dict[str, Dict[str, float]]:

    N, C = predicted.shape
    if concept_names is None:
        concept_names = [f"c{i+1}" for i in range(C)]

    results = {}
    for i, name in enumerate(concept_names):
        pred_i = predicted[:, i]
        tgt_i = target[:, i]

        mse = float(np.mean((pred_i - tgt_i) ** 2))
        mae = float(np.mean(np.abs(pred_i - tgt_i)))

        if np.std(pred_i) > 1e-8 and np.std(tgt_i) > 1e-8:
            pearson_r = float(np.corrcoef(pred_i, tgt_i)[0, 1])
        else:
            pearson_r = 0.0

        results[name] = {
            "mse": mse,
            "mae": mae,
            "pearson_r": pearson_r,
        }

    return results

def hazard_to_risk(hazard_logits: np.ndarray) -> np.ndarray:

    hazard = 1.0 / (1.0 + np.exp(-hazard_logits))
    survival = np.prod(1.0 - hazard, axis=1)
    risk = 1.0 - survival
    return risk

def compute_time_bins(
    survival_times: np.ndarray,
    events: np.ndarray,
    num_bins: int = 4,
) -> np.ndarray:

    event_times = survival_times[events == 1]
    if len(event_times) == 0:

        event_times = survival_times

    quantiles = np.linspace(0, 100, num_bins + 1)[1:]
    bins = np.percentile(event_times, quantiles)

    return bins.astype(np.float32)
