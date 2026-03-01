import numpy as np
from typing import Sequence, Union
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from torch import Tensor

EmbeddingLike = Union[np.ndarray, Sequence[float], Tensor]

def hungarian_cosine_match(list_a: Sequence[EmbeddingLike], list_b: Sequence[EmbeddingLike]):
    """
    Hungarian matching between two embedding lists using cosine similarity.

    Parameters
    ----------
    list_a : list[np.ndarray]  (length = 4)
    list_b : list[np.ndarray]  (variable length)

    Returns
    -------
    matches : list of dict
        [
            {
                "a_index": int,
                "b_index": int,
                "similarity": float
            },
            ...
        ]
    """

    if len(self.list_a) == 0 or len(list_b) == 0:
        return []

    A = np.asarray(list_a, dtype=float)
    B = np.asarray(list_b, dtype=float)

    cost_matrix = cdist(A, B, metric="cosine")

    # --- Hungarian assignment ---
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # --- collect matches ---
    matches = []
    for r, c in zip(row_ind, col_ind):
        similarity = 1 - cost_matrix[r, c]
        matches.append({
            "a_index": int(r),
            "b_index": int(c),
            "similarity": float(similarity),
        })

    return matches