import numpy as np
import matplotlib.pyplot as plt

# --- Helper functions ---

def get_candidate(channel, arr):
    """
    Given a channel and its probability array for one event,
    returns candidate jet indices (as a list) corresponding to the maximum value.
    
    For channel 0 (hadronic, shape (16,16,16)): returns [a, b, c]
      where a,b are the first two jets (to be assigned prov 1) and c the third (prov 2).
    For channel 1 (leptonic, shape (16,)): returns [j]
    For channel 2 (Higgs, shape (16,16)): returns [r, c]
    
    If the maximum value is -∞, returns None.
    """
    max_val = np.max(arr)
    if np.isneginf(max_val):
        return None
    if channel == 0:
        inds = np.where(arr == max_val)
        return [inds[0][0], inds[1][0], inds[2][0]]
    elif channel == 1:
        inds = np.where(arr == max_val)[0]
        return [inds[0]]
    elif channel == 2:
        inds = np.where(arr == max_val)
        return [inds[0][0], inds[1][0]]


def assign_candidate(channel, candidate, i, prov):
    """
    Given a candidate (list of jet indices) for the given channel,
    assign the appropriate provenance in event i.
    - For channel 0 (hadronic): candidate[0] and candidate[1] -> prov 5; candidate[2] -> prov 2.
    - For channel 1 (leptonic): candidate[0] -> prov 3.
    - For channel 2 (Higgs): candidate[0] and candidate[1] -> prov 1.
    """
    if channel == 0:
        prov[i, candidate[0]] = 5
        prov[i, candidate[1]] = 5
        prov[i, candidate[2]] = 2
    elif channel == 1:
        prov[i, candidate[0]] = 3
    elif channel == 2:
        prov[i, candidate[0]] = 1
        prov[i, candidate[1]] = 1


def remove_candidate(channel, candidate, arr):
    """
    Sets the candidate's probability value to -∞ in the provided array.
    """
    if candidate is None:
        return
    if channel == 0:
        arr[candidate[0], candidate[1], candidate[2]] = -np.float32(np.inf)
    elif channel == 1:
        arr[candidate[0]] = -np.float32(np.inf)
    elif channel == 2:
        arr[candidate[0], candidate[1]] = -np.float32(np.inf)


def has_overlap(candidate, assigned):
    """
    Returns True if any element in candidate is already present in assigned.
    Both candidate and assigned should be lists of jet indices.
    """
    if candidate is None:
        return False
    for x in candidate:
        if x in assigned:
            return True
    return False

# --- Standard assignment function (descending order based on overall max. probability) ---

def assign_prov(outputs):
    """
    Standard assignment:
    
    For each event, determine the channel (0: hadronic, 1: leptonic, 2: Higgs)
    with the highest overall probability (taken from the full array for that channel).
    Assign its candidate jets.
    Then, for the remaining two channels, try (in a while-loop) to assign the candidate
    that does not overlap with the jets already chosen.
    The hadronic channel assigns three jets (first two get prov 5, third gets prov 2),
    leptonic assigns one jet (prov 3), and Higgs assigns a pair (prov 1).
    
    Additionally, a count is kept in assign_order_counts (shape (3, 3)) for each channel:
      - Column 0: times assigned as the first candidate
      - Column 1: times assigned as the second candidate
      - Column 2: times assigned as the third candidate
    """
    N = outputs[0].shape[0]
    prov = np.zeros((N, 16), dtype=int)
    assign_order_counts = np.zeros((3, 3), dtype=int)

    for i in range(N):
        out = [outputs[0][i].copy(), outputs[1][i].copy(), outputs[2][i].copy()]
        out2 = [outputs[0][i].copy(), outputs[1][i].copy(), outputs[2][i].copy()]

        # First assignment
        max_vals = [np.max(outputs[ch][i]) for ch in [0, 1, 2]]
        tag = np.argmax(max_vals)
        candidate1 = get_candidate(tag, outputs[tag][i])
        if candidate1 is None:
            continue
        assign_candidate(tag, candidate1, i, prov)
        assign_order_counts[tag, 0] += 1

        # Second assignment
        other = [ch for ch in [0, 1, 2] if ch != tag]
        tag2 = None
        while True:
            vals = [np.max(out[ch]) for ch in other]
            if np.all(np.isneginf(vals)):
                break
            idx = np.argmax(vals)
            tag2 = other[idx]
            candidate2 = get_candidate(tag2, out[tag2])
            if candidate2 is None:
                break
            if not has_overlap(candidate2, candidate1):
                assign_candidate(tag2, candidate2, i, prov)
                assign_order_counts[tag2, 1] += 1
                break
            remove_candidate(tag2, candidate2, out[tag2])

        # Third assignment
        remaining = [ch for ch in [0, 1, 2] if ch not in [tag, tag2]]
        tag3 = remaining[0] if len(remaining) == 1 else other[0]
        while True:
            candidate3 = get_candidate(tag3, out2[tag3])
            if candidate3 is None:
                break
            assigned_list = candidate1.copy()
            if 'candidate2' in locals() and candidate2 is not None:
                assigned_list.extend(candidate2)
            if not has_overlap(candidate3, assigned_list):
                assign_candidate(tag3, candidate3, i, prov)
                assign_order_counts[tag3, 2] += 1
                break
            remove_candidate(tag3, candidate3, out2[tag3])

    return prov, assign_order_counts

# --- Higgs-first assignment function ---

def assign_prov_higgs_first(outputs):
    """
    Higgs-first assignment:
    
    For each event, always assign the Higgs candidate (channel 2) first.
    Then, among the remaining channels (0: hadronic and 1: leptonic), assign in
    descending order of their current maximum, ensuring that candidate jets do not overlap
    with those already chosen.
    """
    N = outputs[0].shape[0]
    prov = np.zeros((N, 16), dtype=int)
    assign_order_counts = np.zeros((3, 3), dtype=int)

    for i in range(N):
        out = [outputs[0][i].copy(), outputs[1][i].copy(), outputs[2][i].copy()]
        out2 = [outputs[0][i].copy(), outputs[1][i].copy(), outputs[2][i].copy()]

        # Higgs first
        candidate_h = get_candidate(2, outputs[2][i])
        if candidate_h is not None:
            assign_candidate(2, candidate_h, i, prov)
            assign_order_counts[2, 0] += 1

        # Leptonic/hadronic second
        other = [0, 1]
        tag2 = None
        while True:
            vals = [np.max(out[ch]) for ch in other]
            if np.all(np.isneginf(vals)):
                break
            idx = np.argmax(vals)
            tag2 = other[idx]
            candidate2 = get_candidate(tag2, out[tag2])
            if candidate2 is None:
                break
            if candidate_h is None or not has_overlap(candidate2, candidate_h):
                assign_candidate(tag2, candidate2, i, prov)
                assign_order_counts[tag2, 1] += 1
                break
            remove_candidate(tag2, candidate2, out[tag2])

        # Third assignment
        remaining = [ch for ch in [0, 1] if ch != tag2]
        tag3 = remaining[0] if len(remaining) == 1 else other[0]
        while True:
            candidate3 = get_candidate(tag3, out2[tag3])
            if candidate3 is None:
                break
            assigned_list = []
            if candidate_h is not None:
                assigned_list.extend(candidate_h)
            if 'candidate2' in locals() and candidate2 is not None:
                assigned_list.extend(candidate2)
            if not has_overlap(candidate3, assigned_list):
                assign_candidate(tag3, candidate3, i, prov)
                assign_order_counts[tag3, 2] += 1
                break
            remove_candidate(tag3, candidate3, out2[tag3])

    return prov, assign_order_counts

def compute_confusion(prov_pred, prov_true):
    """
    Compute a normalized confusion matrix for jet provenance:
    - Rows: predicted categories ['H b1','H b2','t_had b','t_lep b','t_had q1','t_had q2']
    - Columns: true categories ['unmatched','gluons','H b','t_had b','t_lep b','t_had q']

    Returns normalized matrix and labels.
    """
    N = len(prov_pred)

    row_labels = ['H b1', 'H b2', 't_had b', 't_lep b', 't_had q1', 't_had q2']
    col_labels = ['unmatched', 'gluons', 'H b', 't_had b', 't_lep b', 't_had q']
    row_idx = {lbl: i for i, lbl in enumerate(row_labels)}
    col_idx = {lbl: i for i, lbl in enumerate(col_labels)}

    cm = np.zeros((len(row_labels), len(col_labels)), dtype=int)

    for p, t in zip(prov_pred, prov_true):
        p = np.asarray(p).ravel()
        t = np.asarray(t).ravel()

        pred_cat = [None] * 16
        higgs_jets = np.where(p == 1)[0]
        for k, jet in enumerate(sorted(higgs_jets)[:2]):
            pred_cat[jet] = f'H b{k+1}'
        for jet in np.where(p == 2)[0]:
            pred_cat[jet] = 't_had b'
        for jet in np.where(p == 3)[0]:
            pred_cat[jet] = 't_lep b'
        qq_jets = np.where(p == 5)[0]
        for k, jet in enumerate(sorted(qq_jets)[:2]):
            pred_cat[jet] = f't_had q{k+1}'

        true_cat = [None] * 16
        for jet in np.where(t == -1)[0]:
            true_cat[jet] = 'unmatched'
        for jet in np.where(t == 4)[0]:
            true_cat[jet] = 'gluons'
        for jet in np.where(t == 1)[0]:
            true_cat[jet] = 'H b'
        for jet in np.where(t == 2)[0]:
            true_cat[jet] = 't_had b'
        for jet in np.where(t == 3)[0]:
            true_cat[jet] = 't_lep b'
        for jet in np.where(t == 5)[0]:
            true_cat[jet] = 't_had q'

        for jet in range(16):
            rp = pred_cat[jet]
            rt = true_cat[jet]
            if rp is not None and rt is not None:
                cm[row_idx[rp], col_idx[rt]] += 1

    cm_norm = cm / N

    row_perm = [0, 1, 3, 2, 4, 5]
    cm_norm = cm_norm[row_perm, :]
    row_labels = [row_labels[i] for i in row_perm]
    col_perm = [0, 1, 2, 4, 3, 5]
    cm_norm = cm_norm[:, col_perm]
    col_labels = [col_labels[i] for i in col_perm]

    # Apply LaTeX-friendly labels in permuted order
    new_row_labels = ['Higgs $b_1$', 'Higgs $b_2$', 'Had. top b','Lep. top b',  'Had. top $q_1$', 'Had. top $q_2$']
    new_col_labels = ['unmatched', 'gluons', 'Higgs b', 'Had. top b','Lep. top b',  'Had. top q']
    row_labels = [new_row_labels[i] for i in row_perm]
    col_labels = [new_col_labels[i] for i in col_perm]

    return cm_norm, row_labels, col_labels

def plot_confusion(cm, row_labels, col_labels):
    """
    Plot the normalized confusion matrix with annotations.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticklabels(row_labels)

    ax.set_xlabel('Target')
    ax.set_ylabel('Predicted')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f'{cm[i, j]:.3f}'
            ax.text(j, i, text, ha='center', va='center', color='black')

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(outputs, target, strategy = 'standard', print_count=False):
    if strategy == 'standard':
        prov_preds, order_count = assign_prov(outputs)
    else:
        prov_preds, order_count = assign_prov_higgs_first(outputs)

    cm_norm, row_labels, col_labels = compute_confusion(prov_preds, target)
    plot_confusion(cm_norm, row_labels, col_labels)
    if print_count:
        print(order_count/len(prov_preds))


'''
Example use:
#The output from spanet:
outputs
#Target array:
target = jets.prov

plot_confusion_matrix(outputs, target, strategy='higgs_first', True)

plot_confusion_matrix(outputs, target)

#If 'strategy' is left blank, the standard (max. first) assignment method is used. If anything else, the Higgs first assignment method is used.
# The bool at the end is whether or not to print the assignment counter. This shows the fraction of times each decay got assigned first, second or last. Default is False.
'''


