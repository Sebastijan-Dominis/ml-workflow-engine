from ml.utils.hashing.hash_list import hash_list

def compute_cols_for_row_id_fingerprint(cols):
    return hash_list(cols, order_matters=False)