def getshape_instead_of_getdim(getdim_names):
    getshape_names = [f"getshape_{getdim_name[len('getdim_'):]}" for getdim_name in getdim_names]
    return f"implement 'getshape' instead of 'getdim' (expected {getshape_names} but found {getdim_names})"

def too_little_samples_for_class(class_idx, actual, expected):
    return f"class {class_idx} has only {actual} samples (requires at least {expected} samples)"


KD_MIX_WRAPPER_REQUIRES_SEED_OR_CONTEXT = (
    "KDMixWrapper requires either a context or a seed to ensure same mixing of samples and labels "
    "(NOTE: using a seed will additionally make the transform deterministc per sample)"
)

REQUIRES_MIXUP_P_OR_CUTMIX_P = "at least one of mixup_p or cutmix_p is required"
