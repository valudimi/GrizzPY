def get_uid_cmethod(cmethod, cparams):
    """
    Returns a Unique ID (UID) for a compression method/params combination.
    
    Args:
    - cmethod (str): Compression method.
    - cparams (dict): Parameters for the compression method.
    
    Returns:
    - int: UID corresponding to the cmethod/cparams combination.
    """
    uid = 0

    if cmethod == 'LDA':
        if cparams['lda_dimensions'] == 4:
            uid = 0
        elif cparams['lda_dimensions'] == 40:
            uid = 6
        elif cparams['lda_dimensions'] == 100 or cparams['lda_dimensions'] == 5:
            uid = 7
        elif cparams['lda_dimensions'] == 6:
            uid = 10
        elif cparams['lda_dimensions'] == 3:
            uid = 12

    elif cmethod == 'PCA':
        if cparams['pca_dimensions'] == 4:
            uid = 1
        elif cparams['pca_dimensions'] == 40:
            uid = 8
        elif cparams['pca_dimensions'] == 100 or cparams['pca_dimensions'] == 5:
            uid = 9
        elif cparams['pca_dimensions'] == 6:
            uid = 11

    elif cmethod == 'sample':
        if cparams['sel'] == '1ppc':
            uid = 2
        elif cparams['sel'] == '3ppc':
            uid = 3
        elif cparams['sel'] == '20ppc':
            uid = 4
        elif cparams['sel'] == 'allap':
            uid = 5

    return uid
