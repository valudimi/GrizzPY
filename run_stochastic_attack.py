"""
TODO
"""

from compute_ssp_e2_mmap import compute_ssp_e2_mmap
from get_signal_strength_ssp import get_signal_strength_ssp
from compute_params_lda import compute_params_lda
from prepare_data_template_pca_v2 import prepare_data_template_pca_v2
from compute_features_e2_mmap import compute_features_e2_mmap
from compute_template import compute_template
from evaluate_discriminant import evaluate_discriminant
from get_success_info_like import get_success_info_like
from get_leakage_from_map_base import get_leakage_from_map_base
from compute_coef_stochastic import compute_coef_stochastic
from get_signal_strength_coef import get_signal_strength_coef
from prepare_data_template import prepare_data_template
from get_map_base import get_map_base
from get_selection import get_selection
from compute_params_pca import compute_params_pca
import numpy as np

def run_stochastic_attack(m_data_profile, metadata_profile, idx_profile, base,
                          m_data_attack, metadata_attack, idx_attack, atype,
                          aparams, discriminant, rand_iter, nr_traces_vec, eparams=None):
    """
    TODO
    """
    results = {}

    nr_groups = metadata_profile['nr_groups']
    num_profile_traces = len(idx_profile)
    results['metadata_profile'] = metadata_profile
    results['idx_profile'] = idx_profile
    results['metadata_attack'] = metadata_attack
    results['idx_attack'] = idx_attack
    results['atype'] = atype
    results['aparams'] = aparams
    results['discriminant'] = discriminant
    results['rand_iter'] = rand_iter
    results['nr_traces_vec'] = nr_traces_vec
    results['eparams'] = eparams if eparams else {}

    print('Running run_stochastic_attack()...')

    if atype == 'classic':
        # Obtain data for computing the coefficients
        N1 = num_profile_traces // 2
        idx_n1 = idx_profile[:N1]
        # X1 = m_data_profile['data'][0]['X'][:, idx_n1].T.astype(np.float64)
        X1 = m_data_profile['X'][:, idx_n1].T.astype(np.float64)
        D1 = m_data_profile['B'][1, idx_n1].T.astype(np.float64)
        N2 = num_profile_traces - N1
        idx_n2 = idx_profile[N1:]
        X2 = m_data_profile['X'][:, idx_n2].T.astype(np.float64)
        D2 = m_data_profile['B'][1, idx_n2].T.astype(np.float64)

        # Select basis for coefficients # TODO check
        map_base = get_map_base(base)

        # Compute Stochastic coefficients # syntactically checked
        print('Computing Stochastic coefficients...')
        coef = compute_coef_stochastic(X1, D1, map_base)
        print(coef)

        # Approximate mean vectors from stochastic model
        data = np.arange(metadata_profile['nr_groups'])
        smean_r = get_leakage_from_map_base(data, coef, map_base)

        # Compute covariance
        Z2 = X2 - get_leakage_from_map_base(D2, coef, map_base)
        C = (Z2.T @ Z2) / (N2 - 1)

        # Compute signal strengths
        if aparams['signal'] == 'bnorm':
            signals = get_signal_strength_coef(X1, coef, base)
        else:
            signals = get_signal_strength_ssp(smean_r, None, C * (nr_groups * (N2 - 1)), N2)

        # Select points (samples)
        spoints = get_selection(signals[aparams['signal']], aparams['sel'], aparams['p1'], aparams.get('p2'))

        # Restrict coefficients only to selected points
        coef = coef[:, spoints]
        results['coef'] = coef

        # Obtain mean and covariance on selected points
        smean = smean_r[:, spoints]
        scov = C[spoints, :][:, spoints]

        # Set compression/selection parameters
        handle_prepare = prepare_data_template
        pp1, pp2, pp3, pp4, pp5 = spoints, None, None, None, None

    elif atype == 'same_profile':
        # Obtain data for computing the coefficients
        X = m_data_profile['X'][:, idx_profile].T.astype(np.float64)
        D = m_data_profile['B'][1, idx_profile].T.astype(np.float64)

        # Select basis
        map_base = get_map_base(base)

        # Compute Stochastic coefficients
        print('Computing Stochastic coefficients...')
        coef = compute_coef_stochastic(X, D, map_base)

        # Approximate mean vectors from stochastic model
        data = np.arange(metadata_profile['nr_groups'])
        smean_r = get_leakage_from_map_base(data, coef, map_base)

        # Compute covariance
        Z = X - get_leakage_from_map_base(D, coef, map_base)
        C = (Z.T @ Z) / (num_profile_traces - 1)

        # Compute signal strengths
        if aparams['signal'] in ['bnorm', 'bnorm_std']:
            signals = get_signal_strength_coef(X, coef, base)
        else:
            signals = get_signal_strength_ssp(smean_r, None, C * (nr_groups * (num_profile_traces - 1)), num_profile_traces)

        # Select points (samples)
        spoints = get_selection(signals[aparams['signal']], aparams['sel'], aparams['p1'], aparams.get('p2'))

        # Restrict coefficients only to selected points
        coef = coef[:, spoints]
        results['coef'] = coef

        # Obtain mean and covariance on selected points
        smean = smean_r[:, spoints]
        scov = C[spoints, :][:, spoints]

        # Set compression/selection parameters
        handle_prepare = prepare_data_template
        pp1, pp2, pp3, pp4, pp5 = spoints, None, None, None, None

    elif atype == 'pca':
        print('Obtaining sums of squares and cross products on selected bytes...')
        M, B, W = compute_ssp_e2_mmap(m_data_profile, metadata_profile, aparams['idx_traces'], aparams['byte_sel'])
        if 'save_ssp' in results['eparams'] and results['eparams']['save_ssp']:
            results['M'], results['B'], results['W'] = M, B, W

        xmm = np.mean(M, axis=0)

        print('Computing PCA parameters...')
        U, _, _, K = compute_params_pca(M, aparams['pca_threshold'])
        if aparams['pca_dimensions'] > 0:
            U = U[:, :aparams['pca_dimensions']]
        else:
            U = U[:, :K]

        handle_prepare = prepare_data_template_pca_v2
        pp1, pp2, pp3, pp4, pp5 = U, xmm, None, None, None

        # Obtain data for computing the coefficients
        L = m_data_profile['X'][:, idx_profile].T.astype(np.float64)
        X = handle_prepare(L, pp1, pp2, pp3, pp4, pp5)
        D = m_data_profile['B'][1, idx_profile].T.astype(np.float64)

        map_base = get_map_base(base)

        print('Computing Stochastic coefficients...')
        coef = compute_coef_stochastic(X, D, map_base)
        results['coef'] = coef

        smean = get_leakage_from_map_base(np.arange(metadata_profile['nr_groups']), coef, map_base)

        print('Computing covariance...')
        if aparams.get('cov_from_sel', False):
            print('Computing data for covariance from selection...')
            x_cov = compute_features_e2_mmap(m_data_profile, metadata_profile, aparams['idx_traces'],
                                             handle_prepare, pp1, pp2, pp3, pp4, pp5, aparams['byte_sel'])
            _, C = compute_template(x_cov)
            scov = np.mean(C, axis=2)
        else:
            Z = X - get_leakage_from_map_base(D, coef, map_base)
            scov = (Z.T @ Z) / (num_profile_traces - 1)

    elif atype == 'templatepca':
        # Obtain data for computing the coefficients
        X = m_data_profile['X'][:, idx_profile].T.astype(np.float64)
        D = m_data_profile['B'][1, idx_profile].T.astype(np.float64)

        # Select basis
        map_base = get_map_base(base)

        # Compute Stochastic coefficients
        print('Computing Stochastic coefficients...')
        coef = compute_coef_stochastic(X, D, map_base)

        # Approximate mean vectors from stochastic model
        data = np.arange(metadata_profile['nr_groups'])
        smean_r = get_leakage_from_map_base(data, coef, map_base)

        # Compute PCA params
        print('Computing PCA parameters...')
        U, _, xmm, K = compute_params_pca(smean_r, aparams['pca_threshold'], aparams['pca_alternate'])
        if 'pca_dimensions' in aparams and aparams['pca_dimensions'] > 0:
            U = U[:, :aparams['pca_dimensions']]
        else:
            U = U[:, :K]

        # Set compression/selection parameters
        handle_prepare = prepare_data_template_pca_v2
        pp1, pp2, pp3, pp4, pp5 = U, xmm, None, None, None

        # Project data using PCA
        Y = handle_prepare(X, pp1, pp2, pp3, pp4, pp5)

        # Compute Stochastic coefficients in PCA space
        print('Computing Stochastic coefficients...')
        coef = compute_coef_stochastic(Y, D, map_base)
        results['coef'] = coef

        # Approximate mean vectors in PCA space
        smean = get_leakage_from_map_base(data, coef, map_base)

        # Compute covariance in PCA space
        print('Computing covariance...')
        Z = Y - get_leakage_from_map_base(D, coef, map_base)
        scov = (Z.T @ Z) / (num_profile_traces - 1)

    elif atype == 'badpca':
        # Obtain data for computing the coefficients
        X = m_data_profile['X'][:, idx_profile].T.astype(np.float64)
        D = m_data_profile['B'][1, idx_profile].T.astype(np.float64)

        # Select basis
        map_base = get_map_base(base)

        # Compute Stochastic coefficients
        print('Computing Stochastic coefficients...')
        coef = compute_coef_stochastic(X, D, map_base)

        # Approximate mean vectors from stochastic model
        data = np.arange(metadata_profile['nr_groups'])
        smean_r = get_leakage_from_map_base(data, coef, map_base)
        xmm = np.mean(smean_r, axis=0)

        # Compute full covariance estimate
        print('Computing full covariance...')
        Z = X - get_leakage_from_map_base(D, coef, map_base)
        C = (Z.T @ Z) / (num_profile_traces - 1)

        # Compute "bad" PCA params
        print('Computing bad PCA parameters...')
        U, _, _ = np.linalg.svd(C)
        if 'pca_dimensions' in aparams and aparams['pca_dimensions'] > 0:
            U = U[:, :aparams['pca_dimensions']]
        else:
            U = U[:, :1]

        # Set compression/selection parameters
        handle_prepare = prepare_data_template_pca_v2
        pp1, pp2, pp3, pp4, pp5 = U, xmm, None, None, None

        # Project data using PCA
        Y = handle_prepare(X, pp1, pp2, pp3, pp4, pp5)

        # Compute Stochastic coefficients in PCA space
        print('Computing Stochastic coefficients...')
        coef = compute_coef_stochastic(Y, D, map_base)
        results['coef'] = coef

        # Approximate mean vectors in PCA space
        smean = get_leakage_from_map_base(data, coef, map_base)

        # Compute covariance in PCA space
        print('Computing covariance...')
        Z = Y - get_leakage_from_map_base(D, coef, map_base)
        scov = (Z.T @ Z) / (num_profile_traces - 1)

    elif atype == 'lda':
        print('Obtaining sums of squares and cross products on selected bytes...')
        M, B, W = compute_ssp_e2_mmap(m_data_profile, metadata_profile, aparams['idx_traces'], aparams['byte_sel'])
        if 'save_ssp' in eparams and eparams['save_ssp'] != 0:
            results['M'], results['B'], results['W'] = M, B, W
        xmm = np.mean(M, axis=0)

        # Compute LDA params
        print('Computing Fishers LDA parameters...')
        len_byte_sel = len(aparams['byte_sel'])
        nr_traces_per_byte = len(aparams['idx_traces'])
        Spool = W / (len_byte_sel * (nr_traces_per_byte - 1))
        A, _, K = compute_params_lda(B, Spool, len_byte_sel, aparams['lda_threshold'])
        if 'lda_dimensions' in aparams and aparams['lda_dimensions'] > 0:
            FW = A[:, :aparams['lda_dimensions']]
        else:
            FW = A[:, :K]

        # Set compression/selection parameters
        handle_prepare = prepare_data_template_pca_v2
        pp1, pp2, pp3, pp4, pp5 = FW, xmm, None, None, None

        # Obtain data for computing the coefficients
        L = m_data_profile['X'][:, idx_profile].T.astype(np.float64)
        X = handle_prepare(L, pp1, pp2, pp3, pp4, pp5)
        D = m_data_profile['B'][1, idx_profile].T.astype(np.float64)

        # Select basis
        map_base = get_map_base(base)

        # Compute Stochastic coefficients
        print('Computing Stochastic coefficients...')
        coef = compute_coef_stochastic(X, D, map_base)
        results['coef'] = coef

        # Approximate mean vectors from stochastic model
        smean = get_leakage_from_map_base(data, coef, map_base)

        # Compute covariance
        print('Computing covariance...')
        if aparams.get('cov_from_sel', False):
            print('Computing data for covariance from selection...')
            x_cov = compute_features_e2_mmap(m_data_profile, metadata_profile, aparams['idx_traces'],
                                             handle_prepare, pp1, pp2, pp3, pp4, pp5, aparams['byte_sel'])
            _, C = compute_template(x_cov)
            scov = np.mean(C, axis=2)
        else:
            Z = X - get_leakage_from_map_base(D, coef, map_base)
            scov = (Z.T @ Z) / (num_profile_traces - 1)

    elif atype == 'templatelda':
        # Obtain data for computing the coefficients
        # print(m_data_profile.keys())
        X = m_data_profile['X'][:, idx_profile].T.astype(np.float64)
        D = m_data_profile['B'][1, idx_profile].T.astype(np.float64)

        # Select basis
        map_base = get_map_base(base)

        # Compute Stochastic coefficients
        print('Computing Stochastic coefficients...')
        coef = compute_coef_stochastic(X, D, map_base)

        # Approximate raw mean vectors from stochastic model
        data = np.arange(metadata_profile['nr_groups'])
        smean_r = get_leakage_from_map_base(data, coef, map_base)

        # Compute raw covariance matrix
        print('Computing raw covariance...')
        Z = X - get_leakage_from_map_base(D, coef, map_base)
        C = (Z.T @ Z) / (num_profile_traces - 1)

        # Compute raw between-groups matrix B
        print('Computing between-groups matrix B...')
        xmm = np.mean(smean_r, axis=0)
        T = smean_r - np.ones((nr_groups, 1)) @ xmm[np.newaxis, :]
        B = T.T @ T

        # Compute LDA params
        print('Computing Fishers LDA parameters...')
        A, _, K = compute_params_lda(B, C, nr_groups, aparams['lda_threshold'])
        if 'lda_dimensions' in aparams and aparams['lda_dimensions'] > 0:
            FW = A[:, :aparams['lda_dimensions']]
        else:
            FW = A[:, :K]

        # Set compression/selection parameters
        handle_prepare = prepare_data_template_pca_v2
        pp1, pp2, pp3, pp4, pp5 = FW, xmm, None, None, None

        # Project data using LDA
        Y = handle_prepare(X, pp1, pp2, pp3, pp4, pp5)

        # Compute Stochastic coefficients in LDA space
        print('Computing Stochastic coefficients...')
        coef = compute_coef_stochastic(Y, D, map_base)
        results['coef'] = coef

        # Approximate mean vectors in LDA space
        smean = get_leakage_from_map_base(data, coef, map_base)

        # Compute covariance in LDA space
        print('Computing covariance...')
        Z = Y - get_leakage_from_map_base(D, coef, map_base)
        scov = (Z.T @ Z) / (num_profile_traces - 1)

    else:
        raise ValueError(f'Unknown atype: {atype}')

    # Store handle_prepare data
    results['handle_prepare'] = handle_prepare
    results['pp1'], results['pp2'], results['pp3'], results['pp4'], results['pp5'] = pp1, pp2, pp3, pp4, pp5

    # Load data for attack
    print('Computing attack data...')
    x_attack = compute_features_e2_mmap(m_data_attack, metadata_attack, idx_attack,
                                        handle_prepare, pp1, pp2, pp3, pp4, pp5)
    if 'save_xdata' in results['eparams'] and results['eparams']['save_xdata']:
        results['x_attack'] = x_attack

    # Set evaluation parameters
    tmiu = smean
    ic0 = np.linalg.inv(scov)
    handle_eval = evaluate_discriminant
    if discriminant == 'linear':
        pe3, pe4, pe5, pe6 = tmiu, ic0, None, None
    elif discriminant == 'linearnocov':
        pe3, pe4, pe5, pe6 = tmiu, None, None, None
    else:
        raise ValueError(f'Unsupported discriminant type: {discriminant}')

    if 'save_eval' in results['eparams'] and results['eparams']['save_eval']:
        results['handle_eval'], results['pe3'], results['pe4'], results['pe5'], results['pe6'] = handle_eval, pe3, pe4, pe5, pe6

    # Compute the success information
    print('Computing success info...')
    results['success_info'] = get_success_info_like(x_attack, rand_iter, nr_traces_vec,
                                                    handle_eval, pe3, pe4, pe5, pe6)

    return results
