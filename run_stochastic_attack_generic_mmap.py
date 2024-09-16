import numpy as np
from scipy.linalg import pinv
from compute_coef_stochastic import compute_coef_stochastic
from compute_features_generic_mmap import compute_features_generic_mmap
from compute_template import compute_template
from compute_params_pca import compute_params_pca
from compute_params_lda import compute_params_lda
from compute_discriminant import compute_discriminant, compute_dlinear_fast
from get_map_base import get_map_base
from get_leakage_from_map_base import get_leakage_from_map_base
from get_signal_strength_coef import get_signal_strength_coef
from get_signal_strength_ssp import get_signal_strength_ssp
from get_selection import get_selection
from get_success_info_generic import get_success_info_generic
from prepare_data_template import prepare_data_template
from prepare_data_template_pca_v2 import prepare_data_template_pca_v2

def run_stochastic_attack_generic_mmap(
    m_data_profile, D_profile_all, idx_profile_all,
    m_data_attack, D_attack_all, idx_attack_group,
    V_profile, V_attack, V_discriminant,
    base, atype, aparams, discriminant,
    rand_iter, nr_traces_vec, eparams=None):

    num_profile_traces = len(idx_profile_all)
    results = {
        'atype': atype,
        'aparams': aparams,
        'discriminant': discriminant,
        'rand_iter': rand_iter,
        'nr_traces_vec': nr_traces_vec,
        'eparams': eparams if eparams is not None else {}
    }

    print('Running run_stochastic_attack() ...')

    if atype == 'classic':
        print('Obtaining data for stochastic coefficients...')
        N1 = num_profile_traces // 2
        idx_n1 = idx_profile_all[:N1]
        X1 = m_data_profile['data'][0]['X'][:, idx_n1].T.astype(np.float64)
        D1 = D_profile_all[idx_n1]
        N2 = num_profile_traces - N1
        idx_n2 = idx_profile_all[N1:N1 + N2]
        X2 = m_data_profile['data'][0]['X'][:, idx_n2].T.astype(np.float64)
        D2 = D_profile_all[idx_n2]

        print('Selecting basis for coefficients...')
        map_base = get_map_base(base)

        print('Computing Stochastic coefficients on raw data...')
        coef = compute_coef_stochastic(X1, D1, map_base)

        print('Computing raw covariance matrix...')
        Z2 = X2 - get_leakage_from_map_base(D2, coef, map_base)
        C = np.dot(Z2.T, Z2) / (N2 - 1)

        print('Computing signal strength estimate...')
        if aparams['signal'] in ['bnorm', 'bnorm_std']:
            signals = get_signal_strength_coef(X1, coef, base)
        else:
            smean_r = get_leakage_from_map_base(V_profile, coef, map_base)
            signals = get_signal_strength_ssp(smean_r, None, C * (N2 - 1), N2)

        if 'save_signals' in eparams and eparams['save_signals']:
            results['signals'] = signals

        spoints = get_selection(signals[aparams['signal']], aparams['sel'], aparams['p1'], aparams['p2'])
        coef = coef[:, spoints]
        results['coef'] = coef

        print('Obtaining smean/scov for attack...')
        smean = get_leakage_from_map_base(V_discriminant, coef, map_base)
        scov = C[np.ix_(spoints, spoints)]

        handle_prepare = prepare_data_template
        pp1, pp2, pp3, pp4, pp5 = spoints, None, None, None, None

    elif atype == 'selection':
        print('Obtaining data for stochastic coefficients...')
        X = m_data_profile['data'][0]['X'][:, idx_profile_all].T.astype(np.float64)
        D = D_profile_all[idx_profile_all]

        print('Selecting basis...')
        map_base = get_map_base(base)

        print('Computing Stochastic coefficients on raw data...')
        coef = compute_coef_stochastic(X, D, map_base)

        print('Computing raw covariance matrix...')
        Z = X - get_leakage_from_map_base(D, coef, map_base)
        C = np.dot(Z.T, Z) / (num_profile_traces - 1)

        print('Computing signal strength estimate...')
        if aparams['signal'] in ['bnorm', 'bnorm_std']:
            signals = get_signal_strength_coef(X, coef, base)
        else:
            smean_r = get_leakage_from_map_base(V_profile, coef, map_base)
            signals = get_signal_strength_ssp(smean_r, None, C * (num_profile_traces - 1), num_profile_traces)

        if 'save_signals' in eparams and eparams['save_signals']:
            results['signals'] = signals

        spoints = get_selection(signals[aparams['signal']], aparams['sel'], aparams['p1'], aparams['p2'])
        coef = coef[:, spoints]
        results['coef'] = coef

        print('Obtaining smean/scov for attack...')
        smean = get_leakage_from_map_base(V_discriminant, coef, map_base)
        scov = C[np.ix_(spoints, spoints)]

        handle_prepare = prepare_data_template
        pp1, pp2, pp3, pp4, pp5 = spoints, None, None, None, None

    elif atype == 'pca':
        print('Obtaining sums of squares and cross products on selected bytes...')
        M, B, W = compute_ssp_generic_mmap(aparams['m_data_subset'], aparams['D_subset_all'], V_profile, aparams['idx_traces'])
        if 'save_ssp' in eparams and eparams['save_ssp']:
            results['M'], results['B'], results['W'] = M, B, W

        xmm = M.mean(axis=0)

        print('Computing PCA parameters...')
        U, _, _, K = compute_params_pca(M, aparams['pca_threshold'])
        if 'pca_dimensions' in aparams and aparams['pca_dimensions'] > 0:
            U = U[:, :aparams['pca_dimensions']]
        else:
            U = U[:, :K]

        handle_prepare = prepare_data_template_pca_v2
        pp1, pp2, pp3, pp4, pp5 = U, xmm, None, None, None

        print('Obtaining data for stochastic coefficients...')
        L = m_data_profile['data'][0]['X'][:, idx_profile_all].T.astype(np.float64)
        X = handle_prepare(L, pp1, pp2, pp3, pp4, pp5)
        D = D_profile_all[idx_profile_all]

        print('Computing Stochastic coefficients...')
        coef = compute_coef_stochastic(X, D, map_base)
        results['coef'] = coef

        print('Aproximating mean vectors for attack...')
        smean = get_leakage_from_map_base(V_discriminant, coef, map_base)

        print('Computing covariance...')
        if aparams['cov_from_sel']:
            print('Computing data for covariance from selection...')
            x_cov = compute_features_generic_mmap(
                aparams['m_data_subset'], aparams['D_subset_all'], V_profile, aparams['idx_traces'],
                handle_prepare, pp1, pp2, pp3, pp4, pp5)
            _, C = compute_template(x_cov)
            scov = C.mean(axis=2)
        else:
            Z = X - get_leakage_from_map_base(D, coef, map_base)
            scov = np.dot(Z.T, Z) / (num_profile_traces - 1)

    elif atype == 'templatepca':
        print('Obtaining data for stochastic coefficients...')
        X = m_data_profile['data'][0]['X'][:, idx_profile_all].T.astype(np.float64)
        D = D_profile_all[idx_profile_all]

        print('Selecting basis...')
        map_base = get_map_base(base)

        print('Computing Stochastic coefficients for raw data...')
        coef = compute_coef_stochastic(X, D, map_base)

        print('Aproximating raw mean vectors from stochastic model...')
        smean_r = get_leakage_from_map_base(V_profile, coef, map_base)

        print('Computing PCA parameters...')
        U, _, xmm, K = compute_params_pca(smean_r, aparams['pca_threshold'], aparams['pca_alternate'])
        if 'pca_dimensions' in aparams and aparams['pca_dimensions'] > 0:
            U = U[:, :aparams['pca_dimensions']]
        else:
            U = U[:, :K]

        handle_prepare = prepare_data_template_pca_v2
        pp1, pp2, pp3, pp4, pp5 = U, xmm, None, None, None

        Y = handle_prepare(X, pp1, pp2, pp3, pp4, pp5)

        print('Computing Stochastic coefficients in PCA space...')
        coef = compute_coef_stochastic(Y, D, map_base)
        results['coef'] = coef

        print('Aproximating mean vectors for attack...')
        smean = get_leakage_from_map_base(V_discriminant, coef, map_base)

        print('Computing covariance in PCA space...')
        Z = Y - get_leakage_from_map_base(D, coef, map_base)
        scov = np.dot(Z.T, Z) / (num_profile_traces - 1)

    elif atype == 'lda':
        print('Obtaining sums of squares and cross products on selected bytes...')
        M, B, W = compute_ssp_generic_mmap(aparams['m_data_subset'], aparams['D_subset_all'], V_profile, aparams['idx_traces'])
        if 'save_ssp' in eparams and eparams['save_ssp']:
            results['M'], results['B'], results['W'] = M, B, W

        xmm = M.mean(axis=0)

        print('Computing Fisher\'s LDA parameters...')
        nr_values_profile = len(V_profile)
        nr_traces_per_value = len(aparams['idx_traces'])
        Spool = W / (nr_values_profile * (nr_traces_per_value - 1))
        A, _, K = compute_params_lda(B, Spool, nr_values_profile, aparams['lda_threshold'])
        if 'lda_dimensions' in aparams and aparams['lda_dimensions'] > 0:
            FW = A[:, :aparams['lda_dimensions']]
        else:
            FW = A[:, :K]

        handle_prepare = prepare_data_template_pca_v2
        pp1, pp2, pp3, pp4, pp5 = FW, xmm, None, None, None

        print('Obtaining data for stochastic coefficients...')
        L = m_data_profile['data'][0]['X'][:, idx_profile_all].T.astype(np.float64)
        X = handle_prepare(L, pp1, pp2, pp3, pp4, pp5)
        D = D_profile_all[idx_profile_all]

        print('Computing Stochastic coefficients...')
        coef = compute_coef_stochastic(X, D, map_base)
        results['coef'] = coef

        print('Aproximating mean vectors for attack...')
        smean = get_leakage_from_map_base(V_discriminant, coef, map_base)

        print('Computing covariance...')
        if aparams['cov_from_sel']:
            print('Computing data for covariance from selection...')
            x_cov = compute_features_generic_mmap(
                aparams['m_data_subset'], aparams['D_subset_all'], V_profile, aparams['idx_traces'],
                handle_prepare, pp1, pp2, pp3, pp4, pp5)
            _, C = compute_template(x_cov)
            scov = C.mean(axis=2)
        else:
            Z = X - get_leakage_from_map_base(D, coef, map_base)
            scov = np.dot(Z.T, Z) / (num_profile_traces - 1)

    elif atype == 'templatelda':
        print('Obtaining data for stochastic coefficients...')
        X = m_data_profile['data'][0]['X'][:, idx_profile_all].T.astype(np.float64)
        D = D_profile_all[idx_profile_all]

        print('Selecting basis...')
        map_base = get_map_base(base)

        print('Computing Stochastic coefficients for raw data...')
        coef = compute_coef_stochastic(X, D, map_base)

        print('Aproximating raw mean vectors from stochastic model...')
        smean_r = get_leakage_from_map_base(V_profile, coef, map_base)

        print('Computing raw covariance...')
        Z = X - get_leakage_from_map_base(D, coef, map_base)
        C = np.dot(Z.T, Z) / (num_profile_traces - 1)

        print('Computing between-groups matrix B...')
        nr_values_profile = len(V_profile)
        xmm = smean_r.mean(axis=0)
        T = smean_r - np.ones((nr_values_profile, 1)) @ xmm[np.newaxis, :]
        B = np.dot(T.T, T)

        print('Computing Fisher\'s LDA parameters...')
        A, _, K = compute_params_lda(B, C, nr_values_profile, aparams['lda_threshold'])
        if 'lda_dimensions' in aparams and aparams['lda_dimensions'] > 0:
            FW = A[:, :aparams['lda_dimensions']]
        else:
            FW = A[:, :K]

        handle_prepare = prepare_data_template_pca_v2
        pp1, pp2, pp3, pp4, pp5 = FW, xmm, None, None, None

        Y = handle_prepare(X, pp1, pp2, pp3, pp4, pp5)

        print('Computing Stochastic coefficients in LDA space...')
        coef = compute_coef_stochastic(Y, D, map_base)
        results['coef'] = coef

        print('Aproximating mean vectors for attack...')
        smean = get_leakage_from_map_base(V_discriminant, coef, map_base)

        print('Computing covariance in LDA space...')
        Z = Y - get_leakage_from_map_base(D, coef, map_base)
        scov = np.dot(Z.T, Z) / (num_profile_traces - 1)

    else:
        raise ValueError(f'Unknown atype: {atype}')

    # Store handle_prepare data
    results['handle_prepare'] = handle_prepare
    results['pp1'], results['pp2'], results['pp3'], results['pp4'], results['pp5'] = pp1, pp2, pp3, pp4, pp5

    # Load data for attack
    print('Computing attack data...')
    X_attack = compute_features_generic_mmap(
        m_data_attack, D_attack_all, V_attack, idx_attack_group,
        handle_prepare, pp1, pp2, pp3, pp4, pp5)

    if 'save_xdata' in eparams and eparams['save_xdata']:
        results['x_attack'] = X_attack

    print('Computing evaluation parameters...')
    tmiu = smean
    ic0 = np.linalg.inv(scov)

    if discriminant == 'linear':
        handle_discriminant = compute_discriminant
        pe3, pe4, pe5, pe6 = tmiu, ic0, None, None
    elif discriminant == 'linearnocov':
        handle_discriminant = compute_discriminant
        pe3, pe4, pe5, pe6 = tmiu, None, None, None
    elif discriminant == 'linearfast':
        ng_miu, m = tmiu.shape
        Y = np.zeros((ng_miu, m))
        Z = np.zeros(ng_miu)
        for k in range(ng_miu):
            Y[k, :] = tmiu[k, :] @ ic0
            Z[k] = Y[k, :] @ tmiu[k, :].T
        handle_discriminant = compute_dlinear_fast
        pe3, pe4, pe5, pe6 = Y, Z, None, None
    else:
        raise ValueError(f'Unsupported discriminant type: {discriminant}')

    if 'save_eval' in eparams and eparams['save_eval']:
        results['handle_discriminant'], results['pe3'], results['pe4'], results['pe5'], results['pe6'] = handle_discriminant, pe3, pe4, pe5, pe6

    if 'v_attack' in eparams:
        if len(eparams['v_attack']) != len(V_attack):
            raise ValueError('eparams.v_attack given but has incompatible length with V_attack')
        V_attack = eparams['v_attack']

    print('Computing success info...')
    results['success_info'] = get_success_info_generic(
        X_attack, V_attack, V_discriminant,
        rand_iter,
        nr_traces_vec,
        handle_discriminant, pe3, pe4, pe5, pe6)

    return results
