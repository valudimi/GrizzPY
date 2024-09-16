from compute_ssp_e2_mmap_multi import compute_ssp_e2_mmap_multi
from get_signal_strength_ssp import get_signal_strength_ssp
from compute_params_lda import compute_params_lda
from prepare_data_template_pca_v2 import prepare_data_template_pca_v2
from compute_features_e2_mmap_multi import compute_features_e2_mmap_multi
from compute_features_e2_mmap import compute_features_e2_mmap
from compute_features_e2_mmap_adapt import compute_features_e2_mmap_adapt
from compute_template import compute_template
from evaluate_discriminant import evaluate_discriminant
from get_success_info_like import get_success_info_like
from prepare_data_template import prepare_data_template
from get_selection import get_selection
from compute_params_pca import compute_params_pca
import numpy as np

def run_template_attack_adapt(s_profile, s_helper, m_data_attack, metadata_attack, idx_attack,
                              inbytes, atype, cmethod, cparams, discriminant,
                              rand_iter, nr_traces_vec, eparams=None):
    results = {}

    nr_groups = len(inbytes)
    results['s_profile'] = s_profile
    results['s_helper'] = s_helper
    results['metadata_attack'] = metadata_attack
    results['idx_attack'] = idx_attack
    results['bytes'] = inbytes
    results['atype'] = atype
    results['cmethod'] = cmethod
    results['cparams'] = cparams
    results['discriminant'] = discriminant
    results['rand_iter'] = rand_iter
    results['nr_traces_vec'] = nr_traces_vec

    if eparams is None or not eparams:
        eparams = {}

    results['eparams'] = eparams
    print('Running run_template_attack_adapt() ...')

    # Get the sums of squares and cross products
    print('Obtaining sums of squares and cross products for all sets ...')
    if atype in ['roffset', 'roffset_median']:
        M, B, W, np_total = compute_ssp_e2_mmap_multi(s_profile, inbytes, None, eparams['roffset'])
    elif atype == 'boffset':
        M, _, W, np_total = compute_ssp_e2_mmap_multi(s_profile, inbytes)

        m, n = M.shape
        xm = np.mean(M, axis=0).reshape(1, -1)
        # print(f'xm size = {xm.size}\nxm = {xm}')
        R = np.mean(xm) * np.random.rand(m, 1) @ np.ones((1, n))
        # print(f'R shape = {R.shape}\nR = {R}')
        M = M + R
        # print(f'M shape = {M.shape}\nM = {M}')
        xm2 = np.mean(M, axis=0).reshape(1, -1)
        # print(f'xm2 size = {xm2.size}\nxm2 = {xm2}')
        XB = M - np.dot(np.ones((m, 1)), xm2)
        # XB = M - np.ones((m, 1)) @ xm2
        # print(f'XB shape = {XB.shape}\nXB = {XB}')
        B = XB.T @ XB
        # print(f'B shape = {B.shape}\nB = {B}')
    else:
        M, B, W, np_total = compute_ssp_e2_mmap_multi(s_profile, inbytes)

    print(f'M = {M}\nB = {B}\nW = {W}\nnp_total = {np_total}')
    xmm = np.mean(M, axis=0)
    results['M'] = M
    results['B'] = B
    results['W'] = W

    # Estimate correlation via factor analysis if 'famvn' specified
    if atype == 'famvn':
        if 'nr_factors' not in eparams:
            raise ValueError('Need eparams.nr_factors for famvn')
        nr_factors = eparams['nr_factors']
        C = W / (nr_groups * (np_total - 1))
        dvec = np.sqrt(np.diag(C))
        Dinv = np.diag(1.0 / dvec)
        R = Dinv @ C @ Dinv
        U, S, _ = np.linalg.svd(R)
        d = np.diag(S)
        L = U[:, :nr_factors] @ np.diag(np.sqrt(d[:nr_factors]))
        results['L'] = L
        P = np.diag(R - L @ L.T)
        results['P'] = P
        RE = L @ L.T + np.diag(P)
        CE = np.diag(dvec) @ RE @ np.diag(dvec)
        W = CE * (nr_groups * (np_total - 1))

    # Get compression parameters
    if cmethod == 'sample':
        print('Computing selection curves and selected samples ...')
        curves = get_signal_strength_ssp(M, B, W, np_total)
        interest_points = get_selection(curves[cparams['curve']], cparams['sel'], cparams['p1'], cparams.get('p2', None))
        handle_prepare = prepare_data_template
        pp1 = interest_points
        pp2 = xmm
        pp3 = None
        pp4 = None
        pp5 = None
    elif cmethod == 'PCA':
        print('Computing PCA parameters...')
        U, D, xmm, K = compute_params_pca(M, cparams['pca_threshold'], cparams['pca_alternate'])
        if eparams.get('use_elv', False):
            params = {'method': 'maximal', 'max_elvs': round(U.shape[0] / 100)}
            idx = get_elv_order(U, D, params) # TODO: what is get_elv_order?
            U = U[:, idx]
        if cparams.get('pca_dimensions', 0) > 0:
            U = U[:, :cparams['pca_dimensions']]
        else:
            U = U[:, :K]

        handle_prepare = prepare_data_template_pca_v2
        pp1 = U
        pp2 = xmm
        pp3 = None
        pp4 = None
        pp5 = None
    elif cmethod == 'LDA':
        print('Computing Fisher\'s LDA parameters...')
        Spool = W / (nr_groups * (np_total - 1))
        A, D, K = compute_params_lda(B, Spool, nr_groups, cparams['lda_threshold'])
        if eparams.get('use_elv', False):
            params = {'method': 'maximal', 'max_elvs': round(A.shape[0] / 100)}
            idx = get_elv_order(A, D, params)
            A = A[:, idx]
        if cparams.get('lda_dimensions', 0) > 0:
            FW = A[:, :cparams['lda_dimensions']]
        else:
            FW = A[:, :K]

        handle_prepare = prepare_data_template_pca_v2
        pp1 = FW
        pp2 = xmm
        pp3 = None
        pp4 = None
        pp5 = None
    else:
        raise ValueError(f'Unknown compression method: {cmethod}')

    # Store handle_prepare data
    results['handle_prepare'] = handle_prepare
    results['pp1'] = pp1
    results['pp2'] = pp2
    results['pp3'] = pp3
    results['pp4'] = pp4
    results['pp5'] = pp5

    # Load raw leakage data for profile
    print('Computing profiling data from all sets...')
    if atype in ['roffset', 'roffset_median']:
        x_profile = compute_features_e2_mmap_multi(s_profile, handle_prepare, pp1, pp2, pp3, pp4, pp5, inbytes, eparams['roffset'])
    else:
        # TODO check if this is correct
        x_profile = compute_features_e2_mmap_multi(s_profile, handle_prepare, pp1, pp2, pp3, pp4, pp5, inbytes)
    if eparams.get('save_xdata', 0) != 0:
        results['x_profile'] = x_profile

    # Load raw leakage data for attack
    print('Computing attack data...')
    if atype in ['mvn_offset_median', 'multi_offset_median', 'roffset_median']:
        s_adapt = {'type': 'offset_median', 'xmm': xmm}
        x_attack = compute_features_e2_mmap_adapt(m_data_attack, metadata_attack, idx_attack, s_adapt, handle_prepare, pp1, pp2, pp3, pp4, pp5, inbytes)
    else:
        x_attack = compute_features_e2_mmap(m_data_attack, metadata_attack, idx_attack, handle_prepare, pp1, pp2, pp3, pp4, pp5, inbytes)
    if eparams.get('save_xdata', 0) != 0:
        results['x_attack'] = x_attack

    # Compute templates
    if atype in ['mvn', 'mvn_offset_median', 'multi', 'multi_offset_median', 'roffset', 'roffset_median', 'boffset']:
        print('Computing mvn template and evaluation parameters...')
        tmiu, tsigma = compute_template(x_profile)

        handle_eval = evaluate_discriminant
        if discriminant == 'linear':
            c0 = np.mean(tsigma, axis=2)
            ic0 = np.linalg.inv(c0)
            pe3 = tmiu
            pe4 = ic0
            pe5 = None
            pe6 = None
        elif discriminant == 'linearnocov':
            pe3 = tmiu
            pe4 = None
            pe5 = None
            pe6 = None
        elif discriminant == 'log':
            n = tsigma.shape[2]
            tsinv = np.zeros_like(tsigma)
            tlogdet = np.zeros(n)
            for k in range(n):
                tsinv[:, :, k] = np.linalg.inv(tsigma[:, :, k])
                tlogdet[k] = np.linalg.slogdet(tsigma[:, :, k])[1]
            pe3 = tmiu
            pe4 = tsinv
            pe5 = tlogdet
            pe6 = None
        else:
            raise ValueError(f'discriminant not supported for mvn: {discriminant}')
    elif atype == 'famvn':
        print('Computing mvn template and evaluation parameters...')
        tmiu, tsigma = compute_template(x_profile)

        handle_eval = evaluate_discriminant
        if discriminant == 'linear':
            c0 = np.mean(tsigma, axis=2)
            U, S, _ = np.linalg.svd(c0)
            d = np.diag(S)
            L = U[:, :nr_factors] @ np.diag(np.sqrt(d[:nr_factors]))
            P = c0 - L @ L.T
            P = np.diag(P)
            c0_f = L @ L.T + np.diag(P)
            ic0 = np.linalg.inv(c0_f)
            pe3 = tmiu
            pe4 = ic0
            pe5 = None
            pe6 = None
        elif discriminant == 'linearnocov':
            pe3 = tmiu
            pe4 = None
            pe5 = None
            pe6 = None
        elif discriminant == 'log':
            n = tsigma.shape[2]
            tsinv = np.zeros_like(tsigma)
            tlogdet = np.zeros(n)
            for k in range(n):
                U, S, _ = np.linalg.svd(tsigma[:, :, k])
                d = np.diag(S)
                L = U[:, :nr_factors] @ np.diag(np.sqrt(d[:nr_factors]))
                P = tsigma[:, :, k] - L @ L.T
                P = np.diag(P)
                sk_f = L @ L.T + np.diag(P)
                tsinv[:, :, k] = np.linalg.inv(sk_f)
                tlogdet[k] = np.linalg.slogdet(sk_f)[1]
            pe3 = tmiu
            pe4 = tsinv
            pe5 = tlogdet
            pe6 = None
        else:
            raise ValueError(f'discriminant not supported for famvn: {discriminant}')
    else:
        raise ValueError(f'template attack type not supported: {atype}')

    # Store evaluation data if requested
    if eparams.get('save_eval', 0) != 0:
        results['handle_eval'] = handle_eval
        results['pe3'] = pe3
        results['pe4'] = pe4
        results['pe5'] = pe5
        results['pe6'] = pe6

    # Compute the success information
    print('Computing success info...')
    results['success_info'] = get_success_info_like(x_attack, rand_iter, nr_traces_vec, handle_eval, pe3, pe4, pe5, pe6)

    return results
