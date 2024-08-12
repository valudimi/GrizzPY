from compute_ssp_e2_mmap import compute_ssp_e2_mmap
from get_signal_strength_ssp import get_signal_strength_ssp
from compute_params_lda import compute_params_lda
from prepare_data_template_pca_v2 import prepare_data_template_pca_v2
from compute_features_e2_mmap import compute_features_e2_mmap
from compute_template import compute_template
from evaluate_discriminant import evaluate_discriminant

"""
runs a template attack with the given parameters and returns a results
structure that is defined below. This method is intended for any memory
mapped data that has a similar structure as the data from the E2
experiment. See get_mmap for more details.

m_data_profile and metadata_profile should be the memory mapped object
and associated metadata info for the profiling data. Use get_mmap 
on the selected data to obtain these objects.

idx_profile should be a vector of indices specifying which traces from
each group should be used for the profile data.

m_data_attack and metadata_attack should be the memory mapped object
and associated metadata info for the attack data. Use get_mmap 
on the selected data to obtain these objects. You can pass the same
objects for profiling and attack, which should make the attack faster.

idx_attack should be a vector of indices specifying which traces from
each group should be used for the attack data.

inbytes should be a vector of indices specifying which bytes (starting
from 0) will be used for the attack. This might be useful in order to
restrict the attack only to the bytes 0-15 for example (i.e. using 4
bits).

atype should be a string specifying the type of template attack to be
used. Currently supported are:
- 'mvn': which relies on the multivariate normal probability density
function to compute templates and probabilities.

cmethod should be a string specifying the compression method. Currently
supported methods are: 'sample', 'PCA' and 'LDA'.

cparams should be a structure of params specific to the compression
method. For each compression method the params are as follows:
- 'sample':
 -> cparams.curve is a string specifying the signal strength curve to
 be used.
 -> cparams.sel is a string specifying the class of selection.
 -> cparams.p1 is a parameter for the class of selection.
- 'PCA':
 -> cparams.pca_threshold
 -> cparams.pca_alternate
 -> cparams.pca_dimensions
- 'LDA':
 -> cparams.lda_dimensions

discriminant should be a string specifying the type of discriminant to
be used. The possible options are:
- 'linear': uses a pooled common covariance matrix with a linear
discriminant.
- 'linearnocov': does not use a covariance matrix. Might be useful in
particular with LDA, where the covariance should be the
identity if the eigenvectors are chosen carefully.
- 'log': uses individual covariances and log-determinants to compute
the group specific templates (mean and covariance).

rand_iter should be a positive integer specifying the number of
iterations to run the evaluation (guessing_entropy) computation. The
returned results may contain either the individual or the average
results. Check below for details.

nr_traces_vec is a vector containing the number of attack traces to be
used for each element.

The 'results' structure contains the following:
-> results.metadata_profile: the metadata structure for profile.
-> results.idx_profile: the idx_profile vector.
-> results.metadata_attack: the metadata structure for attack.
-> results.idx_attack: the idx_attack vector.
-> results.bytes: the bytes (or inbytes) vector.
-> results.atype: the atype string.
-> results.cmethod: the compression method string.
-> results.cparams: the cparams structure.
-> results.discriminant: the discriminant string.
-> results.rand_iter: the number of iterations.
-> results.nr_traces_vec: the vector with number of attack traces.
-> results.M: the matrix of group means.
-> results.B: the between-groups matrix.
-> results.W: the matrix of variances and covariances across all data.
-> results.x_profile: the profiling data, after compression.
-> results.x_attack: the attack data, after compression.
-> results.success_info: guessing entropy information, as returned by
the get_success_info_like method.

See the paper "Efficient Template Attacks" by Omar Choudary and Markus
Kuhn, presented at CARDIS 2013.
"""

def run_template_attack(m_data_profile, metadata_profile, idx_profile, m_data_attack, 
                        metadata_attack, idx_attack, inbytes, atype, cmethod, cparams,
                        discriminant, rand_iter, nr_traces_vec):
    np = len(idx_profile)
    nr_groups = len(inbytes)
    results = {}
    results['metadata_profile'] = metadata_profile
    results['idx_profile'] = idx_profile
    results['metadata_attack'] = metadata_attack
    results['idx_attack'] = idx_attack
    results['bytes'] = inbytes
    results['atype'] = atype
    results['cmethod'] = cmethod
    results['cparams'] = cparams
    results['discriminant'] = discriminant
    results['rand_iter'] = rand_iter
    results['nr_traces_vec'] = nr_traces_vec

    print('Obtaining sums of squares and cross products...\n');
    results['M'], results['B'], results['W'] = compute_ssp_e2_mmap(m_data_profile,
                                                metadata_profile, idx_profile, inbytes)
    xmm = np.mean(results['M'], axis=0)

    if cmethod == 'sample':
        # [curves] = get_signal_strength_ssp(M, B, W, np);
        
        # interest_points = get_selection(curves.(cparams.curve), cparams.sel, cparams.p1);
        
        # handle_prepare = @prepare_data_template;
        # pp1 = interest_points;
        # pp2 = [];
        # pp3 = [];
        # pp4 = [];
        # pp5 = [];
        print("Don't use this lol") # TODO implement sample method
        print('Computing selection curves and selected samples...\n')
        curves = get_signal_strength_ssp(results['M'], results['B'], results['W'], np)

    elif cmethod == 'LDA':
        print('Computing Fisher\'s LDA parameters...')
        Spool = results['W'] / (nr_groups * (np - 1))
        A, D = compute_params_lda(results['B'], Spool)
        FW = A[:, :cparams['lda_dimensions']]

        handle_prepare = prepare_data_template_pca_v2
        pp1 = FW
        pp2 = xmm  # Ensure xmm is defined in your context
        pp3 = []
        pp4 = []
        pp5 = []
    else:
        print(f'Unknown compression method {cmethod}')
        return
    
    # Load raw leakage data for profile
    print('Computing profiling data...\n')
    _, x_profile = compute_features_e2_mmap(m_data_profile, metadata_profile,
                        idx_profile, handle_prepare, pp1, pp2, pp3, pp4, pp5, inbytes)
    results['x_profile'] = x_profile

    # Load raw leakage data for attack
    print('Computing attack data...\n')
    x_attack = handle_prepare(m_data_attack, pp1, pp2, pp3, pp4, pp5)
    _, x_attack = compute_features_e2_mmap(m_data_attack, metadata_attack,
                        idx_attack, handle_prepare, pp1, pp2, pp3, pp4, pp5, inbytes)
    results['x_attack'] = x_attack

    # Compute templates
    if atype == 'mvn':
        print('Computing mvn template and evaluation parameters...\n')
        tmiu, tsigma = compute_template(x_profile)
        
        handle_eval = evaluate_discriminant
        if discriminant == 'linear':
            c0 = np.mean(tsigma, axis=2)
            ic0 = np.linalg.inv(c0)
            pe3 = tmiu
            pe4 = ic0
            pe5 = []
            pe6 = []
        elif discriminant == 'linearnocov':
            pe3 = tmiu
            pe4 = []
            pe5 = []
            pe6 = []
        elif discriminant == 'log':
            n = tsigma.shape[2]
            tsinv = np.zeros_like(tsigma)
            tlogdet = np.zeros(n)
            for k in range(n):
                tsinv[:, :, k] = np.linalg.inv(tsigma[:, :, k])
                tlogdet[k] = np.linalg.slogdet(tsigma[:, :, k])[1]  # Using slogdet for numerical stability
            pe3 = tmiu
            pe4 = tsinv
            pe5 = tlogdet
            pe6 = []
        else:
            raise ValueError(f'discriminant not supported: {discriminant}')
    else:
        raise ValueError(f'template attack type not supported: {atype}')

    # Compute the success information
    print('Computing success info...\n')
    results['success_info'] = get_success_info_like(x_attack, rand_iter, nr_traces_vec, handle_eval, pe3, pe4, pe5, pe6)
    
    return results