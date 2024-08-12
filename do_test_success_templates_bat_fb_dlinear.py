from get_mmap import get_mmap
from run_template_attack import run_template_attack
import random

# Setup the necessary paths and parameters
fmap = 'e2_bat_fb_beta_raw_s_0_3071.raw'
data_title = 'Templates A2 BAT FB'
path_data = 'results/'
name_data = 'a2_bat_fb_templates_dlinear_n200r_slr_g1000_r10.mat' # TODO figure out
rand_iter = 10
n_profile = 200 # ensure that: n_profile + n_attack < nr_blocks
#nr_traces_vec = [1:10, 20:10:100, 200, 500, 1000]
nr_traces_vec = list(range(1, 11)) + list(range(20, 101, 10)) + [200, 500, 1000]
print(nr_traces_vec)
bytes = list(range(256))
atype = 'mvn'


# Load file
print('Mapping data...')
mmap_data, metadata = get_mmap(fmap)
print('Done mapping data')
# Test:
print(f'mmap data:\n{mmap_data['X'][:10, :]}\n\n')
print(f'Metadata:\n{metadata}')


# Select idx for profile/attack
nr_blocks = 3072
idx = list(range(1, nr_blocks + 1))
idx_profile = [x for x in idx if x % 3 == 1 or x % 3 == 2] # TODO: verify that result is the same as matlab one
idx_profile = random.sample(idx_profile, n_profile) # TODO: verify that this does the same as the matlab one
idx_attack = [x for x in idx if x % 3 == 0]
# Test:
print(f'idx profile: {idx_profile}')
print(f'idx attack: {idx_attack}')


# Set up attack/result cells
# TODO: figure this out (matlab code below)
# results = cell(6, 1);
results = [None] * 6


# Run attack for LDA
# cmethod = 'LDA';
# cparams.lda_dimensions = 4;
# discriminant = 'linearnocov';
# results{1} = run_template_attack(...
#     m_data, metadata, idx_profile, m_data, metadata, idx_attack, ...
#     bytes, atype, cmethod, cparams, discriminant, rand_iter, nr_traces_vec);
cmethod = 'LDA'
cparams = {'lda_dimensions': 4}
discriminant = 'linearnocov'
results[0] = run_template_attack(
    mmap_data, metadata, idx_profile, mmap_data, metadata, idx_attack,
    bytes, atype, cmethod, cparams, discriminant, rand_iter, nr_traces_vec)