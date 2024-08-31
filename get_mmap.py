import struct
import mmap
import numpy as np
from read_metadata import read_metadata

def format_converter(fmt):
    if fmt == 'uint8':
        return 'B', np.uint8
    elif fmt == 'int16':
        return 'h', np.int16
    else:
        raise ValueError(f"Unsupported format: {fmt}")

def get_mmap(fname):
    metadata = read_metadata(fname)

    mmap_data = {}

    with open(fname, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # rindex
        rindex_offset = metadata['ridxoffset']
        rindex_format, rindex_dtype = format_converter(metadata['rifmt'])
        rindex_size = metadata['nr_bytes'] * struct.calcsize(rindex_format)
        rindex_data = mm[rindex_offset:rindex_offset + rindex_size]
        mmap_data['rindex'] = np.frombuffer(rindex_data, dtype=rindex_dtype).reshape((metadata['nr_bytes'],), order='F')

        # X
        X_offset = metadata['xoffset']
        X_format, X_dtype = format_converter(metadata['xfmt'])
        X_size = metadata['nr_points'] * metadata['nr_trials'] * struct.calcsize(X_format)
        X_data = mm[X_offset:X_offset + X_size]
        mmap_data['X'] = np.frombuffer(X_data, dtype=X_dtype).reshape((metadata['nr_points'], metadata['nr_trials']), order='F')

        # B
        B_offset = metadata['boffset']
        B_format, B_dtype = format_converter(metadata['bfmt'])
        B_size = metadata['nr_bytes'] * metadata['nr_trials'] * struct.calcsize(B_format)
        B_data = mm[B_offset:B_offset + B_size]
        mmap_data['B'] = np.frombuffer(B_data, dtype=B_dtype).reshape((metadata['nr_bytes'], metadata['nr_trials']), order='F')

        # R
        R_offset = metadata['roffset']
        R_format, R_dtype = format_converter(metadata['bfmt'])
        R_size = metadata['nr_bytes'] * metadata['nr_trials'] * struct.calcsize(R_format)
        R_data = mm[R_offset:R_offset + R_size]
        mmap_data['R'] = np.frombuffer(R_data, dtype=R_dtype).reshape((metadata['nr_bytes'], metadata['nr_trials']), order='F')

    mm.close()

    return mmap_data, metadata

# Example usage for testing
# fname = 'e2_bat_fb_beta_raw_s_0_3071.raw'
# mmap_data, metadata = get_mmap(fname)
# print(mmap_data['X'][0:30, 0:30]) # Seems to track the matlab output