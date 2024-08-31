import struct

def read_metadata(fname):
    """
    Reads metadata from a given data file.
    metadata = read_metadata(fname)
    reads the metadata from a given data file and returns this as a
    structure.

    fname should be the filename (including path) of the file containing
    the data to be read. The file should be in a supported format.

    This method will return a metadata structure, containing information
    about the data file. The only mandatory and common value is the
    'format' field, which should be used to determine the kind of data
    available in the file and the other fields that are returned. See below
    the exact fields for each type of supported format.

    You can use the returned metadata with get_mmap to obtain a memmapfile
    object to access the data.

    See also get_mmap.
    """
    metadata = {}

    try:
        with open(fname, 'rb') as f:
            # Read format
            fs = struct.unpack('B', f.read(1))[0]
            metadata['format'] = f.read(fs).decode('utf-8')
            f.read(7 - fs) # This somehow helps us ignore padded 0s

            if metadata['format'] == 'raw':
                metadata['machinefmt'] = '<'
                # '<Q' is the format specifier for little endian unsigned long long
                metadata['nr_traces'] = struct.unpack(metadata['machinefmt'] + 'Q', f.read(8))[0]
                metadata['nr_points'] = struct.unpack(metadata['machinefmt'] + 'Q', f.read(8))[0]
                metadata['nr_groups'] = struct.unpack(metadata['machinefmt'] + 'Q', f.read(8))[0]

                # Reads ps bytes and decodes them as utf-8
                ps = struct.unpack(metadata['machinefmt'] + 'B', f.read(1))[0]
                metadata['precision'] = f.read(ps).decode('utf-8')
                f.read(7 - ps) # Ignore padded 0s

                # Marks offset at which the data starts (after header)
                metadata['offset'] = 40

            elif metadata['format'] == 'tpl':
                metadata['machinefmt'] = '<'
                metadata['np'] = struct.unpack(metadata['machinefmt'] + 'Q', f.read(8))[0]
                metadata['nr_points'] = struct.unpack(metadata['machinefmt'] + 'Q', f.read(8))[0]
                metadata['nr_groups'] = struct.unpack(metadata['machinefmt'] + 'Q', f.read(8))[0]
                ps = struct.unpack(metadata['machinefmt'] + 'B', f.read(1))[0]
                metadata['precision'] = f.read(ps).decode('utf-8')
                f.read(7 - ps) # Ignore padded 0s
                metadata['ni'] = struct.unpack(metadata['machinefmt'] + metadata['precision'] * \
                    metadata['np'], f.read(metadata['np'] * struct.calcsize(metadata['precision'])))[0]
                metadata['offset'] = 40 + (8 * metadata['np'])

            elif metadata['format'] == 'mat2':
                metadata['machinefmt'] = '<'
                metadata['nr_rows'] = struct.unpack(metadata['machinefmt'] + 'Q', f.read(8))[0]
                metadata['nr_cols'] = struct.unpack(metadata['machinefmt'] + 'Q', f.read(8))[0]
                ps = struct.unpack(metadata['machinefmt'] + 'B', f.read(1))[0]
                metadata['precision'] = f.read(ps).decode('utf-8')
                f.read(7 - ps)  # Ignore padded 0s
                metadata['offset'] = 32

            # What we're working with: rawe2
            elif metadata['format'] == 'rawe2':
                metadata['machinefmt'] = '<'
                metadata['nr_trials'] = struct.unpack(metadata['machinefmt'] + 'Q', f.read(8))[0]
                metadata['nr_groups'] = struct.unpack(metadata['machinefmt'] + 'Q', f.read(8))[0]
                metadata['nr_points'] = struct.unpack(metadata['machinefmt'] + 'Q', f.read(8))[0]
                s_xfmt = struct.unpack(metadata['machinefmt'] + 'B', f.read(1))[0]
                metadata['xfmt'] = f.read(s_xfmt).decode('utf-8')
                f.read(7 - s_xfmt)

                # d is the format specifier for double
                metadata['samplingrate'] = struct.unpack(metadata['machinefmt'] + 'd', f.read(8))[0]
                metadata['fclock'] = struct.unpack(metadata['machinefmt'] + 'd', f.read(8))[0]
                metadata['tscale'] = struct.unpack(metadata['machinefmt'] + 'd', f.read(8))[0]
                metadata['toffset'] = struct.unpack(metadata['machinefmt'] + 'd', f.read(8))[0]
                metadata['vscale'] = struct.unpack(metadata['machinefmt'] + 'd', f.read(8))[0]
                metadata['voffset'] = struct.unpack(metadata['machinefmt'] + 'd', f.read(8))[0]
                metadata['rvalue'] = struct.unpack(metadata['machinefmt'] + 'd', f.read(8))[0]
                metadata['dccoupling'] = struct.unpack(metadata['machinefmt'] + 'q', f.read(8))[0]
                metadata['nr_bytes'] = struct.unpack(metadata['machinefmt'] + 'Q', f.read(8))[0]
                s_bfmt = struct.unpack(metadata['machinefmt'] + 'B', f.read(1))[0]
                metadata['bfmt'] = f.read(s_bfmt).decode('utf-8')
                f.read(7 - s_bfmt) # Ignore padded 0s
                metadata['address'] = struct.unpack(metadata['machinefmt'] + 'Q', f.read(8))[0]
                s_rifmt = struct.unpack(metadata['machinefmt'] + 'B', f.read(1))[0]
                metadata['rifmt'] = f.read(s_rifmt).decode('utf-8')
                f.read(7 - s_rifmt)  # ignore padded zeros
                metadata['ridxoffset'] = 136
                ribs = get_bytes_class(metadata['rifmt'])
                metadata['xoffset'] = metadata['ridxoffset'] + metadata['nr_bytes'] * ribs
                xbs = get_bytes_class(metadata['xfmt'])
                metadata['boffset'] = metadata['xoffset'] + metadata['nr_trials'] * \
                    metadata['nr_points'] * xbs
                rbs = get_bytes_class(metadata['bfmt'])
                metadata['roffset'] = metadata['boffset'] + metadata['nr_trials'] * \
                    metadata['nr_bytes'] * rbs

            else:
                raise ValueError('Unknown format')
    except ValueError as e:
        print(f'Could not open file {e}')

    return metadata

def get_bytes_class(classname):
    """
    Returns the number of bytes occuppied by a numeric class (uint8, int64, float, etc.)
    [bytes] = GET_BYTES_CLASS(classname)
    """

    class_to_bytes = {
        'double': 8,
        'uint64': 8,
        'int64': 8,
        'integer*4': 8,
        'float64': 8,
        'real*8': 8,
        'single': 4,
        'uint32': 4,
        'int32': 4,
        'uint': 4,
        'int': 4,
        'float': 4,
        'float32': 4,
        'integer*3': 4,
        'real*4': 4,
        'uint16': 2,
        'ushort': 2,
        'int16': 2,
        'integer*2': 2,
        'short': 2,
        'uint8': 1,
        'uchar': 1,
        'int8': 1,
        'integer*1': 1,
        'schar': 1,
        'char*1': 1
    }

    if classname in class_to_bytes:
        return class_to_bytes[classname]
    raise ValueError('Unknown numerical class format')
