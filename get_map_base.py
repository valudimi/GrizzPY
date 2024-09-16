def get_map_base(name):
    """
    Returns a mapping base for use in stochastic models.

    Parameters:
    - name: A string containing the name of the mapping model to be used.

    The mapping base returned is a list of functions, where each function
    takes a value `v` and returns the mapping for that base.

    Available mappings:
    - 'F9': The standard 8-bit model. The first base is constant 1, followed by each bit (from MSB to LSB).
    - 'F17': Similar to F9 but for a 16-bit model.
    - 'F17xor': Model F17 plus 8 bits corresponding to the XOR between the two bytes in the 16-bit value.
    - 'F17tran': Model F17 plus 8 bits corresponding to transitions 0->1 and 1->0.
    """

    def bitget(v, pos):
        return (int(v) >> (pos - 1)) & 1

    if name == 'F9':
        map_base = [
            lambda v: 1,
            lambda v: bitget(v, 8),
            lambda v: bitget(v, 7),
            lambda v: bitget(v, 6),
            lambda v: bitget(v, 5),
            lambda v: bitget(v, 4),
            lambda v: bitget(v, 3),
            lambda v: bitget(v, 2),
            lambda v: bitget(v, 1)
        ]

    elif name == 'F17':
        map_base = [
            lambda v: 1,
            lambda v: bitget(v, 16),
            lambda v: bitget(v, 15),
            lambda v: bitget(v, 14),
            lambda v: bitget(v, 13),
            lambda v: bitget(v, 12),
            lambda v: bitget(v, 11),
            lambda v: bitget(v, 10),
            lambda v: bitget(v, 9),
            lambda v: bitget(v, 8),
            lambda v: bitget(v, 7),
            lambda v: bitget(v, 6),
            lambda v: bitget(v, 5),
            lambda v: bitget(v, 4),
            lambda v: bitget(v, 3),
            lambda v: bitget(v, 2),
            lambda v: bitget(v, 1)
        ]

    elif name == 'F17xor':
        map_base = [
            lambda v: 1,
            lambda v: bitget(v, 16),
            lambda v: bitget(v, 15),
            lambda v: bitget(v, 14),
            lambda v: bitget(v, 13),
            lambda v: bitget(v, 12),
            lambda v: bitget(v, 11),
            lambda v: bitget(v, 10),
            lambda v: bitget(v, 9),
            lambda v: bitget(v, 8),
            lambda v: bitget(v, 7),
            lambda v: bitget(v, 6),
            lambda v: bitget(v, 5),
            lambda v: bitget(v, 4),
            lambda v: bitget(v, 3),
            lambda v: bitget(v, 2),
            lambda v: bitget(v, 1),
            lambda v: bitget(v, 16) ^ bitget(v, 8),
            lambda v: bitget(v, 15) ^ bitget(v, 7),
            lambda v: bitget(v, 14) ^ bitget(v, 6),
            lambda v: bitget(v, 13) ^ bitget(v, 5),
            lambda v: bitget(v, 12) ^ bitget(v, 4),
            lambda v: bitget(v, 11) ^ bitget(v, 3),
            lambda v: bitget(v, 10) ^ bitget(v, 2),
            lambda v: bitget(v, 9) ^ bitget(v, 1)
        ]

    elif name == 'F17tran':
        map_base = [
            lambda v: 1,
            lambda v: bitget(v, 16),
            lambda v: bitget(v, 15),
            lambda v: bitget(v, 14),
            lambda v: bitget(v, 13),
            lambda v: bitget(v, 12),
            lambda v: bitget(v, 11),
            lambda v: bitget(v, 10),
            lambda v: bitget(v, 9),
            lambda v: bitget(v, 8),
            lambda v: bitget(v, 7),
            lambda v: bitget(v, 6),
            lambda v: bitget(v, 5),
            lambda v: bitget(v, 4),
            lambda v: bitget(v, 3),
            lambda v: bitget(v, 2),
            lambda v: bitget(v, 1),
            lambda v: (~bitget(v, 16) & bitget(v, 8)),
            lambda v: (~bitget(v, 15) & bitget(v, 7)),
            lambda v: (~bitget(v, 14) & bitget(v, 6)),
            lambda v: (~bitget(v, 13) & bitget(v, 5)),
            lambda v: (~bitget(v, 12) & bitget(v, 4)),
            lambda v: (~bitget(v, 11) & bitget(v, 3)),
            lambda v: (~bitget(v, 10) & bitget(v, 2)),
            lambda v: (~bitget(v, 9) & bitget(v, 1)),
            lambda v: (bitget(v, 16) & ~bitget(v, 8)),
            lambda v: (bitget(v, 15) & ~bitget(v, 7)),
            lambda v: (bitget(v, 14) & ~bitget(v, 6)),
            lambda v: (bitget(v, 13) & ~bitget(v, 5)),
            lambda v: (bitget(v, 12) & ~bitget(v, 4)),
            lambda v: (bitget(v, 11) & ~bitget(v, 3)),
            lambda v: (bitget(v, 10) & ~bitget(v, 2)),
            lambda v: (bitget(v, 9) & ~bitget(v, 1))
        ]

    else:
        raise ValueError(f"Unknown base name: {name}")

    return map_base
