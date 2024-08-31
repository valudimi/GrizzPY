import numpy as np
from scipy.signal import find_peaks

def get_selection(curve, selection, p1, p2=None):
    """
    Returns a vector of interesting points for template attacks based on the signal curve and selection parameters.
    
    Parameters:
    - curve: 1D array containing the signal curve from which to select points.
    - selection: String defining the selection method to be used.
    - p1: Parameter for the given selection method, e.g., the number of samples distance for 1ppc.
    - p2: Optional parameter which might be used to change some default values, such as the minimum percentile for 1ppc.
    
    Returns:
    - points: Array of selected points based on the given method.
    """
    curve = np.asarray(curve).flatten()
    points = []

    if selection == '1ppc':
        scurve = np.sort(curve)
        if p2 is None:
            p2 = 0.95
        idx = int(p2 * len(curve))
        min_peak_height = scurve[idx]
        peaks, _ = find_peaks(curve, distance=p1, height=min_peak_height)
        points = peaks

    elif selection == '3ppc':
        scurve = np.sort(curve)
        if p2 is None:
            p2 = 0.95
        idx = int(p2 * len(curve))
        min_peak_height = scurve[idx]
        pstart, _ = find_peaks(curve, distance=p1, height=min_peak_height)
        for p in pstart:
            if p > 0:
                points.append(p - 1)
            points.append(p)
            if p < len(curve) - 1:
                points.append(p + 1)
        points = np.array(points)

    elif selection == '20ppc':
        scurve = np.sort(curve)
        if p2 is None:
            p2 = 0.95
        idx = int(p2 * len(curve))
        min_peak_height = scurve[idx]
        pks, _ = find_peaks(curve, distance=p1, height=min_peak_height)
        points = []
        offset = 0 if (pks[0] - p1 // 2) > 0 else abs(pks[0] - p1 // 2)
        for pk in pks:
            start = pk - p1 // 2 + offset
            ival = np.arange(start, start + p1 + 1)
            ival = ival[ival < len(curve)]
            sorted_indices = np.argsort(curve[ival])[::-1]
            pset = []
            for i in sorted_indices[:min(len(ival), 20)]:
                if curve[ival[i]] > scurve[idx]:
                    pset.append(ival[i])
            points.extend(sorted(pset))
        points = np.array(points)

    elif selection == 'allam':
        mcurve = np.mean(curve)
        points = np.where(curve > mcurve)[0]

    elif selection == 'allap':
        if p1 < 0 or p1 >= 1:
            raise ValueError('Wrong percentile number, must be between 0 and 1')
        scurve = np.sort(curve)
        idx = int(p1 * len(curve))
        points = np.where(curve > scurve[idx])[0]

    elif selection == 'all':
        points = np.arange(len(curve))

    return points

# Example usage
# Assuming curve, selection, p1, and p2 are already defined
# points = get_selection(curve, selection, p1, p2)

