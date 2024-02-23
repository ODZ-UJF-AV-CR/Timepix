import numpy as np

def get_vhs(x, y, energy):
    # Returns cluster volume, height, size, x coordinate of height and y coordinate of height
    idx = np.argmax(energy)
    return sum(energy), np.max(energy), len(energy), x[idx], y[idx]

def centroid(x, y):
    # Returns unweighted centroid of the cluster
    return (np.mean(x), np.mean(y))

def centroid_weighted(x, y, energy):
    # Returns energy weighted centroid of the cluster
    x = np.array(x)
    y = np.array(y)
    energy = np.array(energy)
    Cx = sum(x * energy)
    Cy = sum(y * energy)
    weight = sum(energy)
    Cx = int(round(Cx / weight, 0))
    Cy = int(round(Cy / weight, 0))
    return Cx, Cy

def hot_pixel_excluded(energy):
    # Returns cluster volume the cluster without the hottest pixel and the height of the second brightest pixel
    if len(energy) == 1:
        return sum(energy), max(energy)
    energy.remove(max(energy))
    return sum(energy), max(energy)

def calibrate_pixel(en, a, b, c, t):
    # Calibrates the pixel ToT to keV
    discriminant = en**2 - 2 * a * t * en - 2 * b * en + a**2 * t**2 + 2 * a * b * t + 4 * a * c + b**2
    out = (en + a * t - b + np.sqrt(discriminant)) / (2 * a)
    out = round(out, 5)
    return out

def decalibrate_pixel(energy, a, b, c, t):
    # Decalibrates the pixel energy from keV to ToT
    out = (a * np.array(energy) + b - (c / (np.array(energy) - t))) + 0.5
    out = out.astype(int)
    return out

