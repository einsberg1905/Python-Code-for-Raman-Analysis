import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# === File paths for 3 datasets ===
file_paths = []

# === Scan parameters ===
num_x, num_y = 50, 50
scan_width = 30.0   # µm
scan_height = 30.0  # µm

# Force origin to (0,0)
x_range = (0, scan_width)
y_range = (0, scan_height)

# === Raman parameters ===
baseline_ranges = [(1200, 1250), (1450, 1500)]
d_peak, g_peak = 1357, 1595
peak_window = 10  # cm⁻¹

# === Load data ===
def load_data(file_path):
    df_raw = pd.read_csv(file_path, delimiter='\t', engine='python', skip_blank_lines=True)
    df = df_raw[pd.to_numeric(df_raw.iloc[:, 0], errors='coerce').notnull()].copy()
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    wn = df.iloc[:, 0].values
    spectra = df.iloc[:, 1:].values.T

    return wn, spectra

# === Baseline correction ===
def correct_baseline(spectrum, wn, baseline_ranges):
    mask = np.zeros_like(wn, dtype=bool)
    for r in baseline_ranges:
        mask |= (wn >= r[0]) & (wn <= r[1])

    x_b = wn[mask]
    y_b = spectrum[mask]

    if len(x_b) < 2:
        return spectrum

    coeffs = np.polyfit(x_b, y_b, 1)
    baseline = np.polyval(coeffs, wn)

    return spectrum - baseline

# === Peak extraction (NO NEGATIVE VALUES) ===
def get_peak_intensity(spectrum, wn, peak, window):
    mask = (wn >= peak - window) & (wn <= peak + window)
    if not np.any(mask):
        return 0

    value = np.max(spectrum[mask])
    return max(value, 0)  # enforce non-negative intensity

# === Compute ID/IG ===
def compute_id_ig_map(wn, spectra):
    d_list, g_list = [], []

    for spec in spectra:
        corrected = correct_baseline(spec, wn, baseline_ranges)

        d = get_peak_intensity(corrected, wn, d_peak, peak_window)
        g = get_peak_intensity(corrected, wn, g_peak, peak_window)

        d_list.append(d)
        g_list.append(g)

    d = np.array(d_list)
    g = np.array(g_list)

    # Only allow physically meaningful ratios
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where((g > 0) & (d >= 0), d / g, np.nan)

    return ratio.reshape((num_y, num_x))

# === Process datasets ===
maps = []
vmin, vmax = None, None

for label, path in file_paths:
    wn, spectra = load_data(path)
    ratio_map = compute_id_ig_map(wn, spectra)

    maps.append((label, ratio_map))

    current_min = np.nanmin(ratio_map)
    current_max = np.nanmax(ratio_map)

    vmin = current_min if vmin is None else min(vmin, current_min)
    vmax = current_max if vmax is None else max(vmax, current_max)

# === Plot ===
fig, axes = plt.subplots(1, len(maps), figsize=(6 * len(maps), 5), dpi=300, constrained_layout=True)

if len(maps) == 1:
    axes = [axes]

for ax, (label, data) in zip(axes, maps):

    img = ax.imshow(
        data,
        cmap='viridis',
        origin='lower',
        extent=[*x_range, *y_range],
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(label, fontsize=11)
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")

    # === Scale bar ===
    scalebar_length = 5  # µm
    bar_x = 0.05 * scan_width
    bar_y = 0.05 * scan_height

    ax.add_patch(
        mpatches.Rectangle((bar_x, bar_y), scalebar_length, 0.6, color='white')
    )

    ax.text(
        bar_x + scalebar_length / 2,
        bar_y + 1,
        '5 µm',
        color='white',
        ha='center',
        va='bottom',
        fontsize=9
    )

    # === Mean value ===
    mean_val = np.nanmean(data)
    ax.text(
        0.5,
        -0.15,
        f"Mean I(D)/I(G): {mean_val:.2f}",
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=9
    )

# === Shared colorbar ===
cbar = fig.colorbar(img, ax=axes, orientation='vertical', shrink=0.8)
cbar.set_label('I(D)/I(G)', fontsize=10)

# === Save ===
output_path = r"
plt.savefig(output_path, dpi=600)

plt.show()
