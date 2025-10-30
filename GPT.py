import math
import numpy as np
import matplotlib.pyplot as plt
import json

# ------------------------------------------------------------
# 1) compute the box size from (N, eta, sigma) for 2D
# ------------------------------------------------------------
def compute_box_length_2d(N, eta, sigma):
    """Return L = sqrt(N * pi * sigma^2 / (4 * eta))"""
    return math.sqrt(N * math.pi * sigma**2 / (4.0 * eta))

# ------------------------------------------------------------
# 2) build a hexagonal (triangular) lattice
# ------------------------------------------------------------
def build_hex_lattice_2d(N, L, sigma, jitter=0.0, seed=None):
    """
    Place N particles on a triangular (hexagonal) lattice inside a box of length L.
    jitter adds a small random wiggle to break perfect symmetry.
    """
    rng = np.random.default_rng(seed)
    a = sigma  # nearest-neighbor spacing ~ one diameter
    row_height = math.sqrt(3) / 2.0 * a

    coords = []
    iy = 0
    while len(coords) < N:
        y = iy * row_height
        if y >= L:
            break
        x_offset = 0.5 * a if (iy % 2 == 1) else 0.0
        ix = 0
        while True:
            x = ix * a + x_offset
            if x >= L:
                break
            coords.append([x, y])
            if len(coords) >= N:
                break
            ix += 1
        iy += 1
    positions = np.array(coords[:N], dtype=float)
    # add a small random jitter and wrap inside the box
    if jitter > 0:
        positions += rng.uniform(-jitter, jitter, size=positions.shape)
    positions = positions % L
    return positions

# ------------------------------------------------------------
# 3) build a square lattice
# ------------------------------------------------------------
def build_square_lattice_2d(N, L, sigma):
    """Place N particles on a square grid with spacing sigma."""
    a = sigma +0.5
    nx = int(math.floor(L / a))
    ny = int(math.floor(L / a))
    coords = []
    for iy in range(ny):
        y = iy * a
        for ix in range(nx):
            x = ix * a
            coords.append([x, y])
            if len(coords) >= N:
                return np.array(coords, dtype=float) % L
    return np.array(coords[:N], dtype=float) % L

# ------------------------------------------------------------
# 4) plot the configuration
# ------------------------------------------------------------
def plot_disks_pbc(positions, L, sigma, title=""):

    fig, ax = plt.subplots(figsize=(L, L))
    ax.set_aspect('equal')
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    if title:
        ax.set_title(title)

    # Tile positions in a 3x3 grid to mimic periodic images
    shifts = np.array([[dx, dy] for dx in (-L, 0.0, L) for dy in (-L, 0.0, L)])
    tiled = (positions[None, :, :] + shifts[:, None, :]).reshape(-1, 2)

    # Keep only centers that can contribute to the visible box
    pad = sigma / 2.0
    mask = (
        (tiled[:, 0] >= -pad) & (tiled[:, 0] <= L + pad) &
        (tiled[:, 1] >= -pad) & (tiled[:, 1] <= L + pad)
    )
    tiled = tiled[mask]

    # Draw disks (they’ll be clipped to the axes box automatically)
    for (x, y) in tiled:
        circ = plt.Circle((x, y), radius=sigma/2, ec='black', lw=0.5)
        ax.add_patch(circ)

    # Box outline (optional)
    ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], 'k-', lw=1)

    # Make the plot tight: no extra whitespace around the axes
    ax.set_xticks([]); ax.set_yticks([])
    plt.margins(0)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.show()

# ------------------------------------------------------------
# 5) run a simple demo
# ------------------------------------------------------------
def main():
    N = 100
    eta = 0.68
    sigma = 0.3
    lattice = "square"  # change to "square" for cubic arrangement
    jitter = 0.0
    seed = 1

    # compute box size from packing fraction (2D)
    L = compute_box_length_2d(N, eta, sigma)
    print(f"Box length L = {L:.5f}")

    # build the lattice
    if lattice == "hex":
        positions = build_hex_lattice_2d(N, L, sigma, jitter=jitter, seed=seed)
        title = f"Hexagonal lattice (triangular) — N={N}, η={eta}, σ={sigma}, L={L:.3f}"
    elif lattice == "square":
        positions = build_square_lattice_2d(N, L, sigma)
        title = f"Square lattice — N={N}, η={eta}, σ={sigma}, L={L:.3f}"
    else:
        raise ValueError("lattice must be 'hex' or 'square'")

    # print a few positions
    print("First 10 positions:")
    for i, (x, y) in enumerate(positions[:10]):
        print(f"{i:3d}: {x:.5f}  {y:.5f}")

    # visualize
    plot_disks_pbc(positions, L, sigma, title="Cubic (square) lattice with PBC")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
