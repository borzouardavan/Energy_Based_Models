"""
Overdamped Langevin diffusion GIF:
    dX = -grad U(X) dt + sqrt(2 D) dW
Particle in a 2D double-well potential, viscous (no inertia).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

rng = np.random.default_rng(7)

# --- potential: double well in x, harmonic in y ---
def U(x, y):
    return (x**2 - 1.0)**2 + 1.2 * y**2

def gradU(p):
    x, y = p
    return np.array([4.0 * x * (x**2 - 1.0), 2.4 * y])

# --- simulate overdamped Langevin (Euler–Maruyama) ---
dt = 0.004
D = 0.32           # noise strength (kT / gamma)
steps_per_frame = 14
n_frames = 220
n_steps = steps_per_frame * n_frames

p = np.array([-1.0, 0.0])
traj = np.empty((n_steps + 1, 2))
traj[0] = p
sq = np.sqrt(2.0 * D * dt)
for i in range(n_steps):
    p = p - gradU(p) * dt + sq * rng.standard_normal(2)
    traj[i + 1] = p

# quick sanity: did it hop wells?
print("well hops:", np.sum(np.diff(np.sign(traj[::steps_per_frame, 0])) != 0))

# --- figure ---
bg = "#0e1420"
fig, ax = plt.subplots(figsize=(5, 5), dpi=88)
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

gx = np.linspace(-2.0, 2.0, 240)
gy = np.linspace(-1.7, 1.7, 220)
GX, GY = np.meshgrid(gx, gy)
Z = U(GX, GY)

ax.contourf(GX, GY, Z, levels=24, cmap="cividis", alpha=0.85)
cs = ax.contour(GX, GY, Z, levels=12, colors="white", linewidths=0.4, alpha=0.25)

ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-1.7, 1.7)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)

ax.text(0.5, 1.015, r"$dX_t = -\nabla U(X_t)\,dt + \sqrt{2D}\,dW_t$",
        transform=ax.transAxes, ha="center", va="bottom",
        color="#9fb3c8", fontsize=10)
ax.text(0.02, 0.02, "particle in a viscous fluid • double-well potential",
        transform=ax.transAxes, color="#fc0404", fontsize=8)

# trail: several line segments with fading alpha
TRAIL = 90  # frames of history
n_seg = 12
trail_lines = []
for k in range(n_seg):
    ln, = ax.plot([], [], color="#ffd166", lw=1.6,
                  alpha=0.05 + 0.75 * (k + 1) / n_seg,
                  solid_capstyle="round", zorder=4)
    trail_lines.append(ln)

dot, = ax.plot([], [], "o", ms=9, color="#ffd166",
               mec="white", mew=1.1, zorder=6)
glow, = ax.plot([], [], "o", ms=20, color="#ffd166", alpha=0.20, zorder=5)

frames_idx = np.arange(n_frames) * steps_per_frame

def update(f):
    i = frames_idx[f]
    lo = max(0, i - TRAIL * steps_per_frame)
    hist = traj[lo:i + 1]
    # split history into n_seg chunks for fading effect
    splits = np.array_split(np.arange(len(hist)), n_seg)
    for k, idx in enumerate(splits):
        if len(idx) > 1:
            seg = hist[idx[0]:idx[-1] + 2]  # +2 to connect chunks
            trail_lines[k].set_data(seg[:, 0], seg[:, 1])
        else:
            trail_lines[k].set_data([], [])
    x, y = traj[i]
    dot.set_data([x], [y])
    glow.set_data([x], [y])
    return trail_lines + [dot, glow]

anim = FuncAnimation(fig, update, frames=n_frames, blit=True)
plt.tight_layout()
anim.save("langevin_diffusion.gif",
          writer=PillowWriter(fps=25))
print("saved")