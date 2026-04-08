"""
==============================================================================
  2D FDTD Electromagnetic Solver — TM (Transverse Magnetic) Mode
  Video output version — generates MP4 animation with 2×2 panel:
    Top-left:     E_z field
    Top-right:    H_x field
    Bottom-left:  H_y field
    Bottom-right: |H| magnitude + quiver arrows showing H-field circulation
==============================================================================
"""

import numpy as np
import sys
import os

# GUI drawing needs a real backend; switch to Agg only after drawing is done
import matplotlib
if '--no-gui' in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
c0   = 3.0e8
mu0  = 4.0e-7 * np.pi
eps0 = 1.0 / (mu0 * c0**2)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  GRID PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
Nx = 200
Ny = 200
dx = 1e-3
dy = 1e-3

# ─────────────────────────────────────────────────────────────────────────────
# 3.  TIME-STEPPING & CFL STABILITY
# ─────────────────────────────────────────────────────────────────────────────
courant_number = 0.99
dt = courant_number / (c0 * np.sqrt(1.0/dx**2 + 1.0/dy**2))

print(f"Grid       : {Nx} x {Ny}")
print(f"Cell size  : dx = {dx*1e3:.2f} mm,  dy = {dy*1e3:.2f} mm")
print(f"Time step  : dt = {dt:.4e} s  (Courant # = {courant_number})")

n_steps = 400

# ─────────────────────────────────────────────────────────────────────────────
# 4.  SOURCE PARAMETERS  (Gaussian Pulse)
# ─────────────────────────────────────────────────────────────────────────────
src_x   = Nx // 2
src_y   = Ny // 2
spread  = 12.0 * dt
t0      = 3.0 * spread

# ─────────────────────────────────────────────────────────────────────────────
# 5.  FIELD ARRAY ALLOCATION
# ─────────────────────────────────────────────────────────────────────────────
Ez = np.zeros((Nx, Ny), dtype=np.float64)
Hx = np.zeros((Nx, Ny), dtype=np.float64)
Hy = np.zeros((Nx, Ny), dtype=np.float64)

coeff_hx = dt / (mu0 * dy)
coeff_hy = dt / (mu0 * dx)
coeff_ez = dt / eps0

# ─────────────────────────────────────────────────────────────────────────────
# 5b. PEC OBSTACLE — draw interactively or use a preset shape
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   python fdtd_video_generator.py          — opens GUI to draw obstacle
#   python fdtd_video_generator.py --no-gui — uses the preset shape below
#
# GUI controls:
#   Left-click + drag   = draw obstacle (paint brush)
#   Right-click + drag   = erase obstacle
#   Scroll wheel         = change brush size
#   Press 'c'            = clear all
#   Press 'f'            = fill enclosed region (flood fill from edges)
#   Close window / 'q'   = accept & run simulation

def make_obstacle_preset(Nx, Ny, shape="circle", **kwargs):
    """Generate a boolean mask for a PEC obstacle (preset shapes)."""
    mask = np.zeros((Nx, Ny), dtype=bool)
    if shape == "none":
        return mask
    elif shape == "circle":
        cx = kwargs.get('cx', Nx // 4)
        cy = kwargs.get('cy', Ny // 2)
        r  = kwargs.get('r', 15)
        ii, jj = np.ogrid[:Nx, :Ny]
        mask = (ii - cx)**2 + (jj - cy)**2 <= r**2
    elif shape == "rectangle":
        x0, y0 = kwargs.get('x0', Nx//4-10), kwargs.get('y0', Ny//2-20)
        x1, y1 = kwargs.get('x1', Nx//4+10), kwargs.get('y1', Ny//2+20)
        mask[x0:x1, y0:y1] = True
    elif shape == "polygon":
        from matplotlib.path import Path
        vertices = kwargs.get('vertices', [(30,80),(30,120),(60,120),(60,80)])
        path = Path(vertices)
        ii, jj = np.mgrid[:Nx, :Ny]
        points = np.column_stack([ii.ravel(), jj.ravel()])
        mask = path.contains_points(points).reshape(Nx, Ny)
    return mask


def draw_obstacle_gui(Nx, Ny, src_x, src_y):
    """Open a matplotlib window to draw the obstacle mask with the mouse."""
    mask = np.zeros((Nx, Ny), dtype=bool)
    state = {'drawing': False, 'erasing': False, 'brush': 5}

    fig_draw, ax_draw = plt.subplots(figsize=(7, 7))
    ax_draw.set_title(
        "Draw PEC Obstacle\n"
        "Left-click=draw | Right-click=erase | Scroll=brush size | 'c'=clear | Close=accept",
        fontsize=10)
    ax_draw.set_xlabel("j (column)")
    ax_draw.set_ylabel("i (row)")

    # Show grid with source marked
    im_draw = ax_draw.imshow(mask.T.astype(float), origin='lower',
                             cmap='Greys', vmin=0, vmax=1, aspect='equal',
                             interpolation='nearest')
    ax_draw.plot(src_x, src_y, 'r+', markersize=14, mew=2.5, label='Source')
    # Draw PEC boundary
    ax_draw.axhline(0, color='gray', lw=2); ax_draw.axhline(Ny-1, color='gray', lw=2)
    ax_draw.axvline(0, color='gray', lw=2); ax_draw.axvline(Nx-1, color='gray', lw=2)
    ax_draw.set_xlim(-1, Nx); ax_draw.set_ylim(-1, Ny)
    ax_draw.legend(loc='upper right')

    # Brush circle indicator
    brush_circle = plt.Circle((0, 0), state['brush'], fill=False,
                               color='red', lw=1.5, ls='--', visible=False)
    ax_draw.add_patch(brush_circle)

    def paint(x_data, y_data, value):
        """Paint or erase a circle of cells around (x_data, y_data)."""
        ci, cj = int(round(x_data)), int(round(y_data))
        r = state['brush']
        i0, i1 = max(0, ci - r), min(Nx, ci + r + 1)
        j0, j1 = max(0, cj - r), min(Ny, cj + r + 1)
        for i in range(i0, i1):
            for j in range(j0, j1):
                if (i - ci)**2 + (j - cj)**2 <= r**2:
                    mask[i, j] = value
        im_draw.set_data(mask.T.astype(float))
        fig_draw.canvas.draw_idle()

    def on_press(event):
        if event.inaxes != ax_draw or event.xdata is None:
            return
        if event.button == 1:
            state['drawing'] = True
            paint(event.xdata, event.ydata, True)
        elif event.button == 3:
            state['erasing'] = True
            paint(event.xdata, event.ydata, False)

    def on_release(event):
        state['drawing'] = False
        state['erasing'] = False

    def on_motion(event):
        if event.inaxes != ax_draw or event.xdata is None:
            return
        # Update brush indicator
        brush_circle.center = (event.xdata, event.ydata)
        brush_circle.radius = state['brush']
        brush_circle.set_visible(True)
        if state['drawing']:
            paint(event.xdata, event.ydata, True)
        elif state['erasing']:
            paint(event.xdata, event.ydata, False)
        else:
            fig_draw.canvas.draw_idle()

    def on_scroll(event):
        if event.button == 'up':
            state['brush'] = min(40, state['brush'] + 1)
        else:
            state['brush'] = max(1, state['brush'] - 1)
        brush_circle.radius = state['brush']
        ax_draw.set_title(
            f"Brush size: {state['brush']}  |  "
            "Left=draw | Right=erase | Scroll=size | 'c'=clear | Close=accept",
            fontsize=10)
        fig_draw.canvas.draw_idle()

    def on_key(event):
        if event.key == 'c':
            mask[:] = False
            im_draw.set_data(mask.T.astype(float))
            fig_draw.canvas.draw_idle()
        elif event.key == 'q':
            plt.close(fig_draw)

    fig_draw.canvas.mpl_connect('button_press_event', on_press)
    fig_draw.canvas.mpl_connect('button_release_event', on_release)
    fig_draw.canvas.mpl_connect('motion_notify_event', on_motion)
    fig_draw.canvas.mpl_connect('scroll_event', on_scroll)
    fig_draw.canvas.mpl_connect('key_press_event', on_key)

    plt.show(block=True)
    return mask


# ── Choose obstacle mode ──
if '--no-gui' in sys.argv:
    # Preset mode: edit this line to change the shape
    obstacle_mask = make_obstacle_preset(Nx, Ny, "circle", cx=60, cy=100, r=18)
    # obstacle_mask = make_obstacle_preset(Nx, Ny, "rectangle", x0=30, y0=80, x1=55, y1=120)
    # obstacle_mask = make_obstacle_preset(Nx, Ny, "none")
else:
    print("\n>>> Opening obstacle drawing tool...")
    print("    Draw your obstacle, then close the window to start simulation.\n")
    obstacle_mask = draw_obstacle_gui(Nx, Ny, src_x, src_y)

n_obstacle_cells = int(np.sum(obstacle_mask))
print(f"Obstacle   : {n_obstacle_cells} cells")

# Switch to Agg backend for headless video rendering
matplotlib.use('Agg')
plt.switch_backend('Agg')

# ─────────────────────────────────────────────────────────────────────────────
# 6.  VIDEO & OUTPUT SETUP
# ─────────────────────────────────────────────────────────────────────────────
output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(output_dir, exist_ok=True)

frame_interval = 2
fps            = 30

# Quiver subsampling — one arrow every `qstep` cells
qstep = 10
qx = np.arange(0, Nx, qstep)
qy = np.arange(0, Ny, qstep)
QX, QY = np.meshgrid(qx * dx * 1e3, qy * dy * 1e3, indexing='ij')

extent_mm = [0, Nx*dx*1e3, 0, Ny*dy*1e3]
src_x_mm  = src_x * dx * 1e3
src_y_mm  = src_y * dy * 1e3

# Energy tracking
time_axis    = np.zeros(n_steps)
total_energy = np.zeros(n_steps)

# ─────────────────────────────────────────────────────────────────────────────
# 6a. FIGURE SETUP — 2×2 Panel
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ax_ez  = axes[0, 0]
ax_hx  = axes[0, 1]
ax_hy  = axes[1, 0]
ax_hmag = axes[1, 1]

dummy = np.zeros((Ny, Nx))

# --- Ez panel (red-blue diverging) ---
im_ez = ax_ez.imshow(dummy, origin='lower', cmap='RdBu_r',
                     vmin=-1, vmax=1, extent=extent_mm, aspect='equal')
ax_ez.set_xlabel("x [mm]"); ax_ez.set_ylabel("y [mm]")
title_ez = ax_ez.set_title("")
plt.colorbar(im_ez, ax=ax_ez, label="$E_z$ [V/m]", shrink=0.78)
ax_ez.plot(src_x_mm, src_y_mm, 'k+', markersize=8, mew=1.5)

# --- Hx panel (purple-orange diverging) ---
im_hx = ax_hx.imshow(dummy, origin='lower', cmap='PuOr',
                      vmin=-1, vmax=1, extent=extent_mm, aspect='equal')
ax_hx.set_xlabel("x [mm]"); ax_hx.set_ylabel("y [mm]")
title_hx = ax_hx.set_title("")
plt.colorbar(im_hx, ax=ax_hx, label="$H_x$ [A/m]", shrink=0.78)
ax_hx.plot(src_x_mm, src_y_mm, 'k+', markersize=8, mew=1.5)

# --- Hy panel (pink-green diverging) ---
im_hy = ax_hy.imshow(dummy, origin='lower', cmap='PiYG',
                      vmin=-1, vmax=1, extent=extent_mm, aspect='equal')
ax_hy.set_xlabel("x [mm]"); ax_hy.set_ylabel("y [mm]")
title_hy = ax_hy.set_title("")
plt.colorbar(im_hy, ax=ax_hy, label="$H_y$ [A/m]", shrink=0.78)
ax_hy.plot(src_x_mm, src_y_mm, 'k+', markersize=8, mew=1.5)

# --- |H| magnitude + quiver arrows showing circulation ---
im_hmag = ax_hmag.imshow(dummy, origin='lower', cmap='inferno',
                         vmin=0, vmax=1, extent=extent_mm, aspect='equal')
ax_hmag.set_xlabel("x [mm]"); ax_hmag.set_ylabel("y [mm]")
title_hmag = ax_hmag.set_title("")
plt.colorbar(im_hmag, ax=ax_hmag, label="|H| [A/m]", shrink=0.78)
ax_hmag.plot(src_x_mm, src_y_mm, 'k+', markersize=8, mew=1.5)

# Initial quiver (updated each frame)
arrow_len = qstep * dx * 1e3 * 0.55
quiver = ax_hmag.quiver(QX, QY, np.zeros_like(QX), np.zeros_like(QY),
                        color='white', scale=1, scale_units='xy',
                        headwidth=4, headlength=5, alpha=0.85)

# --- Draw PEC obstacle as solid overlay on all 4 panels ---
if n_obstacle_cells > 0:
    # Create an RGBA overlay: solid dark gray where obstacle is, transparent elsewhere
    obs_rgba = np.zeros((Nx, Ny, 4))
    obs_rgba[obstacle_mask, 0] = 0.15   # R
    obs_rgba[obstacle_mask, 1] = 0.15   # G
    obs_rgba[obstacle_mask, 2] = 0.15   # B
    obs_rgba[obstacle_mask, 3] = 0.85   # alpha (solid)

    for ax in [ax_ez, ax_hx, ax_hy, ax_hmag]:
        ax.imshow(obs_rgba.transpose(1, 0, 2), origin='lower',
                  extent=extent_mm, aspect='equal', zorder=5)
        # Draw obstacle outline
        ax.contour(obstacle_mask.T, levels=[0.5], colors='white',
                   linewidths=1.5, extent=extent_mm, origin='lower', zorder=6)

suptitle = fig.suptitle("", fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])

# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN TIME-STEPPING LOOP  +  VIDEO RECORDING
# ─────────────────────────────────────────────────────────────────────────────
video_path = os.path.join(output_dir, "fdtd_all_fields.mp4")
writer = FFMpegWriter(fps=fps,
                      metadata={'title': '2D FDTD TM-mode — All Fields'},
                      bitrate=5000)

print(f"\nRunning {n_steps} time steps and writing combined video …")

with writer.saving(fig, video_path, dpi=120):
    for n in range(n_steps):
        t = n * dt
        time_axis[n] = t

        # --- UPDATE H (n-½ → n+½) ---
        Hx[:, :-1] -= coeff_hx * (Ez[:, 1:] - Ez[:, :-1])
        Hy[:-1, :] += coeff_hy * (Ez[1:, :] - Ez[:-1, :])

        # --- UPDATE E (n → n+1) ---
        Ez[1:, 1:] += coeff_ez * (
            (Hy[1:, 1:] - Hy[:-1, 1:]) / dx
          - (Hx[1:, 1:] - Hx[1:, :-1]) / dy
        )

        # --- INJECT SOURCE ---
        pulse = np.exp(-((t - t0) / spread) ** 2)
        Ez[src_x, src_y] = pulse

        # --- PEC BOUNDARY CONDITIONS ---
        Ez[0, :]  = 0.0;  Ez[-1, :] = 0.0
        Ez[:, 0]  = 0.0;  Ez[:, -1] = 0.0

        # --- PEC OBSTACLE (same physics as walls) ---
        Ez[obstacle_mask] = 0.0

        # --- ENERGY ---
        U_E = 0.5 * eps0 * np.sum(Ez**2) * dx * dy
        U_H = 0.5 * mu0  * np.sum(Hx**2 + Hy**2) * dx * dy
        total_energy[n] = U_E + U_H

        # --- GRAB VIDEO FRAME ---
        if n % frame_interval == 0:
            t_ns = t * 1e9

            # Ez panel
            vmax_ez = max(1e-3, np.max(np.abs(Ez)))
            im_ez.set_data(Ez.T)
            im_ez.set_clim(-vmax_ez, vmax_ez)
            title_ez.set_text(f"$E_z$  (step {n})")

            # Hx panel
            vmax_hx = max(1e-6, np.max(np.abs(Hx)))
            im_hx.set_data(Hx.T)
            im_hx.set_clim(-vmax_hx, vmax_hx)
            title_hx.set_text(f"$H_x$  (step {n})")

            # Hy panel
            vmax_hy = max(1e-6, np.max(np.abs(Hy)))
            im_hy.set_data(Hy.T)
            im_hy.set_clim(-vmax_hy, vmax_hy)
            title_hy.set_text(f"$H_y$  (step {n})")

            # |H| magnitude panel
            Hmag = np.sqrt(Hx**2 + Hy**2)
            vmax_h = max(1e-6, np.max(Hmag))
            im_hmag.set_data(Hmag.T)
            im_hmag.set_clim(0, vmax_h)
            title_hmag.set_text(f"|H| + circulation  (step {n})")

            # Quiver arrows — subsample, normalise, mask weak regions
            Hx_sub = Hx[np.ix_(qx, qy)]
            Hy_sub = Hy[np.ix_(qx, qy)]
            mag_sub = np.sqrt(Hx_sub**2 + Hy_sub**2)
            mag_safe = np.where(mag_sub > 0, mag_sub, 1.0)

            # In-plane H vector: Hx is the x-component, Hy is the
            # y-component — no swap, no sign flip needed.
            U_arr = (Hx_sub / mag_safe) * arrow_len
            V_arr = (Hy_sub / mag_safe) * arrow_len

            # Suppress arrows in very weak regions
            mask = mag_sub < (vmax_h * 0.02)
            U_arr[mask] = 0.0
            V_arr[mask] = 0.0
            quiver.set_UVC(U_arr, V_arr)

            suptitle.set_text(
                f"2D TM-mode FDTD  —  t = {t_ns:.2f} ns   |   "
                f"U = {total_energy[n]:.3e} J"
            )
            writer.grab_frame()

            if n % 40 == 0:
                print(f"  step {n:4d}  |  t = {t_ns:.3f} ns  "
                      f"|  U = {total_energy[n]:.4e} J")

plt.close(fig)
print(f"\nCombined video saved: {video_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  POST-PROCESSING — ENERGY CONSERVATION PLOT
# ─────────────────────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(time_axis * 1e9, total_energy, color='teal', linewidth=1.2)
ax2.axvline(x=(t0 + 3*spread)*1e9, color='gray', ls='--', lw=0.8,
            label='Source off (~$t_0 + 3\\sigma$)')
ax2.set_xlabel("Time  [ns]")
ax2.set_ylabel("Total EM Energy  [J]")
ax2.set_title("Total Electromagnetic Energy in the PEC Cavity")
ax2.legend()
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
energy_path = os.path.join(output_dir, "energy_conservation.png")
fig2.savefig(energy_path, dpi=150)
plt.close(fig2)
print(f"Energy plot saved: {energy_path}")
print(f"\nAll outputs in ./{output_dir}/")
