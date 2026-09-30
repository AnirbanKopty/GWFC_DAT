import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import Image, display

from gwosc.datasets import event_gps
from pycbc.detector import Detector
import ligo.skymap.plot

t_gps = event_gps("GW150914")  # GPS time for the event
det = Detector("H1")  # LIGO Hanford
pol = 0
cmap = plt.get_cmap('YlOrRd_r')


# Lower resolution for faster animation rendering
resol_anim = 50
declination_anim = np.linspace(-np.pi/2, np.pi/2, resol_anim)
right_ascension_anim = np.linspace(0, 2*np.pi, resol_anim)
RA_anim, DEC_anim = np.meshgrid(right_ascension_anim, declination_anim)

# 1 full sidereal day (approx 24 hours, Earth rotation)
num_frames = 24
times = t_gps + np.linspace(0, 86400, num_frames, endpoint=False)

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='astro hours mollweide')
ax2 = fig.add_subplot(122, projection='3d')

def compute_response(t_gps_val):
    F_plus = np.zeros_like(RA_anim)
    F_cross = np.zeros_like(RA_anim)
    # Reusing the det object from earlier
    for i in range(RA_anim.shape[0]):
        for j in range(RA_anim.shape[1]):
            fp, fc = det.antenna_pattern(RA_anim[i,j], DEC_anim[i,j], pol, t_gps_val)
            F_plus[i,j] = fp
            F_cross[i,j] = fc
    return np.sqrt(F_plus**2 + F_cross**2)

# Initial mesh for Mollweide
response = compute_response(t_gps)  # Initial response at event time
cs = ax1.contourf(
    np.degrees(RA_anim),
    np.degrees(DEC_anim),
    response,
    levels=50,
    cmap='YlOrRd_r',
    transform=ax1.get_transform('world'),
    vmin=0, vmax=1
)
ax1.invert_xaxis()

# mesh = ax1.pcolormesh(RA_plot_anim, DEC_anim, np.zeros_like(RA_anim), cmap=cmap, shading='auto', vmin=0, vmax=1)
fig.colorbar(cs, ax=ax1, label='Response', shrink=0.5)
ax1.grid(True)

norm = plt.Normalize(0, 1)

def animate(frame):
    global cs  # To modify the contour set defined outside
    t_val = times[frame]
    response_anim = compute_response(t_val)

    # Update Mollweide
    cs.remove()  # Remove the old contour set
    cs = ax1.contourf(      # plot new contours
        np.degrees(RA_anim),
        np.degrees(DEC_anim),
        response_anim,
        levels=50,
        cmap='YlOrRd_r',
        transform=ax1.get_transform('world'),
        vmin=0, vmax=1
    )
    fig.canvas.draw_idle()
    # cs.set_array(response_anim.ravel())
    ax1.set_title(f'Mollweide Pattern\n+{frame} hours', pad=35)

    # Update 3D
    ax2.clear()
    X = response_anim * np.cos(DEC_anim) * np.cos(RA_anim)
    Y = response_anim * np.cos(DEC_anim) * np.sin(RA_anim)
    Z = response_anim * np.sin(DEC_anim)
    colors = cmap(norm(response_anim))
    ax2.plot_surface(X, Y, Z, facecolors=colors, shade=False, antialiased=False)

    ax2.set_xlim([-1, 1]); ax2.set_ylim([-1, 1]); ax2.set_zlim([-1, 1])
    ax2.set_box_aspect([1, 1, 1])
    ax2.set_axis_off()
    ax2.set_title(f'3D Antenna Pattern\n+{frame} hours')

    return [cs]

print(f"Rendering {num_frames} frames... This will take a moment.")
anim = FuncAnimation(fig, animate, frames=num_frames, interval=200, blit=False)

# Save and display as GIF
gif_filename = "antenna_pattern_rotation.gif"
anim.save(gif_filename, writer=PillowWriter(fps=5))
plt.close(fig) # Close the static figure so it doesn't double-plot

print(f"Saved to {gif_filename}")
# display(Image(filename=gif_filename))
