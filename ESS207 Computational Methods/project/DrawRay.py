import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def DrawRay(
    xmin, xmax,
    zmin, zmax,
    v0,
    rayx_all, rayz_all, rayt_all,
    fig_name="Shadowzone"
):


    nx, nz = 800, 400
    x = np.linspace(xmin, xmax, nx)
    z = np.linspace(zmin, zmax, nz)
    X, Z = np.meshgrid(x, z)
    V = v0(X, Z)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 4), sharey=True
    )

    im_vel = ax1.imshow(
        V,
        extent=[x.min(), x.max(), z.max(), z.min()],
        aspect='auto',
        cmap='jet_r'
    )

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('z (m)')

    ax1.set_title('Velocity model and ray paths')

    colors = plt.cm.viridis(np.linspace(0, 1, len(rayx_all)))
    for i in range(len(rayx_all)):
        ax1.plot(rayx_all[i], rayz_all[i], color=colors[i], lw=2)

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("left", size="4%", pad=0.9)
    cb1 = fig.colorbar(im_vel, cax=cax1)
    cb1.set_label('Velocity (m/s)')
    cax1.yaxis.set_ticks_position('left')
    cax1.yaxis.set_label_position('left')

    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(zmax, zmin)

    ax2.set_title('Ray paths colored by travel time')
    ax2.set_xlabel('x (m)')

    norm = plt.Normalize(
        vmin=min([rt.min() for rt in rayt_all]),
        vmax=max([rt.max() for rt in rayt_all])
    )

    for i in range(len(rayx_all)):
        sc = ax2.scatter(
            rayx_all[i],
            rayz_all[i],
            c=rayt_all[i],
            cmap='plasma',
            norm=norm,
            s=1,
            edgecolors='none'
        )

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="4%", pad=0.1)
    cb2 = fig.colorbar(sc, cax=cax2)
    cb2.set_label('Travel time (s)')

    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(zmax, zmin)

    plt.tight_layout()
    plt.savefig(f"images/{fig_name}.png", dpi=600)
    plt.show()
