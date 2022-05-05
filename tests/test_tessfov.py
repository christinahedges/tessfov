import os

import matplotlib.pyplot as plt
import numpy as np

from tessfov import (
    PACKAGEDIR,
    __version__,
    add_tessfov_outline,
    add_tessfov_shade,
    add_tessfov_text,
    get_edges,
)


def is_action():
    try:
        os.environ["GITHUB_ACTIONS"]
        return True
    except KeyError:
        return False


def test_version():
    assert __version__ == "0.1.2"


def test_edges():
    patches = get_edges(1, 1, 1)
    patches = get_edges(1, 1, 1, wrap_at=180)
    patches = get_edges(1, 1, 1, wrap_at=180, unit="rad")
    patches, labels = get_edges(1, 1, 1, wrap_at=180, unit="rad", return_labels=True)
    patches = get_edges(9, 1, [2, 3], wrap_at=180)
    assert isinstance(patches, list)
    assert isinstance(patches[0], np.ndarray)
    assert isinstance(patches[0][0][0], float)


def test_plots():
    if not is_action():
        dir = "/".join(PACKAGEDIR.split("/")[:-2])
        with plt.style.context("seaborn-white"):
            fig = plt.figure(dpi=150)
            ax = plt.subplot(111, projection="hammer")
            add_tessfov_shade(
                ax, sector=np.arange(56, 60), unit="rad", wrap_at=180, rasterized=True
            )
            ax.set(xlabel="RA", ylabel="Dec")
            ax.grid(True, ls="--")
            fig.savefig(f"{dir}/docs/projection.png", bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(dpi=150)
            ax = plt.subplot(111)
            ax.set(xlabel="RA", ylabel="Dec", xlim=(0, 360), ylim=(-90, 90))
            add_tessfov_outline(ax, sector=np.arange(1, 14, 3))
            fig.savefig(f"{dir}/docs/regular.png", bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(dpi=150)
            ax = plt.subplot(111)
            ax.set(xlabel="RA", ylabel="Dec", xlim=(330, 360), ylim=(-40, 0))
            add_tessfov_outline(ax, sector=2, camera=1, ccd=4, color="grey", ls="--")
            add_tessfov_text(ax, sector=2, camera=1, ccd=4, color="grey", fontsize=12)
            fig.savefig(f"{dir}/docs/zoom.png", bbox_inches="tight")
            plt.close(fig)
