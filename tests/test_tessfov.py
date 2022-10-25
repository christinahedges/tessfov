import os

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-white')

SMALL_SIZE = 13
MEDIUM_SIZE = 15
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

from tessfov import (
    PACKAGEDIR,
    __version__,
    add_tessfov_outline,
    add_tessfov_shade,
    add_tessfov_text,
    get_edges,
    get_completeness
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
            fig.savefig(f"{dir}/docs/projection.png", bbox_inches="tight", dpi=150)
            plt.close(fig)

            fig = plt.figure(dpi=150)
            ax = plt.subplot(111)
            ax.set(xlabel="RA", ylabel="Dec", xlim=(0, 360), ylim=(-90, 90))
            add_tessfov_outline(ax, sector=np.arange(1, 14, 3))
            fig.savefig(f"{dir}/docs/regular.png", bbox_inches="tight", dpi=150)
            plt.close(fig)

            fig = plt.figure(dpi=150)
            ax = plt.subplot(111)
            ax.set(xlabel="RA", ylabel="Dec", xlim=(330, 360), ylim=(-40, 0))
            add_tessfov_outline(ax, sector=2, camera=1, ccd=4, color="grey", ls="--")
            add_tessfov_text(ax, sector=2, camera=1, ccd=4, color="grey", fontsize=12)
            fig.savefig(f"{dir}/docs/zoom.png", bbox_inches="tight", dpi=150)
            plt.close(fig)

def test_completeness():
    ra, dec, onsilicon = get_completeness([1, 2])
    assert np.sum(onsilicon, axis=0).max() == 2
    ra, dec, onsilicon = get_completeness([1, 2], npoints=10)
    if not is_action():
        dir = "/".join(PACKAGEDIR.split("/")[:-2])
        sectors = np.arange(1, 84)
        ras, decs, onsilicon = get_completeness(sectors, 100000)
        #plt.scatter(ras, decs, c=onsilicon.any(axis=0))
        cmap = plt.get_cmap('viridis')
        cmap.set_bad(color='grey')

        color = onsilicon.sum(axis=0).astype(float)
        color[color == 0] = np.nan
        fig, ax = plt.subplots(figsize=(10, 6))
        im = plt.scatter(ras, decs, c=color, cmap=cmap, s=1)
        ax.set(xlabel='RA [degrees]', ylabel='Dec [degrees]', title=f'Number of TESS Visits as of Sector {sectors[-1]}')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Number of Visits by TESS")
        fig.savefig(f"{dir}/docs/completeness1.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots()
        plt.hist(onsilicon.sum(axis=0), np.arange(0, onsilicon.sum(axis=0).max()), density=True, color='k')
        ax.set(xlabel="Number of TESS Observations", ylabel="Density", title="Number of TESS Observations across Sky")       
        fig.savefig(f"{dir}/docs/completeness2.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

        completeness = np.asarray([onsilicon[:idx].any(axis=0).sum()/onsilicon.shape[1] for idx in sectors])
        fig, ax = plt.subplots()
        ax.plot(sectors, completeness * 100, c='k', lw=2)
        ax.set(xlabel='TESS Observing Sector', ylabel='Toatl Sky Coverage [%]', title='TESS Total Sky Observed at Least Once', ylim=(0, 100))
        fig.savefig(f"{dir}/docs/completeness3.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
