from typing import List, Optional, Union

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import Angle, cartesian_to_spherical
from numpy import typing as npt
from tess_stars2px import Levine_FPG

from . import PACKAGEDIR

TPARAMS = pd.read_csv(f"{PACKAGEDIR}/data/tess_params.csv").set_index("sector")


def footprint(npoints=5):
    """Gets the column and row points for CCD edges"""
    column = np.hstack(
        [
            np.zeros(npoints),
            np.linspace(0, 2048, npoints),
            np.linspace(0, 2048, npoints),
            np.ones(npoints) * 2048,
        ]
    )
    row = np.hstack(
        [
            np.linspace(0, 2048, npoints),
            np.zeros(npoints),
            np.ones(npoints) * 2048,
            np.linspace(0, 2048, npoints),
        ]
    )
    return column, row


def sort_patch(patch):
    """Finds the center of a patch, and orders points by angle around the center."""
    angle = np.arctan2(*(patch - np.mean(patch, axis=1)[:, None]))
    p = np.asarray(patch)[:, np.argsort(angle)]
    return p[:, np.hstack([np.arange(p.shape[1]), 0])]


def get_edges(
    sector: Optional[Union[int, List[int], npt.NDArray]] = [1],
    camera: Optional[Union[int, List[int], npt.NDArray]] = None,
    ccd: Optional[Union[int, List[int], npt.NDArray]] = None,
    unit: str = "deg",
    wrap_at: Union[int, float] = 360,
    return_labels: bool = False,
) -> List[npt.NDArray[float]]:
    if camera is None:
        camera = np.arange(1, 5)
    if ccd is None:
        ccd = np.arange(1, 5)
    sector, camera, ccd = (
        np.atleast_1d(sector),
        np.atleast_1d(camera),
        np.atleast_1d(ccd),
    )
    """Finds the edges of specified CCDs for given camera and sector.
    Returns a list of "patches" that mark the edges of each CCD.

    Parameters
    ----------

    sector: Optional[Union[int, List[int], npt.NDArray]]
        Input sectors to process
    camera: Optional[Union[int, List[int], npt.NDArray]]
        Input cameras to process
    ccd: Optional[Union[int, List[int], npt.NDArray]]
        Input CCDs to process
    unit: str
        Astropy unit to use for angles. Use `'deg'` for degrees, `'rad'` for radians.
        For matplotlib projections (e.g. `'mollweide'` or `'hammer'`) use radians.
    wrap_at: Union[int, float]
        The angle at which to "wrap" the edges. By default this is 360 degrees.
        For matplotlib projections (e.g. `'mollweide'` or `'hammer'`) use 180 degrees.
    return_labels: bool
        Returns a list of the sector, camera, and CCD numbers for each patch.

    Returns
    -------

    patches: List[npt.NDArray[float]]
        Returns a list of lists containing the RA and Dec coordinates for each CCD
    """

    patches = []
    labels = []
    for sdx in sector:
        l = Levine_FPG(np.array([TPARAMS.ra[sdx], TPARAMS.dec[sdx], TPARAMS.roll[sdx]]))
        edge_camera, edge_ccd, _, _, ccdxpos, ccdypos = l.radec2pix(
            np.ones(1000) * wrap_at, np.linspace(-90, 90, 1000)
        )
        k = (ccdxpos > 0) & (ccdxpos < 2048) & (ccdypos > 0) & (ccdypos < 2048)
        edge_camera, edge_ccd, ccdxpos, ccdypos = (
            edge_camera[k],
            edge_ccd[k],
            ccdxpos[k],
            ccdypos[k],
        )
        for camdx in camera:
            for ccd_idx in ccd:
                ccdRA, ccdDec = np.asarray(
                    [
                        l.pix2radec_nocheck_single(
                            camdx - 1, ccd_idx - 1, [col - 45.0, row - 1.0]
                        )
                        for col, row in zip(*footprint())
                    ]
                ).T
                ccdRA, ccdDec = Angle(ccdRA, "deg").wrap_at(wrap_at * u.deg), Angle(
                    ccdDec, "deg"
                )
                k = (edge_camera == camdx) & (edge_ccd == ccd_idx)
                edgeDec = np.asarray(
                    [
                        l.pix2radec_nocheck_single(camdx - 1, ccd_idx - 1, (x, y))[1]
                        for x, y in zip(ccdxpos[k], ccdypos[k])
                    ]
                ).T
                if len(edgeDec) != 0:
                    edgeRA1 = Angle(edgeDec * 0 + 1e-6 + wrap_at, "deg").wrap_at(
                        wrap_at * u.deg
                    )
                    edgeRA2 = Angle(edgeDec * 0 - 1e-6 + wrap_at, "deg").wrap_at(
                        wrap_at * u.deg
                    )
                    mask = (
                        np.vstack(
                            [(ccdRA - edgeRA1[0]) ** 2, (ccdRA - edgeRA2[0]) ** 2]
                        )
                        .argmin(axis=0)
                        .astype(bool)
                    )
                    patches.append(
                        sort_patch(
                            [
                                np.hstack([ccdRA[mask], edgeRA2]).to(unit),
                                np.hstack([ccdDec[mask], Angle(edgeDec, "deg")]).to(
                                    unit
                                ),
                            ]
                        )
                    )
                    labels.append([sdx, camdx, ccd_idx])
                    patches.append(
                        sort_patch(
                            [
                                np.hstack([ccdRA[~mask], edgeRA1]).to(unit),
                                np.hstack([ccdDec[~mask], Angle(edgeDec, "deg")]).to(
                                    unit
                                ),
                            ]
                        )
                    )
                    labels.append([sdx, camdx, ccd_idx])
                else:
                    patches.append(sort_patch([ccdRA.to(unit), ccdDec.to(unit)]))
                    labels.append([sdx, camdx, ccd_idx])
    if return_labels:
        return patches, labels
    return patches


def add_tessfov_outline(
    ax: plt.Axes,
    sector: Optional[Union[int, List[int], npt.NDArray]] = [1],
    camera: Optional[Union[int, List[int], npt.NDArray]] = None,
    ccd: Optional[Union[int, List[int], npt.NDArray]] = None,
    unit: str = "deg",
    wrap_at: Union[int, float] = 360,
    color: str = "k",
    **kwargs,
):
    """Adds the outline of the input TESS CCD, camera, and sector to `ax`.

    Parameters
    ----------

    sector: Optional[Union[int, List[int], npt.NDArray]]
        Input sectors to process
    camera: Optional[Union[int, List[int], npt.NDArray]]
        Input cameras to process
    ccd: Optional[Union[int, List[int], npt.NDArray]]
        Input CCDs to process
    unit: str
        Astropy unit to use for angles. Use `'deg'` for degrees, `'rad'` for radians.
        For matplotlib projections (e.g. `'mollweide'` or `'hammer'`) use radians.
    wrap_at: Union[int, float]
        The angle at which to "wrap" the edges. By default this is 360 degrees.
        For matplotlib projections (e.g. `'mollweide'` or `'hammer'`) use 180 degrees.
    color: str
        Color to plot the outline in
    kwargs: dict
        Keywords to pass to `matplotlib.pyplot.plot`
    """
    patches = get_edges(
        sector=sector, camera=camera, ccd=ccd, unit=unit, wrap_at=wrap_at
    )
    for patch in patches:
        ax.plot(patch[0], patch[1], color=color, **kwargs)


def add_tessfov_text(
    ax: plt.Axes,
    sector: Optional[Union[int, List[int], npt.NDArray]] = [1],
    camera: Optional[Union[int, List[int], npt.NDArray]] = None,
    ccd: Optional[Union[int, List[int], npt.NDArray]] = None,
    unit: str = "deg",
    wrap_at: Union[int, float] = 360,
    ha: str = "center",
    va: str = "center",
    color: str = "k",
    **kwargs,
):
    """Adds labels of the input TESS CCD, camera, and sector to `ax`.

    Parameters
    ----------

    sector: Optional[Union[int, List[int], npt.NDArray]]
        Input sectors to process
    camera: Optional[Union[int, List[int], npt.NDArray]]
        Input cameras to process
    ccd: Optional[Union[int, List[int], npt.NDArray]]
        Input CCDs to process
    unit: str
        Astropy unit to use for angles. Use `'deg'` for degrees, `'rad'` for radians.
        For matplotlib projections (e.g. `'mollweide'` or `'hammer'`) use radians.
    wrap_at: Union[int, float]
        The angle at which to "wrap" the edges. By default this is 360 degrees.
        For matplotlib projections (e.g. `'mollweide'` or `'hammer'`) use 180 degrees.
    ha : str
        Horizontal alignment. Default is 'center'
    va : str
        Vertical alignment. Default is 'center'
    color: str
        Color to plot the outline in
    kwargs: dict
        Keywords to pass to `matplotlib.pyplot.plot`
    """
    patches, labels = get_edges(
        sector=sector,
        camera=camera,
        ccd=ccd,
        unit=unit,
        wrap_at=wrap_at,
        return_labels=True,
    )
    for patch, label in zip(patches, labels):
        ax.text(
            patch[0].mean(),
            patch[1].mean(),
            f"Sector {label[0]}\nCam {label[1]}\nCCD {label[2]}",
            ha=ha,
            va=va,
            color=color,
            **kwargs,
        )


def add_tessfov_shade(
    ax: plt.Axes,
    sector: Optional[Union[int, List[int], npt.NDArray]] = [1],
    camera: Optional[Union[int, List[int], npt.NDArray]] = None,
    ccd: Optional[Union[int, List[int], npt.NDArray]] = None,
    unit: str = "deg",
    wrap_at: Union[int, float] = 360,
    color: str = "k",
    alpha: float = 0.15,
    lw: float = 0.75,
    **kwargs,
):

    """Adds a shaded patch corresponding to the input TESS CCD, camera, and sector to `ax`.

    Parameters
    ----------

    sector: Optional[Union[int, List[int], npt.NDArray]]
        Input sectors to process
    camera: Optional[Union[int, List[int], npt.NDArray]]
        Input cameras to process
    ccd: Optional[Union[int, List[int], npt.NDArray]]
        Input CCDs to process
    unit: str
        Astropy unit to use for angles. Use `'deg'` for degrees, `'rad'` for radians.
        For matplotlib projections (e.g. `'mollweide'` or `'hammer'`) use radians.
    wrap_at: Union[int, float]
        The angle at which to "wrap" the edges. By default this is 360 degrees.
        For matplotlib projections (e.g. `'mollweide'` or `'hammer'`) use 180 degrees.
    color: str
        Color to plot the outline in
    alpha: float
        The transparancy (alpha) of the patch
    lw: float
        Line width to outline the patch
    kwargs: dict
        Keywords to pass to `matplotlib.pyplot.Polygon`
    """
    patches = get_edges(
        sector=sector, camera=camera, ccd=ccd, unit=unit, wrap_at=wrap_at
    )
    for patch in patches:
        poly = plt.Polygon(
            np.vstack([patch[0], patch[1]]).T, color=color, alpha=0.15, lw=lw, **kwargs
        )
        ax.add_patch(poly)


def fibonacci_sphere(samples: int = 20000) -> List[npt.NDArray[float]]:
    """Roughly evenly spaces points on a sphere

    Parameters:
    ----------
    samples: int
        Number of points to put on a sphere

    Returns:
    -------
    vec: np.ndarray shape 3 x len(samples)
        The cartesian vector of points on a sphere
    """
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append((x, y, z))
    return np.asarray(points)


def get_completeness(
    sectors: Union[List[int], npt.NDArray] = [1], npoints: int = 100000
) -> List[npt.NDArray[float]]:
    """Returns sky completeness of input sectors for NASA TESS.

    Will evenly distribute points in RA and Dec and then calculate which points are observed
    in a given list of sectors. Returns an array of booleans for each input sector with
    True/False stating whether each point is observed.

    Parameters:
    ----------
    sectors: list or np.ndarray
        The sectors at which to calculate the completeness
    npoints: int
        Number of points to put on a sphere

    Returns:
    -------
    ra: np.ndarray
        The evenly spaced RA locations, shape 1 x npoints
    dec: np.ndarray
        The evenly spaced Dec locations, shape 1 x npoints
    onsilicon: np.ndarray
        Array of booleans, with shape nsectors x npoints.
        True where any point is observed in a given sector.
    """
    # evenly spaced RA and Dec
    a = fibonacci_sphere(samples=npoints)
    lat, lng = cartesian_to_spherical(*a.T)[1:]
    ras, decs = lng.to(u.deg).value, lat.to(u.deg).value

    onsilicon = np.zeros((len(sectors), len(ras)), bool)
    for ldx, sdx in enumerate(sectors):
        l = Levine_FPG(np.array([TPARAMS.ra[sdx], TPARAMS.dec[sdx], TPARAMS.roll[sdx]]))
        vecs = np.asarray(l.sphereToCart(ras, decs))
        camVecs = np.matmul(l.rmat4, vecs)
        # carttosphere
        norm = np.sum(camVecs**2, axis=1) ** 0.5
        lat = np.arcsin(camVecs[:, 2] / norm)
        lng = np.arctan2(camVecs[:, 1], camVecs[:, 0])
        lng = np.mod(lng, 2.0 * np.pi)
        lng, lat = np.rad2deg(lng), np.rad2deg(lat)

        def infov(lng, lat):
            vec = np.asarray(l.sphereToCart(lng, lat))
            vec /= np.sum(vec**2, axis=0) ** 0.5
            xlen = np.abs(np.arctan(vec[0] / vec[2]))
            ylen = np.abs(np.arctan(vec[1] / vec[2]))
            infov = (xlen <= np.deg2rad(12.5)) & (ylen <= np.deg2rad(12.5))
            infov *= lat > 70
            return infov

        onsilicon[ldx] = infov(lng, lat).any(axis=0)
    return ras, decs, onsilicon
