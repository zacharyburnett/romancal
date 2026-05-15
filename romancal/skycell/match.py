"""
This module determines which sky cells overlap with the given image.

Currently this assumes that the sky projected borders of all calibrated L2 images are straight.
"""

import logging

import numpy as np
from gwcs import WCS
from numpy.typing import NDArray
from sphersgeo import SphericalPolygon
from sphersgeo.from_wcs import polygon_from_wcs

import romancal.skycell.skymap as sc

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

__all__ = ["find_skycell_matches"]


def image_footprint_from_wcs(
    cls,
    wcs: WCS,
    extra_vertices_per_edge: int = 0,
) -> SphericalPolygon:
    """create an image footprint from a GWCS object (and image shape, if no bounding box is present)

    Parameters
    ----------
    wcs: WCS :
        WCS object
    extra_vertices_per_edge: int :
        extra vertices to create on each edge to capture distortion (Default value = 0)

    Returns
    -------
    image footprint object
    """

    if (
        extra_vertices_per_edge <= 0
        and hasattr(wcs, "bounding_box")
        and wcs.bounding_box is not None
    ):
        return SphericalPolygon(wcs.footprint(center=False))
    else:
        return polygon_from_wcs(wcs, steps=extra_vertices_per_edge + 1)


def possible_intersecting_projregion_distance(footprint: SphericalPolygon) -> float:
    """maximum possible distance to the center of an intersecting projection region"""
    return (footprint.length + sc.ProjectionRegion.MAX_LENGTH) / 2.0


def possible_intersecting_skycell_distance(footprint: SphericalPolygon) -> float:
    """maximum possible distance to the center of an intersecting sky cell"""
    return (footprint.length + sc.SkyCells.length) / 2.0


def possibly_intersecting_projregions(footprint: SphericalPolygon) -> int:
    """number of possibly intersecting projection regions"""
    if footprint.area > sc.SkyCells.area:
        return (
            # the number of times the smallest projection region could fit in the image footprint
            np.ceil(footprint.area / sc.ProjectionRegion.MIN_AREA)
            # plus multiplier for partial intersections on the perimeter
            * 8
        )
    else:
        # 4 foundational intersections
        return 4


def possibly_intersecting_skycells(footprint: SphericalPolygon) -> int:
    """number of possibly intersecting skycells"""
    if footprint.area > sc.SkyCells.area:
        return (
            # number of times a skycell could fit in the image footprint
            np.ceil(footprint.area / sc.SkyCells.area)
            # plus multiplier for partial intersections on the perimeter
            * 8
        )
    else:
        # 4 foundational intersections
        return 4


def find_skycell_matches(
    image_corners: list[tuple[float, float]] | NDArray[float] | WCS,
    skymap: sc.SkyMap = None,
) -> list[int]:
    """Find sky cells overlapping the provided image footprint

    Parameters
    ----------
    image_corners : list | np.ndarray | WCS :
        Either a squence of 4 (ra, dec) pairs, or
        equivalent 2-d numpy array, or a GWCS instance.
        A GWCS instance must have `.bounding_box` or `.pixel_shape` attribute defined.
    skymap : sc.SkyMap :
        skymap instance; defaults to global SKYMAP (Default value = None)

    Returns
    -------
    Indices of all skycells (from the loaded skymap reference file) that overlap the supplied image.
    """

    if isinstance(image_corners, WCS):
        footprint = image_footprint_from_wcs(image_corners, extra_vertices_per_edge=3)
    else:
        footprint = SphericalPolygon(image_corners)

    if skymap is None:
        skymap = sc.SKYMAP

    intersecting_skycell_indices = []

    # query the global k-d tree of projection regions for possible intersection candidates in (normalized) 3D space
    nearby_projregion_indices = np.array(
        skymap.projection_regions_kdtree.query_ball_point(
            footprint.centroid.xyz,
            r=footprint.possible_intersecting_projregion_distance * 1.1,
        )
    )
    nearby_projregion_indices = nearby_projregion_indices[
        nearby_projregion_indices != len(skymap.model.projection_regions)
    ]

    for projregion_index in nearby_projregion_indices:
        projregion = sc.ProjectionRegion(projregion_index)
        if footprint.polygon.intersects(projregion.polygon):
            # query the LOCAL k-d tree of skycells for possible intersection candidates in (normalized) 3D space
            projregion_nearby_skycell_indices = np.array(
                projregion.skycells.kdtree.query_ball_point(
                    footprint.centroid.xyz,
                    r=possible_intersecting_skycell_distance(footprint) * 1.1,
                )
            )

            projregion_nearby_skycells = sc.SkyCells(
                np.array(
                    projregion_nearby_skycell_indices[
                        projregion_nearby_skycell_indices != len(projregion.skycells)
                    ]
                )
                + projregion.data["skycell_start"],
                skymap=skymap,
            )

            # find polygons that intersect the image footprint
            for skycell_index, skycell_polygon in zip(
                projregion_nearby_skycells.indices,
                projregion_nearby_skycells.polygons,
                strict=True,
            ):
                if footprint.polygon.intersects(skycell_polygon):
                    intersecting_skycell_indices.append(skycell_index)

    return intersecting_skycell_indices
