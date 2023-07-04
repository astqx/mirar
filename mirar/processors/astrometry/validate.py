"""
Module for validating astrometric solutions
"""
import logging
from collections.abc import Callable

import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.table import Table

from mirar.catalog.base_catalog import BaseCatalog
from mirar.data import Image, ImageBatch
from mirar.processors.base_catalog_xmatch_processor import BaseProcessorWithCrossMatch

logger = logging.getLogger(__name__)

# All the Sextractor parameters required for this script to run
REQUIRED_PARAMETERS = [
    "X_IMAGE",
    "Y_IMAGE",
    "FWHM_WORLD",
    "FWHM_IMAGE",
    "FLAGS",
    "ALPHAWIN_J2000",
    "DELTAWIN_J2000",
]


def get_fwhm(cleaned_img_cat: Table):
    """
    Calculate median FWHM from a ldac path
    Args:


    Returns:
    """
    mean_fwhm, med_fwhm, std_fwhm = sigma_clipped_stats(cleaned_img_cat["FWHM_WORLD"])

    mean_fwhm_pix, med_fwhm_pix, std_fwhm_pix = sigma_clipped_stats(
        cleaned_img_cat["FWHM_IMAGE"]
    )
    return med_fwhm, mean_fwhm, std_fwhm, med_fwhm_pix, mean_fwhm_pix, std_fwhm_pix


def default_sextractor_catalog_purifier(catalog: Table, image: Image) -> Table:
    """
    Default function to purify the photometric image catalog
    """
    edge_width_pixels = 100
    fwhm_threshold_arcsec = 4.0
    x_lower_limit = edge_width_pixels
    x_upper_limit = image.get_data().shape[1] - edge_width_pixels
    y_lower_limit = edge_width_pixels
    y_upper_limit = image.get_data().shape[0] - edge_width_pixels

    clean_mask = (
        (catalog["FLAGS"] == 0)
        & (catalog["FWHM_WORLD"] < fwhm_threshold_arcsec / 3600.0)
        & (catalog["X_IMAGE"] > x_lower_limit)
        & (catalog["X_IMAGE"] < x_upper_limit)
        & (catalog["Y_IMAGE"] > y_lower_limit)
        & (catalog["Y_IMAGE"] < y_upper_limit)
    )

    return catalog[clean_mask]


class AstrometryStatsWriter(BaseProcessorWithCrossMatch):
    """
    Processor to calculate astrometry statistics
    """

    def __init__(
        self,
        ref_catalog_generator: Callable[[Image], BaseCatalog],
        temp_output_sub_dir: str = "phot",
        image_photometric_catalog_purifier: Callable[
            [Table, Image], Table
        ] = default_sextractor_catalog_purifier,
        crossmatch_radius_arcsec: float = 3.0,
        write_regions: bool = False,
        cache: bool = False,
    ):
        super().__init__(
            ref_catalog_generator=ref_catalog_generator,
            temp_output_sub_dir=temp_output_sub_dir,
            crossmatch_radius_arcsec=crossmatch_radius_arcsec,
            sextractor_catalog_purifier=image_photometric_catalog_purifier,
            write_regions=write_regions,
            cache=cache,
            required_parameters=REQUIRED_PARAMETERS,
        )

    def _apply_to_images(
        self,
        batch: ImageBatch,
    ) -> ImageBatch:
        for image in batch:
            ref_cat, _, cleaned_img_cat = self.setup_catalogs(image)
            fwhm_med, _, fwhm_std, med_fwhm_pix, _, _ = get_fwhm(cleaned_img_cat)
            image["FWHM_MED"] = fwhm_med
            image["FWHM_STD"] = fwhm_std
            image["FWHM_PIX"] = med_fwhm_pix

            _, _, d2d = self.xmatch_catalogs(
                ref_cat=ref_cat,
                image_cat=cleaned_img_cat,
                crossmatch_radius_arcsec=self.crossmatch_radius_arcsec,
            )

            image.header["ASTUNC"] = np.nanmedian(d2d.value)
            image.header["ASTFIELD"] = np.arctan(
                image.header["CD1_2"] / image.header["CD1_1"]
            ) * (180 / np.pi)

        return batch