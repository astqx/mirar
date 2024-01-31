"""
Module for querying GAIA catalog
"""
import logging

import numpy as np
from astropy.table import Table

from mirar.catalog.vizier.base_vizier_catalog import VizierCatalog


class Gaia(VizierCatalog):
    """
    PanStarrs 1 catalog
    """

    catalog_vizier_code = "II/349"
    abbreviation = "ps1"

    ra_key = "RAJ2000"
    dec_key = "DEJ2000"
