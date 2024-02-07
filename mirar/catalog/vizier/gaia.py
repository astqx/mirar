"""
Module for querying Gaia DR3 catalog
"""
import logging

import numpy as np
from astropy.table import Table

from mirar.catalog.vizier.base_vizier_catalog import VizierCatalog

class Gaia(VizierCatalog):
    """
    Gaia DR3 catalog
    """

    catalog_vizier_code = "I/355/gaiadr3"
    abbreviation = "gaiadr3"

    ra_key = "RAJ2000"
    dec_key = "DEJ2000"
    
    UBVRI_GAIA_FILTERS = {
        'U': 'BP',
        'B': 'BP',
        'V': 'G',
        'R': 'RP',
        'I': 'RP',
        'None': 'G',
    }
    
    # def filter_catalog(self, table: astropy.table.Table) -> astropy.table.Table:
    #     return super().filter_catalog(table)
