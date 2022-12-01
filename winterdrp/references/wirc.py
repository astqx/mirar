import logging
from glob import glob

import numpy as np
from astropy.io import fits
from astropy.time import Time

from winterdrp.data import Image
from winterdrp.references.base_reference_generator import BaseReferenceGenerator

logger = logging.getLogger(__name__)


class WIRCRef(BaseReferenceGenerator):
    abbreviation = "wirc_file_lookup"

    def __init__(self, filter_name: str, object_name: str, images_directory_path: str):
        super().__init__(filter_name)
        self.filter_name = filter_name
        self.object_name = object_name
        self.images_directory_path = images_directory_path

    def get_reference(self, image: Image) -> fits.PrimaryHDU:

        header = image.get_header()

        full_imagelist = np.array(
            glob(
                f"{self.images_directory_path}/{self.object_name}/{self.object_name}_{self.filter_name}_*.fits"
            )
        )
        logger.debug(
            f"Searching for references in {self.images_directory_path}/{self.object_name}/ \
                                        {self.object_name}_{self.filter_name}_*.fits"
        )
        logger.debug(full_imagelist)
        imagelist = np.array([x for x in full_imagelist if "resamp" not in x])

        try:
            mjds = [fits.getval(x, "MJD-OBS") for x in imagelist]
        except KeyError:
            dates = Time([fits.getval(x, "DATE") for x in imagelist])
            mjds = dates.mjd

        maxind = np.argmax(mjds)
        ref_image = imagelist[maxind]

        logger.info(f"Found reference image {ref_image}")
        with fits.open(ref_image) as hdul:
            refHDU = hdul[0].copy()
        refHDU.header["ZP"] = refHDU.header["TMC_ZP"]
        return refHDU
