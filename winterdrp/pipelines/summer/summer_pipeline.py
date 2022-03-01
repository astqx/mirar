import os
import astropy.io.fits
import numpy as np
from astropy.io.fits import HDUList
from winterdrp.pipelines.base_pipeline import Pipeline

from winterdrp.processors.dark import DarkCalibrator
from winterdrp.processors.utils import ImageSaver
from winterdrp.processors.astromatic import SextractorRunner

wirc_flats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))


class SummerPipeline(Pipeline):

    name = "summer"

    astrometry_cal = ("GAIA", 9., 13.)
    photometry_cal = {
        "J": ()
    }

    # Set up elements to use

    header_keys = [
        "UTSHUT",
        'OBJECT',
        "FILTER",
        "EXPTIME",
        "COADDS",
    ]

    batch_split_keys = ["RAWIMAGEPATH"]

    pipeline_configurations = {
        None: [
            (DarkCalibrator,),
            # "flat",
            (ImageSaver, "preprocess"),
            (SextractorRunner, ),
            # "stack",
            # "dither"
        ]
    }

    @staticmethod
    def reformat_raw_data(
            img: HDUList,
            path: str
    ) -> [np.array, astropy.io.fits.Header]:
        header = img[0].header
        header["FILTER"] = header["AFT"].split("__")[0]
        header["OBSCLASS"] = ["calibration", "science"][header["OBSTYPE"] == "object"]
        header["CALSTEPS"] = ""
        header["BASENAME"] = os.path.basename(path)
        img[0].header = header
        return img[0].data, img[0].header

    # def apply_reduction(self, raw_image_list):
    #     return