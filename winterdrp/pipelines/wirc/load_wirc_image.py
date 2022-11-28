import os
import logging
import numpy as np
import astropy
from astropy.io import fits
from astropy.time import Time
from winterdrp.paths import base_name_key, raw_img_key, coadd_key, proc_history_key, proc_fail_key

logger = logging.getLogger(__name__)


def load_raw_wirc_image(
        path: str
) -> tuple[np.array, astropy.io.fits.Header]:
    with fits.open(path) as img:
        data = img[0].data
        header = img[0].header
        header["FILTER"] = header["AFT"].split("__")[0]

        if header["OBJECT"] in ["acquisition", "pointing", "focus", "none"]:
            header["OBSTYPE"] = "calibration"

        header["OBSCLASS"] = ["calibration", "science"][header["OBSTYPE"] == "object"]

        header[base_name_key] = os.path.basename(path)
        header[raw_img_key] = path
        header["TARGET"] = header["OBJECT"].lower()
        header["UTCTIME"] = header["UTSHUT"]
        header["MJD-OBS"] = Time(header['UTSHUT']).mjd
        if coadd_key not in header.keys():
            logger.debug(f"No {coadd_key} entry. Setting coadds to 1.")
            header[coadd_key] = 1

        header[proc_history_key] = ""
        header[proc_fail_key] = ""

        filter_dict = {'J': 1, 'H': 2, 'Ks': 3}

        if "FILTERID" not in header.keys():
            header["FILTERID"] = filter_dict[header["FILTER"]]
        if "FIELDID" not in header.keys():
            header["FIELDID"] = 99999
        if "PROGPI" not in header.keys():
            header["PROGPI"] = "Kasliwal"
        if "PROGID" not in header.keys():
            header["PROGID"] = 0
        if "ZP" not in header.keys():
            if "TMC_ZP" in header.keys():
                header['ZP'] = header['TMC_ZP']
                header['ZP_std'] = header['TMC_ZPSD']
            if "ZP_AUTO" in header.keys():
                header['ZP'] = header['ZP_AUTO']
                header['ZP_std'] = header['ZP_AUTO_std']
        data = data.astype(float)
        data[data == 0.] = np.nan
    return data, header