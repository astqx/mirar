from mirar.pipelines.wifes_autoguider.paths import *

import sys 
import os
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import astropy.constants as c
import astropy.units as u
from astropy.io import fits
from IPython.display import display
import logging
from pathlib import Path
from datetime import datetime
# import pyregion
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import WCSAxes
from astropy.coordinates import SkyCoord
import importlib
import pickle
import pandas as pd
try:
    logging.getLogger('matplotlib').disabled = True
except:
    pass
import pandas as pd
pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.max_rows', None)
import shutil
from astropy.table import QTable, Table
from matplotlib.patches import (Ellipse, Rectangle)
import matplotlib.patches as mpatches
import itertools
import shapely
import shapely.plotting
from shapely.geometry.point import Point
from shapely import affinity
from scipy.integrate import dblquad
from uncertainties import ufloat
from scipy.optimize import curve_fit
from astropy.visualization import (
    MinMaxInterval, 
    SqrtStretch,
    ImageNormalize,
    simple_norm
)
from astropy import visualization
import matplotlib.cm as cm
from pprint import pprint
import pylustrator
from astropy.stats import sigma_clipped_stats
from astropy.modeling import models
from astropy.convolution.kernels import CustomKernel
from astropy.stats import gaussian_fwhm_to_sigma
from photutils.utils._parameters import as_pair
from astropy.convolution import discretize_model
from pathlib import Path
from copy import deepcopy

# mirar
# from mirar.pipelines.wifes_autoguider.wifes_autoguider_pipeline import WifesAutoguiderPipeline
from mirar.processors.astromatic.sextractor.background_subtractor import (
    SextractorBkgSubtractor,
)
from mirar.processors.astromatic.sextractor.sextractor import Sextractor
from mirar.processors.utils import (
    CustomImageBatchModifier,
    HeaderAnnotator,
    ImageBatcher,
    ImageDebatcher,
    ImageLoader,
    ImageSaver,
    ImageSelector,
    MEFLoader,
)
from mirar.processors.utils.header_annotate import (
    HeaderEditor,
    # SextractorHeaderCorrector,
)
from mirar.data import (
    Image,
    Dataset, 
    ImageBatch,
    SourceBatch,
    SourceTable
)
from mirar.io import open_raw_image
from mirar.paths import (
    BASE_NAME_KEY,
    COADD_KEY,
    GAIN_KEY,
    LATEST_SAVE_KEY,
    LATEST_WEIGHT_SAVE_KEY,
    OBSCLASS_KEY,
    PROC_FAIL_KEY,
    PROC_HISTORY_KEY,
    RAW_IMG_KEY,
    SATURATE_KEY,
    FILTER_KEY,
    TARGET_KEY,
    TIME_KEY,
    DIFF_IMG_KEY,
    REF_IMG_KEY,
    SCI_IMG_KEY,
    XPOS_KEY,
    YPOS_KEY,
    NORM_PSFEX_KEY,
    FITS_MASK_KEY,
    core_fields,
    EXPTIME_KEY,
    get_output_dir
)
from mirar.processors.astromatic import PSFex, Scamp
# from mirar.processors.photometry.psf_photometry import SourcePSFPhotometry
# from mirar.processors.photometry.aperture_photometry import SourceAperturePhotometry
from mirar.processors.sources import (
    SourceWriter
)
# from mirar.processors.sources.source_detector import (
#     SourceGenerator
# )
from mirar.processors.base_processor import (
    BaseImageProcessor,
    BaseSourceProcessor,
    BaseSourceGenerator,
)
# from mirar.processors.photometry.base_photometry import (
#     BaseSourcePhotometry,
# )
# from mirar.utils.pipeline_visualisation import flowify
from mirar.io import (
    open_fits,
    save_to_path,
    save_fits,
)
from mirar.processors.base_processor import PrerequisiteError
from mirar.processors.utils.image_selector import select_from_images
from mirar.processors.photcal import (
    PhotCalibrator, 
    PhotometryError,
    PhotometryReferenceError,
    PhotometrySourceError,
    PhotometryCrossMatchError,
    PhotometryCalculationError,
    get_maglim,
    REQUIRED_PARAMETERS
)

from mirar.utils.ldac_tools import (
    save_table_as_ldac,
    get_table_from_ldac
)
from mirar.processors.astromatic.sextractor.sextractor import SEXTRACTOR_HEADER_KEY
from mirar.catalog.base_catalog import CatalogFromFile
from mirar.catalog.vizier.gaia import Gaia
from mirar.catalog.vizier.base_vizier_catalog import VizierCatalog
from mirar.processors.astromatic.sextractor.sextractor import sextractor_checkimg_map
from mirar.processors.photometry.utils import get_mags_from_fluxes
from mirar.errors import ProcessorError
from mirar.paths import MAGLIM_KEY, ZP_KEY, ZP_NSTARS_KEY, ZP_STD_KEY, get_output_dir
from mirar.processors.astrometry.validate import get_fwhm
from mirar.processors.base_catalog_xmatch_processor import (
    BaseProcessorWithCrossMatch,
    default_image_sextractor_catalog_purifier
)
from mirar.catalog.base_catalog import BaseCatalog
from mirar.processors.sources.source_filter import BaseSourceFilter
from mirar.processors.sources.source_table_modifier import CustomSourceTableModifier
from mirar.catalog.vizier.base_vizier_catalog import VizierCatalog
from mirar.processors.mask import MaskPixelsFromFunction
from mirar.processors.cosmic_rays import LACosmicCleaner
from mirar.processors.utils import ImageRejector
from mirar.processors.flat import (
    FlatCalibrator,
    MasterFlatCalibrator
)
from mirar.paths import get_output_dir
from mirar.processors.base_processor import (
    BaseProcessor,
    ABC
)
from mirar.data.base_data import DataBatch

# photutils
from photutils.centroids import (
    centroid_com,
    centroid_2dg
)
from photutils.psf import (
    EPSFBuilder,
    EPSFFitter,
    IterativePSFPhotometry,
    SourceGrouper,
    extract_stars
)

# photutils
from photutils.background import Background2D
from photutils.detection import DAOStarFinder
from photutils.psf import extract_stars
from photutils.psf import EPSFBuilder

import astropy
from photutils.background.interpolators import BkgZoomInterpolator
from photutils.background.core import (
    SExtractorBackground,
    StdBackgroundRMS
)
from astropy.nddata import NDData
from astropy.stats import SigmaClip
from typing import Callable
from astropy.convolution import convolve
from astropy.convolution import (
    Kernel,
    CustomKernel,
)
from photutils.segmentation import (
    make_2dgaussian_kernel,
    SourceFinder,
    SourceCatalog,
    SegmentationImage
)
from astropy.wcs import WCS
from astropy.stats import median_absolute_deviation, sigma_clip, sigma_clipped_stats
from astropy.io.fits.verify import VerifyWarning
from astropy.modeling.fitting import (
    TRFLSQFitter,
    LMLSQFitter
)
from astropy.modeling.functional_models import Gaussian2D
import warnings
from reproject import reproject_interp

logger = logging.getLogger(__name__)

sex_all_ground = CustomKernel(np.array([
    [1,2,1],
    [2,4,2],
    [1,2,1]
]))

class PrepareOutputDirectories(BaseProcessor, ABC):
    """
    Class to prepare all output directories
    """
    
    base_key = 'prepare_output_dirs'
    
    def __init__(
        self,
        output_dirs
    ):
        super().__init__()
        self.output_dirs = output_dirs
        
    def _apply(
        self, 
        batch: DataBatch
    ) -> DataBatch:
        
        for dir in self.output_dirs:
            output_dir = get_output_dir(sub_dir=self.night_sub_dir,dir_root=dir)
            output_dir.mkdir(exist_ok=True)
            
        return batch
        

def default_select_acquisition(
    images: ImageBatch,
) -> ImageBatch:
    """
    Returns images in a batch with are tagged as error

    :param images: set of images
    :return: subset of bias images
    """
    return select_from_images(images, key=OBSCLASS_KEY, target_values=ACQ_KEY)

def load_object(path):
    with open(path,'rb') as file:
        return pickle.load(file)
    
def dump_object(data,path):
    with open(path,'wb') as file:
        return pickle.dump(data,file)

def table_to_fake_image(table):
    return Image(data=np.zeros([1,1]),header=table.get_metadata())

def gaussian_fwhm_2d(std_x,std_y):
    return 2*(np.log(2)*(std_x**2 + std_y**2))**0.5

def fwhm_from_gaussian(
    table: SourceTable,
    fitted_model: Gaussian2D,
    fitter = None,
) -> SourceTable:
    pix_scale = table[PIXSCALE_KEY]

    std_x = fitted_model.x_stddev.value
    std_y = fitted_model.y_stddev.value

    fwhm_pix = gaussian_fwhm_2d(std_x,std_y)
    fwhm_deg = fwhm_pix*pix_scale 
    fwhm_arcsec = fwhm_deg*3600
    
    if std_x >= 0 and std_y >= 0:
        table[PSFMODEL_ELLIPTICITY_KEY] = 1 - min(std_x,std_y)/max(std_x,std_y)
    else:
        table[PSFMODEL_ELLIPTICITY_KEY] = str(None)
    
    table[PSFMODEL_FWHM_PIX_KEY] = fwhm_pix
    table[PSFMODEL_FWHM_KEY] = fwhm_deg
    table[PSFMODEL_FWHM_ARCSEC_KEY] = fwhm_arcsec
    table[PSFMODEL_AMP_KEY] = fitted_model.amplitude
    table[PSFMODEL_X_POS_KEY] = fitted_model.x_mean
    table[PSFMODEL_Y_POS_KEY] = fitted_model.y_mean
    table[PSFMODEL_X_STD_KEY] = fitted_model.x_stddev
    table[PSFMODEL_Y_STD_KEY] = fitted_model.y_stddev
    
    if fitter is not None:
        cov_matrix = fitter.fit_info.param_cov
        param_uncertainty = np.sqrt(np.diag(cov_matrix))
        std_x = ufloat(fitted_model.x_stddev.value,param_uncertainty[-3])
        std_y = ufloat(fitted_model.y_stddev.value,param_uncertainty[-2])
        
        fwhm_pix = gaussian_fwhm_2d(std_x,std_y)
        fwhm_deg = fwhm_pix*pix_scale 
        fwhm_arcsec = fwhm_deg*3600
        
        table[PSFMODEL_FWHM_ARCSEC_KEY] = fwhm_arcsec.n
        table[PSFMODEL_FWHM_ERR_ARCSEC_KEY] = fwhm_arcsec.s
        table[PSFMODEL_GFIT_KEY] = fitter.fit_info.optimality
        table[PSFMODEL_STATUS_KEY] = fitter.fit_info.status
    
    return table

def save_night_log_seeing(
    table: SourceTable,
    output_dir: Path | str,
    additional_cols = None,
):  
    
    log_name = 'seeing.log.csv'
    log_path = os.path.join(output_dir, log_name)
    
    data = {'NUMBER': 1}
    if os.path.exists(log_path):
        row_num = pd.read_csv(log_path)['NUMBER'].iloc[-1] + 1
        data['NUMBER'] = row_num
    
    # observation
    col_names = [
        TIME_KEY,
        TARGET_KEY,
    ]
    col_names.extend(additional_cols)
    
    # computed
    col_names.extend([
        FILTER_KEY, 
        EXPTIME_KEY, 
        XMATCH_NSTARS_KEY, 
        FWHM_MED_ARCSEC_KEY, 
        FWHM_STD_ARCSEC_KEY, 
        FWHM_PIX_KEY,
        PSFMODEL_FWHM_ARCSEC_KEY, 
        PSFMODEL_FWHM_ERR_ARCSEC_KEY, 
        PSFMODEL_FWHM_PIX_KEY, 
        PSFMODEL_ELLIPTICITY_KEY,
        PSFMODEL_GFIT_KEY,
        PSFMODEL_STATUS_KEY,
        ZP_KEY,
        ZP_MAD_KEY,
        ZP_STD_KEY,
        MAGLIM_KEY,
        MAGSYS_KEY,
        PSF_COMMENT_KEY,
        SEGM_COMMENT_KEY
    ])
    
    metadata = table.get_metadata()
    if col_names is not None:
        for col in col_names:
            if col in metadata.keys():
                data[col] = table[col]
            else:
                data[col] = None
    
    arcsec_suffix = '_ARCSEC'
    deg_to_arcsec = [FWHM_MED_KEY,FWHM_STD_KEY]
    
    for col in deg_to_arcsec:
        if col in metadata.keys() and table[col] is not None:
            data[col+arcsec_suffix] = table[col]*3600
    
    df = pd.DataFrame(data,index=[0])
    
    # df[ZP_NSTARS_KEY] = table[]
    # df[ZP_MAD] = table[]
    # df[ZP_STD] = table[]
    # df[ZP] = table[]
    
    if os.path.exists(log_path):
        df.to_csv(log_path, mode='a', header=False)
    else:
        df.to_csv(log_path)

def flipped_array(arr):
    return np.flipud(np.array(arr))

def invert_pix(mask,coords):
    mask[coords] = abs(mask[coords]-1)
    return mask

def default_scale_box_size(box_size,header):
    return box_size

def scale_boxsize(box_size,header):
    x,y = header['NAXIS1'],header['NAXIS2']
    scale = (x+y)/(1027+1072)
    return np.round(np.array(box_size)*scale).astype(int)

def segmentation_image_to_mask(
    data: SegmentationImage | np.ndarray,
    resampler = None,
    resample_custom_function = None,
    binning = None,
    bin_size_map = None,
    **kwargs
) -> np.ndarray:
    if isinstance(data,SegmentationImage):
        data = data.data
    if resampler is not None:
        data = resample_custom_function(
            image=data,
            resampler=resampler,
            binning=binning,
            bin_size_map=bin_size_map
        )
    data = data > 0
    return ~data

def get_segmentation_mask(
    image: Image | SourceTable,
    resampler = None,
    resample_custom_function = None,
    bin_size_map = None,
    # cache_dir: str | Path,
) -> np.ndarray:
    
    if isinstance(image,Image):
        header = image.header
    elif isinstance(image,SourceTable):
        header = image.get_metadata()
        
    if SEGMOBJ_KEY in header.keys():
        segm = load_object(header[SEGMOBJ_KEY])
        binning = tuple(map(int,header['CCDSUM'].split(' ')))
        return segmentation_image_to_mask(
            segm,
            resampler=resampler,
            resample_custom_function=resample_custom_function,
            binning=binning,
            bin_size_map=bin_size_map
        )
    else:
        logger.info(f"segmentation image does not exist for {image[BASE_NAME_KEY]}")
        return np.zeros(image.get_data().shape).astype(bool)

def invert_adjacent_to_false(arr,radius):
    # source: OpenAI
    #   create a python function that takes an 2d numpy array 
    #   of bools and inverts True values that are next to False values
    
    # Ensure the array is a numpy array of bools
    if not isinstance(arr, np.ndarray) or arr.dtype != bool:
        raise ValueError("Input must be a 2D numpy array of bools")

    # Create a padded version of the array to handle edge cases
    padded_arr = np.pad(arr, pad_width=radius, mode='constant', constant_values=True)

    # Create an empty array of the same size to store the output
    output_arr = np.empty_like(arr)

    # Iterate through the original array
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            # If any adjacent cell is False, invert the current cell
            if not np.all(padded_arr[i:i+(1+radius*2), j:j+(1+radius*2)]):
                output_arr[i, j] = False
            else:
                output_arr[i, j] = True

    return output_arr

def get_extended_mask(
    image: Image | SourceTable,
    base_mask_function,
    extension_mask_function,
    *args,
    extension_mask_function_kwargs = None,
    **kwargs
) -> np.ndarray:
    
    mask = base_mask_function(image,*args,**kwargs)
    mask_extended = extension_mask_function(mask,**extension_mask_function_kwargs)
    return ~mask_extended

def make_imshow_params(data):
    mean, std = np.nanmean(data), np.nanstd(data)
    vmin = mean - std
    vmax = mean + 10 * std
    return {
        'interpolation': "nearest",
        'cmap': "gray",
        'vmin': vmin,
        'vmax': vmax,
        'origin': "lower",
    }

central_mask_matrix = np.array([
    [0.412,0.435], # (bottom left)
    [0.595,0.567]  # (top right)
])

central_mask_matrix_hires = np.array([
    [0.413,0.426],
    [0.596,0.560]
])

default_central_mask_matrix = np.array([
    [0.413,0.426],
    [0.596,0.567]
])

def make_bad_pix_mask(
    image: Image,
    # image: np.ndarray,
    central_mask_matrix: np.ndarray = default_central_mask_matrix
) -> np.ndarray:
    size = image.get_data().shape
    # size = image.shape
    bad_pix_mask = np.ones(size)

    center = np.array([size[0]//2,size[1]//2])

    central_mask_pos_factor = np.array([
        flipped_array(central_mask_matrix[0]),
        flipped_array(central_mask_matrix[1])
    ])

    central_mask_pos = np.round(size * central_mask_pos_factor).astype(int)

    bad_pix_mask[
        central_mask_pos[0,0]:central_mask_pos[1,0],
        central_mask_pos[0,1]:central_mask_pos[1,1]
    ] = 0
    
    return ~bad_pix_mask.astype(bool)

def correct_mask(mask):
    return ~(mask-np.min(mask)).astype(bool)

class ReuseVisierXMatch(VizierCatalog):
    
    abbreviation = "reuse_catalog"
    
    def __init__(
        self,
        catalog_path: Path | str,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.catalog_path = catalog_path
    
    def get_catalog(self, ra_deg: float, dec_deg: float) -> Table:
        
        return get_table_from_ldac(self.catalog_path)

class RemoveEmptySourceTables(BaseSourceFilter):
    
    def _apply_to_sources(
        self, 
        batch: SourceBatch
    ) -> SourceBatch:
        
        batch_updated = SourceBatch([table for table in batch if len(table) > 0])
        return batch_updated

class WifesAutoguiderVisier():
    
    def __init__(
        self,
        visier_catalog: VizierCatalog,
        cache: bool = False,
        reuse_cached_catalog: bool = False,
        reuse_cache_dir: Path | str = None,
    ):
        self.visier_catalog = visier_catalog
        self.cache = cache
        self.reuse_cached_catalog = reuse_cached_catalog
        self.reuse_cache_dir = reuse_cache_dir
        
    def generator(
        self,
        image: Image,
        min_mag = 10,
        max_mag = 20,
    ) -> VizierCatalog | CatalogFromFile:
        
        logger.debug(image)
        filter_name = image[FILTER_KEY]

        search_radius_arcmin = (
            np.max([image["NAXIS1"], image["NAXIS2"]])
            * np.max([np.abs(image["CD1_1"]), np.abs(image["CD1_2"])])
            * 60
        ) / 2.0
        
        # match closest to UBVRI
        if self.visier_catalog.abbreviation == 'gaiadr3':
            filter_name = self.visier_catalog.UBVRI_GAIA_FILTERS[filter_name]

        if not self.reuse_cached_catalog:
            return self.visier_catalog(
                min_mag=min_mag,
                max_mag=max_mag,
                search_radius_arcmin=search_radius_arcmin,
                filter_name=filter_name,
                cache_catalog_locally=self.cache,
            )
        else:
            # TODO: make getting the path more robust, check relevant get_output_dir
            return ReuseVisierXMatch(
                min_mag=min_mag,
                max_mag=max_mag,
                search_radius_arcmin=search_radius_arcmin,
                filter_name=filter_name,
                catalog_path = os.path.join(
                    self.reuse_cache_dir,
                    image[BASE_NAME_KEY].replace(
                        '.fits',f".{self.visier_catalog.abbreviation}.cat"
                    )
                )
            )

def wifes_autoguider_photometric_catalog_purifier(
    catalog: Table,
    image: Image
) -> Table:
    logger.debug('Using filter: wifes_autoguider_photometric_catalog_purifier')
    # TODO: filter
    # clean_mask = np.ones(catalog.to_pandas().shape,dtype=bool)
    # return catalog[clean_mask]
    
    
    return catalog

def get_mask(default_mask,header):
    """
    Retrieve mask from header if mask is set to None and FITS_MASK_KEY exists
    """
    if default_mask is None and FITS_MASK_KEY in header:
        mask = correct_mask(
            fits.getdata(header[FITS_MASK_KEY])
        )
    else:
        return default_mask
    return mask

class PhotutilsBkgSubtractor(BaseImageProcessor):
    
    base_key = "photutilsbkgsubtractor"
    
    def __init__(
        self,
        box_size = 40,
        box_size_scale_function = default_scale_box_size,
        mask=None, 
        coverage_mask=None, 
        coverage_mask_as_mask: bool = False,
        fill_value=0.0, 
        exclude_percentile=10.0, 
        filter_size=(3, 3), 
        filter_threshold=None, 
        edge_method='pad', 
        sigma_clip=SigmaClip(
            sigma=3.0, 
            sigma_lower=3.0, 
            sigma_upper=3.0,
            maxiters=10, 
            cenfunc='median', 
            stdfunc='std', 
            grow=False
        ), 
        bkg_estimator=SExtractorBackground(sigma_clip=None), 
        bkgrms_estimator=StdBackgroundRMS(sigma_clip=None),
        interpolator=BkgZoomInterpolator(),
        output_sub_dir = 'background',
        select_images: Callable[[ImageBatch], ImageBatch] = default_select_acquisition, #change
        dev: bool = False,
        cache: bool = False,
        save_bkg: bool = True,
        save_bkg_rms: bool = True,
        bzero_correction: bool = False,
    ):
        super().__init__()
        self.box_size = box_size
        self.mask = mask
        self.coverage_mask = coverage_mask
        self.fill_value = fill_value
        self.exclude_percentile = exclude_percentile
        self.filter_size = filter_size
        self.filter_threshold = filter_threshold
        self.edge_method = edge_method
        self.sigma_clip = sigma_clip
        self.bkg_estimator = bkg_estimator
        self.bkgrms_estimator = bkgrms_estimator
        self.interpolator = interpolator
        self.cache = cache
        self.output_sub_dir = output_sub_dir
        self.dev = dev
        self.select_images = select_images
        self.save_bkg = save_bkg
        self.save_bkg_rms = save_bkg_rms
        self.coverage_mask_as_mask = coverage_mask_as_mask
        self.box_size_scale_function = box_size_scale_function
        self.bzero_correction = bzero_correction
    
    def _apply_to_images(
        self,
        batch: ImageBatch,
    ) -> ImageBatch:
        
        images = self.select_images(batch)
        
        for image in images:
            data = image.get_data()
            header = image.get_header()      
            
            if self.coverage_mask_as_mask:
                coverage_mask = get_mask(self.mask,header)
            else:
                coverage_mask = self.coverage_mask     
            
            background = Background2D(
                data=data,
                box_size=self.box_size_scale_function(self.box_size,header),
                mask = get_mask(self.mask,header),
                coverage_mask = coverage_mask,
                fill_value = self.fill_value,
                exclude_percentile = self.exclude_percentile,
                filter_size = self.filter_size,
                filter_threshold = self.filter_threshold,
                edge_method = self.edge_method,
                sigma_clip = self.sigma_clip,
                bkg_estimator = self.bkg_estimator,
                bkgrms_estimator = self.bkgrms_estimator,
                interpolator = self.interpolator
            )
            background_map = background.background
            
            header[BGMED_KEY] = background.background_median
            header[BGRMSMED_KEY] = background.background_rms_median
            
            bkgsub = data - background_map
            image.set_data(bkgsub)
            
            save_images = {}
            output_dir = get_output_dir(self.output_sub_dir, self.night_sub_dir)
            
            if self.save_bkg:
                save_images[BGPATH_KEY] = 'background'
            
            if self.save_bkg_rms:
                save_images[BGRMSPATH_KEY] = 'background_rms'
            
            # TODO: 
            if self.cache:
                save_images += ['background']
                # bkg_image_name = image[BASE_NAME_KEY].replace('fits','background.fits')
                bkg_image_name = image[BASE_NAME_KEY]+'.background'
                header[BGPATH_KEY] = str(output_dir.joinpath(bkg_image_name))
                
            if self.dev:
                save_images[sextractor_checkimg_map['MINIBACKGROUND']] = 'background_mesh'
                save_images[sextractor_checkimg_map['MINIBACK_RMS']] = 'background_rms_mesh'
                save_name = image[BASE_NAME_KEY].replace('fits','background.pkl')
                dump_object(
                    data=background,
                    path=output_dir.joinpath(save_name)
                )
                
            for im in save_images.keys():
                # save_name = image[BASE_NAME_KEY].replace('fits',im+'.fits')
                save_name = image[BASE_NAME_KEY]+f".{save_images[im]}"
                save_path = output_dir.joinpath(save_name)
                
                save_data=eval('background.'+save_images[im])
                if not self.bzero_correction: # TODO: also include scale  
                    save_to_path(
                        data=save_data,
                        header=image.header,
                        path=save_path,
                        overwrite=True
                    )
                else:
                    safe_safe_fits(Image(save_data,image.header),save_path)
                image[im] = str(save_path)
            
            image.set_header(header)
        
        return batch

class PhotutilsSourceFinder(BaseImageProcessor):
    """
    Processor to detect sources using photutils.segmentation.SourceFinder
    
    Args
        convolution_fwhm: FWHM of convolution mask
        convolution_kernel_size: size of convolution kernel
        npixels: number of connected pixels for detection
        threshold_factor: threshold factor of background RMS median
        connectivity: {4,8} source pixel grouping
        deblend: Whether to deblend overlapping sources.
        nlevels: The number of multi-thresholding levels to use for deblending.
        contrast: The fraction of the total source flux that a local peak must have 
            (at any one of the multi-thresholds) to be deblended as a separate object.
        mode: The mode used in defining the spacing between the multi-thresholding levels.
        relabel: If True (default), then the segmentation image will be relabeled after deblending
        nproc: The number of processes to use for multiprocessing (deblending)
        progress_bar: Whether to display a progress bar. 
        
    Returns
        ImageBatch
    """
    
    base_key = 'photutilssourcedetection'
    
    def __init__(
        self,
        mask = None,
        output_sub_dir: Path | str = 'detection',
        convolve: bool = False,
        dev_params = None,
        convolution_kernel: Kernel | None = None,
        convolution_fwhm: float | None = 2, 
        convolution_kernel_size: int | None = 3,
        npixels: int = 10, # default from SE config
        threshold_factor: float = 1.5, 
        connectivity: int = 8, # default from SE config
        deblend: bool = True,
        nlevels: int = 32, # default from SE config
        contrast: float = 0.001,
        mode: str = 'exponential', # default from SE config
        relabel: bool = True,
        nproc: int = 1,
        progress_bar: bool = False,
        dev: bool = False,
        cache: bool = False,
    ):
        super().__init__()
        self.cache = cache
        self.output_sub_dir = output_sub_dir
        self.npixels = npixels
        self.threshold_factor = threshold_factor
        self.connectivity = connectivity
        self.convolution_fwhm = convolution_fwhm
        self.convolution_kernel_size = convolution_kernel_size
        self.deblend = deblend
        self.contrast = contrast
        self.nlevels = nlevels
        self.mode = mode
        self.relabel = relabel
        self.nproc = nproc
        self.progress_bar = progress_bar
        self.dev = dev
        self.dev_params = dev_params,
        self.convolve = convolve
        self.convolution_kernel = convolution_kernel
        self.mask = mask
    
    def _apply_to_images(
        self,
        batch: ImageBatch,
    ) -> ImageBatch:
        
        batch_updated = SourceBatch()
        
        for image in batch:
            data = image.get_data()
            header = image.get_header()
    
            # convolve the data
            if self.convolve:
                if self.convolution_kernel is not None:
                    kernel = self.convolution_kernel
                else:
                    kernel = make_2dgaussian_kernel(
                        self.convolution_fwhm, 
                        size=self.convolution_kernel_size
                    )  
                convolved_data = convolve(data, kernel)
            else:
                convolved_data = data
            
            # detect the sources
            if BGRMSMED_KEY not in header.keys():
                raise PrerequisiteError(
                    f"{BGRMSMED_KEY} key not found in image. "
                    f"PhotutilsBkgSubtractor must be run before running this processor"
                )
            threshold = self.threshold_factor * image[BGRMSMED_KEY]  # per-pixel threshold
            finder = SourceFinder(
                npixels=self.npixels, 
                connectivity=self.connectivity,
                deblend = self.deblend,
                nlevels = self.nlevels,
                contrast = self.contrast,
                mode = self.mode,
                relabel = self.relabel,
                nproc=self.nproc,
                progress_bar=self.progress_bar,
            )
            segm = finder(convolved_data, threshold, get_mask(self.mask,header))
            
            if segm is None:
                # TODO: add logger message
                continue
            
            output_dir = get_output_dir(self.output_sub_dir, self.night_sub_dir)
            save_name = image[BASE_NAME_KEY]+'.segm'
            save_path = output_dir.joinpath(save_name)
            save_path_obj = str(save_path)+'.pkl' # maybe a different way?
            
            header[SEGMOBJ_KEY] = str(save_path_obj)
            header[SEGMPATH_KEY] = str(save_path)
            
            if self.convolve:
                save_path_conv = str(save_path).replace('segm','conv')
                header[CONVPATH_KEY] = str(save_path_conv)
            else:
                header[CONVPATH_KEY] = str(None)
            
            with open(save_path_obj, 'wb') as file:
                pickle.dump(segm, file) # better format?
            save_to_path(         # maybe move to dev as need only object not image
                data=segm.data,
                header=header,
                path=save_path,
                overwrite=True
            )
            save_to_path(
                data=convolved_data,
                header=header,
                path=save_path_conv,
                overwrite=True
            )
            
            if self.dev:
                if self.dev_params is None:
                    params = ['cmap','polygons','segments','areas']
                else:
                    params = self.dev_params[0]
                for param in params:
                    with open(f"{save_path}.{param}.pkl", 'wb') as file:
                        pickle.dump(getattr(segm, param),file)

            image.set_header(header)
            
        return batch

def get_binning(header):
    return tuple(map(int,header['CCDSUM'].split(' ')))

#TODO: multiple aperture
class PhotutilsSourceCatalog(BaseSourceGenerator):
    """
    Args
        localbkg_width: The width of the rectangular annulus used to 
            compute a local background around each source.
        detection_cat: A SourceCatalog object for the detection image.
        make_cutouts: Whether to make cutouts for psf modeling
        cutout_size: size of cutout (if make_cutouts)
    
    Returns

    """
    
    base_key = "photutilssourcecatalog"
    
    def __init__(
        self, 
        calc_total_error: bool = False,
        error = None, # TODO: [somewhat done] maybe PhotutilsTotalErrorCalculator or calc_total_error internally
        mask = None,
        wcs = None, 
        localbkg_width = 15, # default: 0, mirar aper phot default: 15
        background = None,
        use_background = False,
        apermask_method = 'correct', # default from SE config
        kron_params = [2.5, 1.5], # SE default # [2.5, 3.5] mirar default from SE config
        detection_cat = None,
        progress_bar: bool = False,
        make_psf_cutouts: bool = True,
        psf_cutout_size: int = 21,
        output_sub_dir: str = "detection",
        copy_image_keywords: str | list[str] = None,
        cache: bool = False,
        update_seeing: bool = False,
        binning_correction: bool = False,
    ):    
        super().__init__()
        self.output_sub_dir = output_sub_dir
        self.copy_image_keywords = copy_image_keywords
        if isinstance(copy_image_keywords, str):
            self.copy_image_keywords = [self.copy_image_keywords]
        self.cache=cache
        self.error = error
        self.mask = mask
        self.wcs = wcs 
        self.localbkg_width = localbkg_width
        self.apermask_method = apermask_method
        self.kron_params = kron_params
        self.detection_cat = detection_cat
        self.progress_bar = progress_bar
        self.make_psf_cutouts = make_psf_cutouts
        self.psf_cutout_size = psf_cutout_size
        self.calc_total_error = calc_total_error
        self.background = background
        self.use_background = use_background
        self.update_seeing = update_seeing
        self.binning_correction = binning_correction
        
    def failed_log(
        self, 
        image: Image, 
        comment = None
    ):
        if self.update_seeing:
            image[SEGM_COMMENT_KEY] = comment
            
    def get_binning(
        self,
        header
    ):
        return get_binning(header)
        
    def _apply_to_images(
        self,
        batch: ImageBatch,
    ) -> SourceBatch:
        
        src_batch = SourceBatch()
        
        for image in batch:
            
            data = image.get_data()
            header = image.get_header()
            
            # TODO: check pre-rec or as a processor function
            if SEGMOBJ_KEY not in header:
                self.failed_log(image,'No sources detected')
                logger.error(f"No sources detected in {image[BASE_NAME_KEY]}")
                df = pd.DataFrame({'failed': ['no sources','None']})
                src_batch.append(SourceTable(df,metadata=header))
                continue
               
            with open(header[SEGMOBJ_KEY],'rb') as file:
                segment_img = pickle.load(file)

            if self.calc_total_error: 
                if LATEST_WEIGHT_SAVE_KEY in header:
                    error = fits.getdata(image[LATEST_WEIGHT_SAVE_KEY])
                else:
                    error = None
            else:
                error = self.error
            
            if self.use_background:
                if self.background is not None:
                    background = fits.getdata(header[BGPATH_KEY]) # [!imp] restimate?
                else:
                    background = self.background
            else:
                background = None
                
            if self.wcs is None:
                wcs = WCS(header=header)
            else:
                wcs = self.wcs
                
            if self.binning_correction:
                localbkg_width = np.round(
                    self.localbkg_width/self.get_binning(header)[0]
                )
            else:
                localbkg_width = self.localbkg_width
            
            srccat = SourceCatalog(
                data=data,
                segment_img=segment_img,
                convolved_data=fits.getdata(header[CONVPATH_KEY]),
                error=error,
                mask=get_mask(self.mask,header),
                background=background,
                wcs=wcs,
                localbkg_width=localbkg_width,
                apermask_method=self.apermask_method,
                kron_params = self.kron_params,
                detection_cat = self.detection_cat,
                progress_bar = self.progress_bar,
            )
            srccat_table = srccat.to_table()
            
            pix_scale = np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix)))
            # TODO: maybe rename col instead of duplicate
            #       or make a table from scratch with only necessary columns
            srccat_table['NUMBER'] = srccat.label
            srccat_table['fwhm'] = srccat.fwhm
            srccat_table['ellipticity'] = srccat.ellipticity
            srccat_table['elong'] = srccat.elongation
            srccat_table[XPOS_KEY] = srccat.xcentroid
            srccat_table[YPOS_KEY] = srccat.ycentroid
            srccat_table['xcentroid_win'] = srccat.xcentroid_win
            srccat_table['ycentroid_win'] = srccat.ycentroid_win
            srccat_table['sky_centroid_icrs'] = srccat.sky_centroid_icrs
            srccat_table['sky_centroid_win'] = srccat.sky_centroid_win
            # logger.debug(srccat.sky_centroid_win)
            # cov_eigvals = srccat.covariance_eigvals
            # srccat_table['cov_eigvals'] = cov_eigvals
            # srccat_table['aimage'] = cov_eigvals[0]
            # srccat_table['bimage'] = cov_eigvals[1]
            srccat_table['aimage'] = srccat.semimajor_sigma
            srccat_table['bimage'] = srccat.semiminor_sigma
            srccat_table['THETA_IMAGE'] = srccat.orientation
            srccat_table['kron_aperture'] = srccat.kron_aperture
            
            # sex compatability
            srccat_table['ALPHAWIN_J2000'] = [coords.ra for coords in srccat.sky_centroid_win]
            srccat_table['DELTAWIN_J2000'] = [coords.dec for coords in srccat.sky_centroid_win]
            srccat_table['X_IMAGE'] = srccat.xcentroid_win
            srccat_table['Y_IMAGE'] = srccat.ycentroid_win
            srccat_table['FWHM_IMAGE'] = srccat.fwhm
            srccat_table['FWHM_WORLD'] = srccat.fwhm * pix_scale
            
            mag, mag_unc = get_mags_from_fluxes(
                flux_list = srccat.kron_flux,
                fluxunc_list = np.zeros(len(srccat_table)),
                zeropoint = 0.0,
                zeropoint_unc = 0.0,
            )
            srccat_table['MAG_AUTO'] = np.array(mag,dtype=float)
            
            # TODO: compute this somehow 
            aper_mags = {
            
            }
        
            if len(aper_mags) > 0:
                aper_fluxes = np.array(list(aper_mags.values()))
                mag, mag_unc = get_mags_from_fluxes(
                    flux_list = aper_fluxes,
                    fluxunc_list = np.zeros(aper_fluxes.shape),
                    zeropoint = 0.0,
                    zeropoint_unc = 0.0,
                )
                srccat_table['MAG_APER'] = np.array(mag,dtype=float)
            
            # TODO: CHANGE!!
            srccat_table['FLAGS'] = 0
            
            # mirar compatability
            srccat_table[DIFF_IMG_KEY] = image[LATEST_SAVE_KEY] # has to be 
            srccat_table[SCI_IMG_KEY] = '' #image[LATEST_SAVE_KEY]
            srccat_table[REF_IMG_KEY] = ''
    
            if self.make_psf_cutouts:
                # TODO: check if/what different for dithers with wcs
                
                # twiddle -->
                twiddle_keys = {
                    'label': 'id',
                    'xcentroid': 'x',
                    'ycentroid': 'y'
                }
                for key in twiddle_keys.keys():
                    srccat_table.rename_column(key,twiddle_keys[key])
                    
                stars = extract_stars( 
                    data=NDData(data=data),
                    catalogs=srccat_table[*twiddle_keys.values()],
                    size=self.psf_cutout_size
                )
                
                # twiddle <--
                for key in twiddle_keys.keys():
                    srccat_table.rename_column(twiddle_keys[key],key)
                
                save_name = image[BASE_NAME_KEY]+'.cutouts.pkl'
                output_dir = get_output_dir(self.output_sub_dir, self.night_sub_dir)
                save_path = output_dir.joinpath(save_name)
                with open(save_path,'wb') as file:
                    pickle.dump(stars,file)
                
                header[PSF_CUTOUTS_PATH_KEY] = str(save_path)
                header[PSF_CUTOUTS_SIZE_KEY] = self.psf_cutout_size
            
            output_dir = get_output_dir(self.output_sub_dir, self.night_sub_dir)
            output_cat = output_dir.joinpath(
                image[BASE_NAME_KEY].replace(".fits", ".cat")
            )
            
            header[SEXTRACTOR_HEADER_KEY] = str(output_cat)
            # header[CALSTEPS] += Sextractor.base_key
            save_table_as_ldac(
                tbl = srccat_table,
                file_path = output_cat
            )
            
            image.set_header(header)
            
            metadata = {}
            for key in image.keys():
                if key != "COMMENT":
                    metadata[key] = image[key]
                    
            metadata[PIXSCALE_KEY] = pix_scale
            
            if len(aper_mags) > 0:
                srccat_table['MAG_APER'] = Table(aper_mags)
            src_batch.append(SourceTable(srccat_table.to_pandas(),metadata=metadata))
            
        return src_batch

class SourceCrossMatch(BaseProcessorWithCrossMatch):
    
    def __init__(
        self,
        ref_catalog_generator: Callable[[Image], BaseCatalog],
        temp_output_sub_dir: str = "phot",
        image_photometric_catalog_purifier: Callable[
            [Table, Image], Table
        ] = default_image_sextractor_catalog_purifier,
        crossmatch_radius_arcsec: float = 1.0,
        write_regions: bool = False,
        cache: bool = False,
        ref_cat_cols = None,
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
        self.ref_cat_cols = ref_cat_cols
        
    def check_prerequisites(self):
        return True
        
    def _apply_to_images(
        self,
        batch: SourceBatch
    ) -> SourceBatch:
        
        batch_updated = SourceBatch()
        
        for table in batch:
            
            if not SEXTRACTOR_HEADER_KEY in table.get_metadata().keys():
                batch_updated.append(table)
                logger.warning(f"{SEXTRACTOR_HEADER_KEY} not found in {table[BASE_NAME_KEY]}")
                continue
            
            ref_cat, _, cleaned_img_cat = self.setup_catalogs(
                table_to_fake_image(table)
            )
            matched_img_cat, matched_ref_cat, _ = self.xmatch_catalogs(
                ref_cat=ref_cat,
                image_cat=cleaned_img_cat,
                crossmatch_radius_arcsec=self.crossmatch_radius_arcsec, 
            )
            
            for cols in self.ref_cat_cols:
                matched_img_cat[cols] = matched_ref_cat[cols]
            
            matched_img_table = SourceTable(
                matched_img_cat.to_pandas(),
                metadata=table.metadata
            )
            
            nan_mask = ~np.isnan(matched_img_cat['label'])
            matched_img_table[XMATCH_NSTARS_KEY] = len(matched_img_cat[nan_mask])
            
            batch_updated.append(matched_img_table)
            
        return batch_updated

class PhotutilsModelPSF(BaseSourceProcessor):
    
    base_key = 'photutils_model_psf'
    
    def __init__(
        self,
        oversampling=4, # don't oversample unless necessary*
        shape=None, # needs to be checked (edge pixles)
        shape_from_cutout: bool = False,
        smoothing_kernel='quartic', 
        recentering_func=centroid_com, 
        recentering_maxiters=20, 
        fitter=LMLSQFitter,
        fitter_params=dict(),
        fitter_kwargs=dict(),
        maxiters=10, 
        norm_radius=None, #5.5, 
        recentering_boxsize=None, # (5, 5), 
        center_accuracy=0.001, 
        sigma_clip=SigmaClip(
            sigma=3, 
            sigma_lower=3, 
            sigma_upper=3, 
            maxiters=10, 
            cenfunc='median', 
            stdfunc='std', 
            grow=False
        ),
        progress_bar=False, 
        fit_boxsize=5, # EPSFFitter default 
        psf_cutout_size: int = 21, 
        init_cutout_size: int = 29,
        min_pixels: int = 15,
        psf_cutout_size_auto: bool = False,
        make_psf_cutouts_from_catalog: bool = False,
        make_psf_cutouts: bool = True, # TODO: implement star detection/local peak finding to detect point-like objects specifically
        dev = False,
        output_sub_dir: Path | str = 'psf_model',
        update_seeing: bool = False,
        cache: bool = False,
    ):
        super().__init__()
        self.cache=cache
        self.output_sub_dir = output_sub_dir
        self.oversampling=oversampling
        self.shape=shape
        self.smoothing_kernel=smoothing_kernel
        self.recentering_func=recentering_func
        self.recentering_maxiters=recentering_maxiters
        self.fitter=fitter
        self.fitter_params=fitter_params
        self.fitter_kwargs=fitter_kwargs
        self.maxiters=maxiters
        self.norm_radius=norm_radius
        self.recentering_boxsize=recentering_boxsize
        self.center_accuracy=center_accuracy
        self.sigma_clip=sigma_clip
        self.progress_bar=progress_bar
        self.fit_boxsize=fit_boxsize
        self.dev = dev
        self.psf_cutout_size = psf_cutout_size
        self.make_psf_cutouts = make_psf_cutouts
        self.make_psf_cutouts_from_catalog = make_psf_cutouts_from_catalog
        self.psf_cutout_size_auto = psf_cutout_size_auto
        self.update_seeing = update_seeing
        self.shape_from_cutout = shape_from_cutout
        self.init_cutout_size = init_cutout_size
        self.min_pixels = min_pixels
        
    def failed_log(
        self, 
        table: SourceTable, 
        comment = None
    ):
        if self.update_seeing:
            table[PSF_COMMENT_KEY] = comment
    
    # source: FLOWS
    def star_size(self, fwhm) -> int:
        # Make cutouts of stars using extract_stars:
        # Scales with FWHM
        size = int(np.round(self.init_cutout_size * fwhm / 6))
        size += 0 if size % 2 else 1  # Make sure it's odd
        size = max(size, self.min_pixels)  # Never go below 15 pixels
        return size
        
    def _apply_to_sources(
        self,
        batch: SourceBatch,
    ) -> SourceBatch:
        
        batch_updated = SourceBatch()
        
        for table in batch:
            
            data = table.get_data()
            metadata = table.get_metadata()
            image_data = fits.getdata(metadata[LATEST_SAVE_KEY])
            
            if len(table) == 0 or 'fwhm' not in data.columns:
                err = f"{metadata[BASE_NAME_KEY]} has no sources"
                self.failed_log(table,comment='no sources')
                logger.warning(err)
                batch_updated.append(table)
                continue
            
            suffix = '.cutouts.pkl'
            
            fwhm_avg, fwhm_med, fwhm_std = sigma_clipped_stats(data['fwhm'])
            
            if self.psf_cutout_size_auto:
                # psf_cutout_size = int(np.round(fwhm_med) + 2*fwhm_std)*3
                # psf_cutout_size += int(psf_cutout_size%2 - 1)
                psf_cutout_size = self.star_size(fwhm_med)
            else:
                psf_cutout_size = self.psf_cutout_size
            
            # update to use PSF_CUTOUTS_PATH_KEY
            # stars_file = Path(table[LATEST_SAVE_KEY]).parent.joinpath(table[BASE_NAME_KEY]+suffix)
            if PSF_CUTOUTS_PATH_KEY in metadata:
                stars_file = metadata[PSF_CUTOUTS_PATH_KEY]
                with open(stars_file,'rb') as file:
                    stars = pickle.load(file)
            
            elif self.make_psf_cutouts_from_catalog:
                twiddle_keys = {
                    'label': 'id',
                    'xcentroid': 'x',
                    'ycentroid': 'y'
                }
                # data.rename(twiddle_keys, axis='columns')
                
                stars_catalog = Table.from_pandas(data[list(twiddle_keys.keys())])
                for key in twiddle_keys.keys():
                    stars_catalog.rename_column(key,twiddle_keys[key])
                
                stars = extract_stars( 
                    data=NDData(data=image_data),
                    # catalogs=data[*twiddle_keys.values()],
                    catalogs=stars_catalog,
                    size=psf_cutout_size
                )
                
                # twiddle <--
                # data.rename({v: k for k, v in twiddle_keys.items()}, axis='columns')
                
                save_name = metadata[BASE_NAME_KEY]+'.cutouts.pkl'
                output_dir = get_output_dir(self.output_sub_dir, self.night_sub_dir)
                save_path = output_dir.joinpath(save_name)
                with open(save_path,'wb') as file:
                    pickle.dump(stars,file)
                
                table[PSF_CUTOUTS_PATH_KEY] = str(save_path)
                table[PSF_CUTOUTS_SIZE_KEY] = psf_cutout_size
        
            if self.fit_boxsize is None:
                # fit_boxsize = int(np.round(fwhm_med))
                fit_boxsize = max(int(np.round(1.5 * fwhm_med)), 5)
            else:
                fit_boxsize = self.fit_boxsize
                
            fitter = EPSFFitter(
                fitter=self.fitter(**self.fitter_params),
                fit_boxsize=fit_boxsize,
                **self.fitter_kwargs
            )
            
            if self.shape_from_cutout:
                shape = psf_cutout_size
            else:
                shape = self.shape
                
            if self.norm_radius is None:
                norm_radius = max(fwhm_med, 5)
            else:
                norm_radius = self.norm_radius
                
            if self.recentering_boxsize is None:
                recentering_boxsize = max(int(np.round(2 * fwhm_med)), 5)
                if recentering_boxsize % 2 == 0:
                    recentering_boxsize += 1
            else:
                recentering_boxsize = self. recentering_boxsize
            
            epsf_builder = EPSFBuilder(
                oversampling=self.oversampling,
                shape=shape,
                smoothing_kernel=self.smoothing_kernel,
                recentering_func=self.recentering_func,
                recentering_maxiters=self.recentering_maxiters,
                fitter=fitter,
                maxiters=self.maxiters,
                norm_radius=norm_radius,
                recentering_boxsize=recentering_boxsize,
                center_accuracy=self.center_accuracy,
                sigma_clip=self.sigma_clip,
                progress_bar=self.progress_bar
            )
            
            try:
                epsf, fitted_stars = epsf_builder(stars)
            except Exception as e:
                self.failed_log(table, comment=f"psf modelling failed: {type(e).__name__}")
                logger.error(e,f"in {table[BASE_NAME_KEY]}")
                batch_updated.append(table)
                continue
            
            output_dir = get_output_dir(self.output_sub_dir, self.night_sub_dir)
            epsf_save_path = output_dir.joinpath(table[BASE_NAME_KEY].replace('fits','psfmodel.pkl'))
            with open(epsf_save_path,'wb') as file:
                pickle.dump(epsf, file)
                
            table[NPSFPATH_KEY] = str(epsf_save_path) # save fits to this header and create a new header for object
            table[OVERSAMPLE_KEY] = self.oversampling
            
            if self.dev:
                # ? dev - maybe move to main
                data['psf_center_flat'] = np.full(data.shape[0],np.nan)
                for i,star in enumerate(fitted_stars.all_good_stars):
                    data.at[star.id_label-1,'psf_center_flat'] = str(fitted_stars.center_flat[i])
                
                # dev
                epsf_img_save_path = str(epsf_save_path).replace('pkl','fits')
                save_to_path(
                    data=epsf.data,
                    header=None,
                    path=epsf_img_save_path,
                    overwrite=True
                )
                
                save_suffix = suffix.replace('cutouts','cutouts.fitted')
                save_path = output_dir.joinpath(table[BASE_NAME_KEY]+save_suffix)
                with open(save_path,'wb') as file:
                    pickle.dump(fitted_stars, file)
            
            table.set_data(data)
            batch_updated.append(table)
            
        return batch_updated

def wifes_autoguider_gal_filter(
    batch: SourceBatch
) -> SourceBatch:
    batch_updated = SourceBatch()
    
    for table in batch:
        data = table.get_data()
        data = data[data['Gal'] == 0]
        batch_updated.append(SourceTable(data,metadata=table.metadata))
    
    return batch_updated  

class SourcePhotCalibrator(PhotCalibrator):
    
    def __init__(
        self,
        *args,
        update_seeing: bool = False,
        **kwargs
    ):  
        super().__init__(*args,**kwargs)
        self.update_seeing = update_seeing
        
    def get_sextractor_apertures(self) -> list[float]:
        # TODO: do it!
        return []
    
    def check_prerequisites(self):
        return True
    
    def table_to_fake_image(
        self,
        table
    ):
        return Image(data=np.zeros([1,1]),header=table.get_metadata())
    
    def failed_log(
        self, 
        table: SourceTable, 
        comment = None
    ):
        if self.update_seeing:
            table["FWHM_MED"] = None
            table["FWHM_STD"] = None
            table["FWHM_PIX"] = None
            table[ZP_KEY] = None
            table[ZP_STD_KEY] = None
            table[ZP_MAD_KEY] = None
            table[ZP_NSTARS_KEY] = None
            table[MAGSYS_KEY] = "AB"
            if not SEGM_COMMENT_KEY in table.get_metadata():
                table[SEGM_COMMENT_KEY] = comment
    
    def calculate_zeropoint(
        self,
        table: Table,
    ) -> list[dict]:
        """
        Function to calculate zero point from two catalogs
        Args:
            ref_cat: Reference catalog table
            clean_img_cat: Catalog of sources from image to xmatch with ref_cat
        Returns:
        """
        
        apertures = self.get_sextractor_apertures()  # aperture diameters
        zeropoints = []

        for i, aperture in enumerate(apertures):
            offsets = np.ma.array(
                table["magnitude"] - table["MAG_APER"][:, i]
            )
            for outlier_thresh in self.outlier_rejection_threshold:
                cl_offset = sigma_clip(offsets, sigma=outlier_thresh)
                num_stars = np.sum(np.invert(cl_offset.mask))

                zp_mean, zp_med, zp_std = sigma_clipped_stats(
                    offsets, sigma=outlier_thresh
                )

                if num_stars > self.num_matches_threshold:
                    break

            check = [np.isnan(x) for x in [zp_mean, zp_med, zp_std]]
            if np.sum(check) > 0:
                err = (
                    f"Error with nan when calculating sigma stats: \n "
                    f"mean: {zp_mean}, median: {zp_med}, std: {zp_std}"
                )
                logger.error(err)
                raise PhotometryCalculationError(err)

            zp_mad = median_absolute_deviation(offsets)
            
            zero_dict = {
                "diameter": aperture,
                "zp_mean": zp_mean,
                "zp_median": zp_med,
                "zp_std": zp_std,
                "zp_mad": zp_mad,
                "nstars": num_stars,
                "mag_cat": table["magnitude"][np.invert(cl_offset.mask)],
                "mag_apers": table["MAG_APER"][:, i][
                    np.invert(cl_offset.mask)
                ],
            }
            zeropoints.append(zero_dict)

        for outlier_thresh in self.outlier_rejection_threshold:
            offsets = np.ma.array(
                table["magnitude"] - table["MAG_AUTO"]
            )
            cl_offset = sigma_clip(offsets, sigma=outlier_thresh)
            num_stars = np.sum(np.invert(cl_offset.mask))
            zp_mean, zp_med, zp_std = sigma_clipped_stats(offsets, sigma=outlier_thresh)
            
            # one match? KeyError
            # zero_auto_mag_cat = table["magnitude"][np.invert(cl_offset.mask)]
            # zero_auto_mag_img = table["MAG_AUTO"][np.invert(cl_offset.mask)]
            zero_auto_mag_cat = None
            zero_auto_mag_img = None
            

            if num_stars > self.num_matches_threshold:
                break
        
        zp_mad = median_absolute_deviation(offsets)
        
        zeropoints.append(
            {
                "diameter": "AUTO",
                "zp_mean": zp_mean,
                "zp_median": zp_med,
                "zp_std": zp_std,
                "zp_mad": zp_mad,
                "nstars": num_stars,
                "mag_cat": zero_auto_mag_cat,
                "mag_apers": zero_auto_mag_img,
            }
        )

        return zeropoints

    def apply_to_images(
        self,
        batch: SourceBatch,
    ) -> SourceBatch:
        phot_output_dir = self.get_phot_output_dir()
        phot_output_dir.mkdir(parents=True, exist_ok=True)

        batch_updated = SourceBatch()
        
        for table in batch: 
            
            data = table.get_data()
            metadata = table.get_metadata()
            
            if not SEXTRACTOR_HEADER_KEY in table.get_metadata().keys():
                batch_updated.append(table)
                logger.warning(f"{SEXTRACTOR_HEADER_KEY} not found in {table[BASE_NAME_KEY]}")
                continue
            
            fwhm_med, _, fwhm_std, med_fwhm_pix, _, _ = get_fwhm(data)

            header_map = {
                "FWHM_MED": fwhm_med,
                "FWHM_STD": fwhm_std,
                "FWHM_PIX": med_fwhm_pix,
            }
            for key, value in header_map.items():
                if np.isnan(value):
                    value = -999.0
                metadata[key] = value

            if len(table) < self.num_matches_threshold:
                err = (
                    f"Not enough sources ({len(table)} found in reference catalog "
                    f"to calculate a reliable zeropoint. "
                    f"Require at least {self.num_matches_threshold} matches."
                )
                logger.error(err)
                self.failed_log(table,f"expected at least {self.num_matches_threshold} matches")
                # raise PhotometryReferenceError(err)
                batch_updated.append(table)
                continue

            logger.debug(f"Found {len(table)} clean sources in image.")

            zp_dicts = self.calculate_zeropoint(
                table=data
            )

            aperture_diameters = []
            zp_values = []

            with warnings.catch_warnings(record=True):
                warnings.simplefilter("ignore", category=VerifyWarning)

                for zpvals in zp_dicts:
                    metadata[f"ZP_{zpvals['diameter']}"] = zpvals["zp_mean"]
                    metadata[f"ZP_{zpvals['diameter']}_std"] = zpvals["zp_std"]
                    metadata[f"ZP_{zpvals['diameter']}_mad"] = zpvals["zp_mad"]
                    metadata[f"ZP_{zpvals['diameter']}_nstars"] = zpvals["nstars"]
                    try:
                        aperture_diameters.append(float(zpvals["diameter"]))
                        zp_values.append(zpvals["zp_mean"])
                    except ValueError:
                        continue

                aperture_diameters.append(med_fwhm_pix * 2)
                zp_values.append(metadata["ZP_AUTO"])

                if sextractor_checkimg_map["BACKGROUND_RMS"] in metadata.keys():
                    logger.debug(
                        "Calculating limiting magnitudes from background RMS file"
                    )
                    limmags = get_maglim(
                        table[sextractor_checkimg_map["BACKGROUND_RMS"]],
                        zp_values,
                        np.array(aperture_diameters) / 2.0,
                    )
                else:
                    limmags = [-99] * len(aperture_diameters)

                for ind, diam in enumerate(aperture_diameters[:-1]):
                    metadata[f"MAGLIM_{np.rint(diam)}"] = limmags[ind]
                metadata[MAGLIM_KEY] = limmags[-1]

                # [!] ERROR
                metadata[ZP_KEY] = metadata["ZP_AUTO"]
                # image[ZP_STD_KEY] = image["ZP_AUTO_STD"]
                metadata[ZP_STD_KEY] = metadata["ZP_AUTO_std"]
                # image[ZP_NSTARS_KEY] = image["ZP_AUTO_NSTARS"]
                metadata[ZP_MAD_KEY] = metadata["ZP_AUTO_mad"]
                metadata[ZP_NSTARS_KEY] = metadata["ZP_AUTO_nstars"]
                # [!] 
                metadata[MAGSYS_KEY] = "AB"
                
                table_updated = SourceTable(data,metadata)
                batch_updated.append(table_updated)

        return batch_updated
    
    def _apply_to_images(
        self,
        batch: SourceBatch,
    ) -> SourceBatch:    
        # batch_photcal = super()._apply_to_images(batch_fake_images) # if _apply_to_images in main file
        batch_updated = self.apply_to_images(batch)
        return batch_updated

class SeeingCalculator(BaseSourceProcessor):
    
    base_key = "seeing_calculator"
    
    def __init__(
        self,
        model_size: int = 21,
        x_stddev_guess: int = 5,
        y_stddev_guess: int = 5,
        output_sub_dir: Path | str = 'seeing',
        additional_cols = None,
        dev: bool = False,
    ):
        super().__init__()
        self.model_size = model_size
        self.x_stddev_guess = x_stddev_guess
        self.y_stddev_guess = y_stddev_guess
        self.output_sub_dir = output_sub_dir
        self.additional_cols = additional_cols
        self.dev = dev
    
    def fit_gaussian(
        self,
        table: SourceTable,
    ):
            
        prec = table[OVERSAMPLE_KEY]
        size = self.model_size*prec
        x, y = np.mgrid[0:size, 0:size]/prec
        center = (size//2/prec,size//2/prec)

        psfmodel = load_object(table[NPSFPATH_KEY])
        if psfmodel is not None: # check if PSF model exists
            psf_array = psfmodel.evaluate(x,y,1,*center)
        else:
            return None
        
        model = Gaussian2D(
            amplitude=np.max(psf_array),
            x_mean=center[0],
            y_mean=center[1],
            x_stddev=self.x_stddev_guess, 
            y_stddev=self.y_stddev_guess
        )
        fitter = TRFLSQFitter(calc_uncertainties=True)
        fitted_model = fitter(model,x,y,psf_array)
        
        if self.dev:
            output_dir = get_output_dir(self.output_sub_dir, self.night_sub_dir)
            save_path = output_dir.joinpath(table[BASE_NAME_KEY].replace('fits','psf_fitted_gaussian.pkl'))
            dump_object(fitted_model,save_path)
        
        return fitted_model, fitter
    
    def failed_log(
        self,
        table: SourceTable
    ):
        table[PSFMODEL_ELLIPTICITY_KEY] = None
        table[PSFMODEL_ELLIPTICITY_KEY] = None
        table[PSFMODEL_FWHM_PIX_KEY] = None
        table[PSFMODEL_FWHM_KEY] = None
        table[PSFMODEL_FWHM_ARCSEC_KEY] = None
        table[PSFMODEL_FWHM_ERR_ARCSEC_KEY] = None
        
        return table
        
    def _apply_to_sources(
        self,
        batch: SourceBatch,
    ) -> SourceBatch:
        
        batch_updated = SourceBatch()
        
        for table in batch:
            
            if OVERSAMPLE_KEY in table.get_metadata().keys():
                fit_return = self.fit_gaussian(table)
                if fit_return is not None:
                    fitted_model, fitter = fit_return
                    table_updated = fwhm_from_gaussian(
                        table=table,
                        fitter=fitter,
                        fitted_model=fitted_model
                    )
                else:
                    table_updated = self.failed_log(table)
            else:
                table_updated = self.failed_log(table)
            output_dir = get_output_dir(sub_dir=self.night_sub_dir,dir_root=self.output_sub_dir)
            save_night_log_seeing(
                table_updated,
                output_dir,
                additional_cols=self.additional_cols
            )
            batch_updated.append(table_updated)
        
        return batch_updated
     
class SourceBatchToImageBatch(BaseSourceProcessor):
    
    base_key = "source_batch_to_image_batch"
    
    def __init__(self):
        super().__init__()
        
    def _apply_to_sources(
        self,
        batch: SourceBatch
    ) -> ImageBatch:
        
        image_batch = ImageBatch()
        
        for table in batch:
            image = Image(
                data=fits.getdata(table[LATEST_SAVE_KEY]),
                header=table.get_metadata()
            )
            image_batch.append(image)
            
        return image_batch

def safe_safe_fits(
    image: Image,
    path: str | Path,
):
    header = image.get_header()
    if 'BZERO' in header.keys():
        image = Image(image.get_data() - header['BZERO'], header)
        
    save_fits(image, path)    

class ImageSaverSafe(ImageSaver):
    
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args,**kwargs)
        
    def save_fits(
        self,
        image: Image,
        path: str | Path,
    ):
        
        safe_safe_fits(image,path)

class FlatCalibratorSafe(FlatCalibrator):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args,**kwargs)
        
    def save_fits(
        self,
        image: Image,
        path: str | Path,
    ):
    
        safe_safe_fits(image,path)

wifes_autoguider_bin_size_map = {
    (1,1): {
        'size': (1027, 1072),
        'crop': [[None,-1],[None,-1]]
    },
    (2,2): {
        'size': (513, 536),
        'crop': [[None,None],[None,-1]]
    },
    (3,3): {
        'size': (342, 357),
        'crop': [[None,None],[None,None]]
    }
}

def downsampler(
    array: np.ndarray,
    factor: tuple,
):
    if array.shape[0] % factor[0] != 0 or array.shape[1] % factor[1] != 0:
        raise ValueError("Array dimensions must be divisible by box_size")

    reshaped_array = array.reshape((array.shape[0]//factor[0], factor[0], 
                                    array.shape[1]//factor[1], factor[1]))
    resampled_array = reshaped_array.sum(axis=(1, 3))

    return resampled_array

def downsampler2(
    arr: np.ndarray, 
    factor: tuple
):
    if np.all(np.array(arr.shape) % factor == 0):
        return arr.reshape(arr.shape[0]//factor[0], factor[0], arr.shape[1]//factor[1], factor[1]).mean(axis=(1,3))
    else:
        return arr

def upsampler(
    arr: np.ndarray,
    factor: tuple,
):  
    """
    Upsample the array by preserving the sum 

    """
    arr_up = arr.repeat(factor[0], axis=0)/factor[0]
    arr_up = arr_up.repeat(factor[1], axis=1)/factor[1]
    return arr_up

def upsample(
    image: Image | np.ndarray,
    resampler,
    bin_size_map,
    binning = None,
) -> Image | np.ndarray | None:
    
    if isinstance(image, Image):
        data = image.get_data()
        header = image.get_header()
        binning = tuple(map(int,header['CCDSUM'].split(' ')))
    elif isinstance(image, np.ndarray):
        data = image
        binning = binning
        
    size = data.shape
    
    if binning in bin_size_map.keys() and bin_size_map[binning]['size'] == size:
        crop = bin_size_map[binning]['crop']
        image_resamp = resampler(data,binning)
        image_resamp = image_resamp[
            crop[0][0]:crop[0][1],
            crop[1][0]:crop[1][1],
        ]
        if isinstance(image, Image):
            return Image(image_resamp,fits.Header(header))
        elif isinstance(image, np.ndarray):
            return image_resamp
    else:
        return None
    
def downsample(
    image: Image,
    resampler,
    bin_size_map,
) -> Image | None:
    
    data = image.get_data()
    header = image.get_header()
    binning = tuple(map(int,header['CCDSUM'].split(' ')))
    
    if binning in bin_size_map.keys():
        crop = np.array(bin_size_map[binning]['crop'])
        crop[crop == None] = 0
        pad = np.abs(crop)
        
        data_padded = np.pad(
            data,
            (
                (pad[0][0],pad[0][1]),
                (pad[1][0],pad[1][1]),
            ),
            mode='constant',
            constant_values=(np.nan,)
        )
        data_resamp = resampler(data_padded,binning)
        return Image(data_resamp,fits.Header(header))
    else:
        return None

class WifesAutoguiderImageResampler(BaseImageProcessor):
    
    base_key = 'wifes_autoguider_image_resampler'
    
    def __init__(
        self,
        sample: str = 'up',
        upsampler = None,
        downsampler = None,
        bin_size_map = wifes_autoguider_bin_size_map,
    ):
        super().__init__()
        self.bin_size_map = bin_size_map
        self.sample = sample
        self.upsampler = upsampler
        self.downsampler = downsampler
        
    def upsample(
        self,
        image: Image
    ) -> Image | None:
        
        return upsample(
            image,
            self.upsampler,
            self.bin_size_map
        )
        
    def downsample(
        self,
        image: Image
    ) -> Image | None:
        
        return downsample(
            image,
            self.downsampler,
            self.bin_size_map
        )
        
    def _apply_to_images(
        self,
        batch: ImageBatch
    ) -> ImageBatch:
        
        # master_size = # TODO: compute lowest based on all data
        batch_updated = ImageBatch()
        
        for image in batch:
            if self.sample == 'up':
                image_updated = self.upsample(image)
            elif self.sample == 'down':
                image_updated = self.downsample(image)
            else:
                raise ValueError("sample can either be 'up' or 'down'")
                
            if image_updated is None:
                logger.warning(f"{image[BASE_NAME_KEY]} not resampled")
            else:
                batch_updated.append(image_updated)
        
        return batch_updated
         
def load_wifes_guider_fits(
    path: str | Path
) -> tuple[np.array, astropy.io.fits.Header]:
    data, header = open_fits(path)
    header[OBSCLASS_KEY] = ACQ_KEY
    header[TARGET_KEY] = header['OBJECT']
    header[COADD_KEY] = 1
    header[GAIN_KEY] = 1
    header['CALSTEPS'] = '' # find path
    header[PROC_FAIL_KEY] = ''
    if 'RADECSYS' in header:
        sys = header.pop('RADECSYS')
        header['RADESYSa'] = sys 
        header['RADECSYS'] = sys
    if FILTER_KEY not in header:
        header[FILTER_KEY] = header['TVFILT']
    return data, header

def load_wifes_guider_image(path: str | Path) -> Image:
    return open_raw_image(path, load_wifes_guider_fits)

def get_master_flat():
    cal_dir = os.environ['CAL_DATA_DIR']
    return [Path(cal_dir).joinpath(file) for file in os.listdir(cal_dir) if file.startswith('flat')][0]
