from mirar.pipelines.wifes_autoguider.processors import *
from mirar.pipelines.wifes_autoguider.paths import *

load = [
    ImageLoader(load_image=load_wifes_guider_image)
]

bkg_sub = [
    PhotutilsBkgSubtractor(
        box_size=(10,10),
        select_images=default_select_acquisition,
        output_sub_dir=OUTPUT_DIRS['BKG'],
        dev=False,
        save_bkg=False,
    ),
    ImageSaver(output_dir_name=OUTPUT_DIRS['BKG'])
]

src_det = [
    PhotutilsSourceFinder(
        convolve=True,
        convolution_kernel=sex_all_ground,
        output_sub_dir=OUTPUT_DIRS['DET'],
        dev=True
    ),
    PhotutilsSourceCatalog(
        make_psf_cutouts=False,
        use_background=False,
        output_sub_dir=OUTPUT_DIRS['DET']
    ),
    SourceWriter(output_dir_name=OUTPUT_DIRS['DET'])
]

xmatch = [
    SourceCrossMatch(
        ref_catalog_generator=WifesAutoguiderVisier(Gaia).generator,
        temp_output_sub_dir="phot",
        crossmatch_radius_arcsec=3.0, # or 2 TODO: test
        write_regions=True,
        cache=True,
        image_photometric_catalog_purifier=wifes_autoguider_photometric_catalog_purifier,
        ref_cat_cols=['Gal']
    ),
    CustomSourceTableModifier(modifier_function=wifes_autoguider_gal_filter),
    # SourceWriter(output_dir_name=OUTPUT_DIRS['XMAT'])
]

psfmodel = [
    PhotutilsModelPSF(
        oversampling=8,
        make_psf_cutouts_from_catalog=True,
        psf_cutout_size_auto=True,
        # psf_cutout_size=41,
        dev = True,
        cache=True,
        output_sub_dir=OUTPUT_DIRS['PSF_MODEL'],
        fitter=TRFLSQFitter,
        update_seeing=True,
        # fit_boxsize=5*8+1,
        # fitter_params={'calc_uncertainties':True},
        # fitter_kwargs=dict(),
        # shape=None,
    ),
    SourceWriter(output_dir_name=OUTPUT_DIRS['PSF_MODEL'])
]

seeing = [
    SeeingCalculator(
        output_sub_dir = OUTPUT_DIRS['SEE'],
        additional_cols=['OBSBLKID','PROPID','RA','DEC']
    ),
    SourceWriter(output_dir_name=OUTPUT_DIRS['SEE'])
]

photcal = [
    SourcePhotCalibrator(
        ref_catalog_generator=WifesAutoguiderVisier(Gaia).generator,
        temp_output_sub_dir="phot",
        crossmatch_radius_arcsec=3.0, # or 2 TODO: test
        write_regions=True,
        cache=True,
        outlier_rejection_threshold=[1.5, 2.0, 3.0],
        image_photometric_catalog_purifier=wifes_autoguider_photometric_catalog_purifier,
        num_matches_threshold=0, # TODO: Change! (testing only ? maybe)
    ),
    SourceWriter(output_dir_name=OUTPUT_DIRS['PHOTCAL'])
]

test_config = list(itertools.chain(
    load,
    bkg_sub,
    src_det,
    # photcal
    xmatch,
    psfmodel,
    seeing,
))

prod_config = [
    PrepareOutputDirectories(output_dirs=OUTPUT_DIRS.values()),
    ImageLoader(load_image=load_wifes_guider_image),
    MaskPixelsFromFunction(
        mask_function = make_bad_pix_mask,
        write_masked_pixels_to_file = True,
        output_dir = OUTPUT_DIRS['MASK'],
        # only_write_mask=True,
    ),
    LACosmicCleaner(
        effective_gain_key = GAIN_KEY,
        readnoise=0 # TODO: check
    ),
    WifesAutoguiderImageResampler( # TODO: > 3 binning
        sample='up',
        upsampler=upsampler
    ),
    MasterFlatCalibrator(
        master_image_path=get_master_flat()
    ),
    WifesAutoguiderImageResampler(
        sample='down',
        downsampler=downsampler
    ),
    # ImageSaverSafe(output_dir_name=OUTPUT_DIRS['FLAT']), # !! needs to be removed !!
    PhotutilsBkgSubtractor(
        box_size=(32,32),
        select_images=default_select_acquisition,
        output_sub_dir=OUTPUT_DIRS['BKG'],
        dev=False,
        save_bkg=False,
        coverage_mask_as_mask=True,
        box_size_scale_function=scale_boxsize,
        bzero_correction=True,
    ),
    ImageSaverSafe(output_dir_name=OUTPUT_DIRS['BKG']), # LATEST_SAVE_KEY
    PhotutilsSourceFinder(
        convolve=True,
        convolution_kernel=sex_all_ground,
        threshold_factor=3,
        output_sub_dir=OUTPUT_DIRS['DET'],
        dev=True, # !! needs to be removed !!
        dev_params=['cmap'], # !! needs to be removed !!
    ),
    PhotutilsSourceCatalog(
        make_psf_cutouts=False,
        use_background=False,
        output_sub_dir=OUTPUT_DIRS['DET'],
        binning_correction=True,
    ),
    # SourceWriter(OUTPUT_DIRS['DET']), # !! needs to be removed !!
    SourceCrossMatch(
        # ref_catalog_generator=WifesAutoguiderVisier(
        #     visier_catalog=Gaia,
        #     reuse_cached_catalog=True,
        #     reuse_cache_dir=OUTPUT_DIRS['XMAT'],
        # ).generator,
        ref_catalog_generator=WifesAutoguiderVisier(Gaia).generator,
        temp_output_sub_dir="xmatch",
        crossmatch_radius_arcsec=3.0, # or 2 TODO: test
        write_regions=False,
        cache=False,
        image_photometric_catalog_purifier=wifes_autoguider_photometric_catalog_purifier,
        ref_cat_cols=['Gal','magnitude'],
    ),
    SourcePhotCalibrator(
        ref_catalog_generator=WifesAutoguiderVisier(Gaia).generator,
        temp_output_sub_dir="phot",
        update_seeing=True,
        crossmatch_radius_arcsec=3.0, # or 2 TODO: test
        write_regions=True,
        cache=True,
        outlier_rejection_threshold=[1.5, 2.0, 3.0],
        image_photometric_catalog_purifier=wifes_autoguider_photometric_catalog_purifier,
        num_matches_threshold=1, # TODO: Change! (testing only ? maybe)
    ),
    PhotutilsModelPSF(
        oversampling=1,
        make_psf_cutouts_from_catalog=True,
        psf_cutout_size_auto=True,
        # psf_cutout_size=41,
        dev = False,
        cache=True,
        output_sub_dir=OUTPUT_DIRS['PSF_MODEL'],
        fitter=TRFLSQFitter,
        update_seeing=True,
        maxiters=100
        # fit_boxsize=5*8+1,
        # fitter_params={'calc_uncertainties':True},
        # fitter_kwargs=dict(),
        # shape=None,
    ),
    SeeingCalculator(
        output_sub_dir = OUTPUT_DIRS['SEE'],
        additional_cols=['OBSBLKID','PROPID','RA','DEC']
    ),
    # RemoveEmptySourceTables(), # !! needs to be removed !!
    # SourceWriter(output_dir_name=OUTPUT_DIRS['SEE']), # !! needs to be removed !!
]

master_flat_config = [
    PrepareOutputDirectories(output_dirs=OUTPUT_DIRS.values()),
    ImageLoader(load_image=load_wifes_guider_image),
    MaskPixelsFromFunction(
        mask_function = make_bad_pix_mask,
        write_masked_pixels_to_file = True,
        output_dir = OUTPUT_DIRS['MASK'],
        # only_write_mask=True,
    ),
    LACosmicCleaner(
        effective_gain_key = GAIN_KEY,
        readnoise=0 # TODO: check
    ),
    ImageSaverSafe(output_dir_name=OUTPUT_DIRS['COSMIC']), # -->
    PhotutilsBkgSubtractor(
        box_size=(32,32),
        select_images=default_select_acquisition,
        output_sub_dir=OUTPUT_DIRS['BKG'],
        dev=False,
        save_bkg=False,
        coverage_mask_as_mask=True,
        box_size_scale_function=scale_boxsize
    ),
    # ImageSaver(output_dir_name=OUTPUT_DIRS['BKG']),
    PhotutilsSourceFinder(
        convolve=True,
        convolution_kernel=sex_all_ground,
        threshold_factor=3,
        output_sub_dir=OUTPUT_DIRS['DET'],
        dev=False,
        dev_params=['cmap'],
    ),
    PhotutilsSourceCatalog(
        make_psf_cutouts=False,
        use_background=False,
        output_sub_dir=OUTPUT_DIRS['DET']
    ),
    SourceBatchToImageBatch(), # <--
    # ImageBatcher(["FILTER"]), #TODO: future
    WifesAutoguiderImageResampler(
        sample='up',
        upsampler=upsampler
    ),
    MaskPixelsFromFunction(
        mask_function = lambda image: get_extended_mask(
            image=image,
            base_mask_function=get_segmentation_mask,
            extension_mask_function=invert_adjacent_to_false,
            extension_mask_function_kwargs = {'radius': 5},
            resampler=upsampler,
            resample_custom_function=upsample,
            bin_size_map=wifes_autoguider_bin_size_map,
        ),
        # write_masked_pixels_to_file = True,
        # output_dir = OUTPUT_DIRS['FLAT'],
    ),
    FlatCalibratorSafe(
        select_flat_images=default_select_acquisition
    ),
    # ImageSaverSafe(output_dir_name=OUTPUT_DIRS['FLAT']),
]

bkg_sub_test = [
    PrepareOutputDirectories(output_dirs=OUTPUT_DIRS.values()),
    ImageLoader(load_image=load_wifes_guider_image),
    PrepareOutputDirectories(output_dirs=OUTPUT_DIRS.values()),
    ImageLoader(load_image=load_wifes_guider_image),
    MaskPixelsFromFunction(
        mask_function = make_bad_pix_mask,
        write_masked_pixels_to_file = True,
        output_dir = OUTPUT_DIRS['MASK'],
        # only_write_mask=True,
    ),
    LACosmicCleaner(
        effective_gain_key = GAIN_KEY,
        readnoise=0 # TODO: check
    ),
    WifesAutoguiderImageResampler( # TODO: > 3 binning
        sample='up',
        upsampler=upsampler
    ),
    MasterFlatCalibrator(
        master_image_path=get_master_flat()
    ),
    WifesAutoguiderImageResampler(
        sample='down',
        downsampler=downsampler
    ),
    ImageSaverSafe(output_dir_name=OUTPUT_DIRS['FLAT']), # !! needs to be removed !!
    PhotutilsBkgSubtractor(
        box_size=(32,32),
        select_images=default_select_acquisition,
        output_sub_dir=OUTPUT_DIRS['BKG'],
        dev=True,
        save_bkg=True,
        coverage_mask_as_mask=True,
        box_size_scale_function=scale_boxsize,
        bzero_correction=True,
    ),
    ImageSaverSafe(output_dir_name=OUTPUT_DIRS['BKG']), # LATEST_SAVE_KEY
]

resample_test_config = [
    ImageLoader(load_image=load_wifes_guider_image),
    WifesAutoguiderImageResampler(
        sample='up',
        upsampler=upsampler
    ),
    ImageSaver(output_dir_name=OUTPUT_DIRS['XMAT']),
    WifesAutoguiderImageResampler(
        sample='down',
        downsampler=downsampler2
    ),
    ImageSaver(output_dir_name=OUTPUT_DIRS['SEE']),
]
