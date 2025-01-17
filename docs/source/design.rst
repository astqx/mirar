Design Philosophy
=================

mirar ia a modular python package for analaysis astronomy images.

There are a few guiding principles that ensures the modularity works:

- Transparent requirements: Processors should either be self-contained, or be explicit about their requirements. Processors have a dedicated function, check_prerequisites, for this purpose. The goal is that processors should work out of the box, regardless of preceding processors, or fail immediately because a required precursor is missing.

- Flexibility for input images: There are a small number of fields which will always be provided in every image. These are core fields. Processors should not assume other keys are present, unless they are generated by explicitly required prerequisites.

- Consistent data: astronomy data might look different when it comes out of the instrument, but when it's read in by mirar the pipeline will make sure that the data is converted to the right format. You should only ever open fits files using the open_fits function in mirar.io, and never use astropy.io directly. The open_fits is a wrapper which also performs additional checks as needed.

- Consistent naming: mirar follows the official python naming conventions, as enumerated in pep:
    - Classes are CamelCase
    - Functions are snake_case
    - Variables are snake_case
    - Constants are ALL_CAPS
    - Modules are lowercase
    - Filenames are lowercase
    - Directories are lowercase

- Optional image saving: A lot of astronomy software will repeatedly write and rewrite science images. mirar DOES NOT DO THIS! If a processor requires a science image to be written, e.g because it is a wrapper to other software, it should be saved as a temporary file then reloaded and the temporary image deleted. The only time a science image gets permanently saved is if an ImageSaver processor is used. The same holds for masks. However, ancillary images (such as scorr images or weight image) which are generated by external tools (such as Swarp) may be written to disk by processors.

- Masks as nans: mirar follows a masking convention where masked pixels are written as nabs in the science image. For conveience, a mask image can be grabbed and/or written to file, with the convewntion that 0 is masked and 1 is not masked.

- Consistent logging: mirar uses the python logger. You should never include print statements. Warnings and errors should be logged at the appropriate level. Statements at the dataset level can be 'info' or 'debug', image-level statements for debugging purposes should be logged under debug.

- Instrument-agnostic: Most of the code (especially processors!) is instrument-agnostic. Each instrument needs a dedicated pipeline subdirectory, containing all instrument-specific info. The rest of the code should work for any instrument.

- Pythonic error handling: the code should never fail silently or return integer error codes. Instead, it should follow pythonic conventions, and explicitly raise an exception. This exception must inherit from the base ProcessorError class in mirar.errors, so it can be caught, logged and handled correctly.
