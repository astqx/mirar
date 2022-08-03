import logging
from winterdrp.pipelines.base_pipeline import Pipeline
from winterdrp.pipelines.wirc.wirc_pipeline import WircPipeline
from winterdrp.pipelines.summer.summer_pipeline import SummerPipeline
from winterdrp.pipelines.wirc_imsub.wirc_imsub_pipeline import WircImsubPipeline

logger = logging.getLogger(__name__)

# Convention: lowercase names


def get_pipeline(instrument, selected_configurations=None, *args, **kwargs):

    try:
        pipeline = Pipeline.pipelines[instrument.lower()]
        logger.info(f"Found {instrument} pipeline")
    except KeyError:
        err = f"Unrecognised pipeline {instrument}. Available pipelines are: {Pipeline.pipelines.keys()}"
        logger.error(err)
        raise KeyError(err)

    return pipeline(selected_configurations=selected_configurations, *args, **kwargs)
