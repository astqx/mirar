import logging
from mirar.pipelines.base_pipeline import Pipeline
from mirar.pipelines.wifes_autoguider.blocks import (
    prod_config,
    master_flat_config
)

PIPELINE_NAME = "wifes_autoguider"

logger = logging.getLogger(__name__)

class WifesAutoguiderPipeline(Pipeline):
    
    name = PIPELINE_NAME
    
    non_linear_level = 99999
    
    # all_pipeline_configurations = {}
    
    all_pipeline_configurations = {
        "default": prod_config,
        "prod_config": prod_config,
        "master_flat_config": master_flat_config
    }
    
    