import logging
from mirar.pipelines.base_pipeline import Pipeline

PIPELINE_NAME = "wifes_autoguider"

logger = logging.getLogger(__name__)

class WifesAutoguiderPipeline(Pipeline):
    
    name = PIPELINE_NAME
    
    non_linear_level = 99999
    
    all_pipeline_configurations = {}
    