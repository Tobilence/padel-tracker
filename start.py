from core.pipeline import PadelTrackingPipeline
import logging
logging.basicConfig(level=logging.DEBUG)

pipeline = PadelTrackingPipeline("data/video.mp4")
pipeline.run()
