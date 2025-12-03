from pydantic import BaseModel

class GeneratorSetting(BaseModel):
    num_segments: int = 3
    segment_length: int = 1000
    change_point_type: str = 'sudden_shift'  # Options: 'sudden_shift', 'gradual_drift', 'periodic_change'
    seed:int = 12

class CUSUMSetting(BaseModel):
    warmup_period: int = 500
    delta: float = 3.0
    threshold: float = 10.0

class ProbCUSUMSetting(BaseModel):
    warmup_period: int = 500
    threshold_probability: float = 0.01

    