from pydantic import BaseModel

class GeneratorSetting(BaseModel):
    num_segments: int = 3
    segment_length: int = 1000
    change_point_type: str = 'sudden_shift'  # Options: 'sudden_shift', 'gradual_drift', 'periodic_change'
    seed:int = 12
    