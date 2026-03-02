import pandas as pd
from generator.change_point_generator import ChangePointGenerator


# Generate time series data with change points
generator = ChangePointGenerator(num_segments=3, 
                                 segment_length=1000, 
                                 change_point_type='sudden_shift', 
                                 seed=1000)
generator.generate_data()

data = generator.get_data()

# Save the generated data to a CSV file
df = pd.DataFrame(data, columns=['value'])
df.to_csv('synthetic_data.csv', index=False)
