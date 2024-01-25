import numpy as np
import json


class WaveLocationScatter:
    def __init__(self, location_name, lat,lng, dataset_duration, water_depth,
                 Hm0_start_value, Hm0_step_size, Tx_start_value, Tx_step_size):
        self.location_name = location_name
        self.lat = lat
        self.lng = lng
        self.dataset_duration = dataset_duration
        self.water_depth = water_depth
        self.Hm0_start_value = Hm0_start_value
        self.Hm0_step_size = Hm0_step_size
        self.Tx_start_value = Tx_start_value
        self.Tx_step_size = Tx_step_size

        # Generate location scatter array
        self.location_scatter = self._generate_location_scatter()

    def _generate_location_scatter(self):
        # Generate np array based on provided parameters
        Hm0_values = np.arange(self.Hm0_start_value,
                               self.Hm0_start_value + self.dataset_duration * self.Hm0_step_size,
                               self.Hm0_step_size)
        Tm_values = np.arange(self.Tx_start_value,
                              self.Tx_start_value + self.dataset_duration * self.Tx_step_size,
                              self.Tx_step_size)

        # Create 2D array (scatter array) using meshgrid
        Hm0_mesh, Tm_mesh = np.meshgrid(Hm0_values, Tm_values)

        # Use provided location scatter data instead of generated meshgrid
        if Hm0_mesh.shape == (self.dataset_duration, len(Hm0_values)) and Tm_mesh.shape == (self.dataset_duration, len(Tm_values)):
            return np.array([row for row in zip(Hm0_mesh.flatten(), Tm_mesh.flatten())])
    def to_dict(self):
        return {
            "location_name": self.location_name,
            "lat": self.lat,
            "lng": self.lng,
            "dataset_duration": self.dataset_duration,
            "water_depth": self.water_depth,
            "Hm0_start_value": self.Hm0_start_value,
            "Hm0_step_size": self.Hm0_step_size,
            "Tx_start_value": self.Tx_start_value,
            "Tx_step_size": self.Tx_step_size,
            "location_scatter": self.location_scatter.tolist()  # Convert numpy array to a list for JSON serialization
        }

    @classmethod
    def from_dict(cls, data):
        obj = cls(
            data["location_name"],
            data["lat"],
            data["lng"],
            data["dataset_duration"],
            data["water_depth"],
            data["Hm0_start_value"],
            data["Hm0_step_size"],
            data["Tx_start_value"],
            data["Tx_step_size"]
        )
        # Convert the list back to a numpy array
        obj.location_scatter = np.array(data["location_scatter"])
        return obj

    def save_to_json(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.to_dict(), file, indent=2)

    @classmethod
    def load_from_json(cls, filename):
        try:
            with open(filename, 'r') as file:
                data = json.load(file)
                return cls.from_dict(data)
        except FileNotFoundError:
            return None




    def get_axis(self):
      Tp_Tm_ratio = 1.29
      Tp_Te_ratio = 1.12
      Tm_Te_ratio = Tp_Te_ratio / Tp_Tm_ratio

      # Determine the dimensions of the location_scatter matrix
      num_rows, num_columns = self.location_scatter.shape
      Hm0_end_value = self.Hm0_start_value + ((num_rows-0.9)*self.Hm0_step_size )  # Assuming each column corresponds to an Hm0 value
      Tx_end_value = self.Tx_start_value + ((num_columns-0.9)*self.Tx_step_size )  # Assuming each row corresponds to a Tx value
     

      # Define Tx and Hm0 axes
      #Hm0 axes
      Hm0_array = np.arange(self.Hm0_start_value, Hm0_end_value, self.Hm0_step_size)

      # Create Tm (x-axis) and Hm0 (y-axis)
      Tm_array = np.arange(self.Tx_start_value, Tx_end_value, self.Tx_step_size)
      Tp_array = Tm_array * Tp_Tm_ratio
      Te_array = Tp_array / Tp_Te_ratio
      return(Tm_array,Tp_array,Te_array,Hm0_array)



    def display_info(self):
        print(f"Location Name: {self.location_name}")
        print(f"lat: {self.lat}")
        print(f"lng: {self.lng}")
        print(f"Dataset Duration: {self.dataset_duration}")
        print(f"Water Depth: {self.water_depth}")
        print(f"Location Scatter Array:")
        print(self.location_scatter)
        print("\n")
