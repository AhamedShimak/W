import json
from wave_location import WaveLocationScatter
import numpy as np

class WaveLocationScatterCollection:
    def __init__(self):
        self.location_scatters = []

    def add_location_scatter(self, location_scatter):
        self.location_scatters.append(location_scatter)

    def save_to_json(self, filename):
        data = {"location_scatters": [location_scatter.to_dict() for location_scatter in self.location_scatters]}
        with open(filename, 'w') as file:
            json.dump(data, file, indent=2)

    @classmethod
    def load_from_json(cls, filename):
        try:
            with open(filename, 'r') as file:
                data = json.load(file)
                collection = cls()
                for location_scatter_data in data.get("location_scatters", []):
                    location_scatter = WaveLocationScatter.from_dict(location_scatter_data)
                    collection.add_location_scatter(location_scatter)
                return collection
        except FileNotFoundError:
            return None

    def display_info(self):
        for i, location_scatter in enumerate(self.location_scatters):
            print(f"Location Scatter #{i + 1}")
            location_scatter.display_info()


##############################################################################
#manually creating a locationscatter object

fiji=WaveLocationScatter("UK EMEC",0,4,5,20,1.0,1.0,3.5,1.0)
fiji.location_scatter=np.array([[400,1457,800,306,116,41,17,6,1,0,1,0],
        [16,325,1416,574,202,87,29,11,1,0,0,0],
        [0,2,165,957,296,63,29,15,4,1,0,0],
        [0,0,0,90,491,103,18,5,3,2,0,0],
        [0,0,0,0,64,211,27,6,3,1,0,0],
        [0,0,0,0,1,53,61,5,1,0,0,0],
        [0,0,0,0,1,2,31,13,1,0,0,0],
        [0,0,0,0,0,0,5,25,2,0,0,0],
        [0,0,0,0,0,0,0,8,6,0,0,0],
        [0,0,0,0,0,0,0,0,4,1,0,0],
        [0,0,0,0,0,0,0,0,1,1,0,0]])

ferro=WaveLocationScatter("Faroe Islands W1",61.525,-6.900,5,20,0.25,0.5,2.5,1.0)
ferro.location_scatter=np.array([[87,355,567,817,775,346,98,34,3,0,0,0,0,0,0,0,0],
        [77,1864,3080,3019,3320,2148,1360,540,124,26,0,0,0,0,0,0,0],
        [0,435,3417,3479,3008,2536,1347,793,295,51,17,0,0,0,0,0,0],
        [0,0,1201,3439,2748,2160,1877,1001,517,204,75,5,0,0,0,0,0],
        [0,0,69,2203,2584,1742,1618,1113,487,243,108,21,6,0,0,0,0],
        [0,0,1,795,2577,1684,1191,942,471,205,108,31,0,0,0,0,0],
        [0,0,0,67,1662,1647,1085,771,398,264,96,33,12,0,0,0,0],
        [0,0,0,2,773,1764,916,558,442,224,114,35,16,2,0,0,0],
        [0,0,0,0,115,1367,765,521,316,211,123,48,12,2,0,0,0],
        [0,0,0,0,3,739,800,428,245,120,67,33,1,15,0,0,0],
        [0,0,0,0,0,269,845,378,179,90,48,42,12,0,0,0,0],
        [0,0,0,0,0,25,571,345,144,78,32,26,3,0,0,0,0],
        [0,0,0,0,0,2,279,312,111,52,23,11,0,0,0,0,0],
        [0,0,0,0,0,0,81,316,81,38,19,8,1,0,0,0,0],
        [0,0,0,0,0,0,7,211,66,44,9,3,2,0,0,0,0],
        [0,0,0,0,0,0,2,85,74,17,3,2,2,0,0,0,0],
        [0,0,0,0,0,0,0,39,76,15,2,5,6,0,0,0,0],
        [0,0,0,0,0,0,0,14,47,19,2,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,4,89,106,18,1,0,0,0,0,0]])














##############################################################################
# Assume you have created and filled the objects
#collection = WaveLocationScatterCollection()

# Add WaveLocationScatter objects to the collection
#collection.add_location_scatter(fiji)
#collection.add_location_scatter(ferro)
# ... add more as needed

# Save the collection to a JSON file
#collection.save_to_json('Database\location_collection.json')