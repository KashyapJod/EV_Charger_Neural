from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class Location:
    latitude: float
    longitude: float
    is_existing: bool = False
    
    def __eq__(self, other):
        if not isinstance(other, Location):
            return False
        return (self.latitude == other.latitude and 
                self.longitude == other.longitude)
    
    def __hash__(self):
        return hash((self.latitude, self.longitude))

def haversine_distance(loc1: Location, loc2: Location) -> float:
    """Calculate distance between two points using Haversine formula"""
    R = 6371  # Earth's radius in kilometers
    lat1, lon1 = np.radians(loc1.latitude), np.radians(loc1.longitude)
    lat2, lon2 = np.radians(loc2.latitude), np.radians(loc2.longitude)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c
