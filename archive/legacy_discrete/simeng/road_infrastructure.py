from discrete.simeng.qsim_engine import pds, simEnvironment, simEntity, Queue, Service

carLength = 2

class road():
    def __init__(self, name, road_length = 100, lanesLeft=2, lanesRight=2, oneWay = False):
        leftLane = Service( )