from qsim_engine import pds, simEnvironment, simEntity
import networkx as nx
import numpy as np

pds.verbose == True
class Compass:
    direction: str = 'N'
    angle: int = 0
    points: int
    atomic_angle: int
    def __init__(self, points):
        if points == 8 or points == 6 or points == 4:
            self.points = points
            self.atomic_angle = 360 / points
        else:
            raise Exception("Only 4, 6 or 8 point compass available in this release")

    def _angle_quantize(self,angle):
        """ Quantize angle into discrete allowable points """
        if angle // self.atomic_angle == angle / self.atomic_angle:
            pass
        else:
            pds.verbose_print(f"Warning: Angle quantized, since compass is {self.points} points")
            angle = angle // self.atomic_angle * self.atomic_angle 
        angle = np.remainder(angle,360)       
        return angle

    def _angle_to_dir(self):
        """ Update str based direction from self.angle """
        if self.points == 8 or self.points == 4:
            angles = [0, 45, 90, 135, 180, 225, 270, 315]
            directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            idx = self.angle == angles
            self.direction = directions[idx]
        else:
            self.direction = 'CW_'+str(self.angle)

    def update_absolute_angle(self, angle=90):
        angle = self._angle_quantize(angle)
        self.angle = angle
        self._angle_to_dir()

    def clockwise(self, relative_angle):
        angle = self.angle + relative_angle
        self.angle = self._angle_quantize(angle)
        self._angle_to_dir()

    def anti_clockwise(self, relative_angle):
        angle = self.angle - relative_angle
        self.angle = self._angle_quantize(angle)
        self._angle_to_dir()

class Agent(simEntity):
    name: str
    direction: Compass
    task_list: list
    current_path: list
    current_location: None
    env: simEnvironment
    _route_method: None
    
    def __init__(self, name, env, current_location = 1, next_location = 2, task_list = [], current_path = []):
        self.name = name
        self.direction = Compass(points=8) # non-graph bound orientation
        self.env = env
        self.current_location = current_location
        self.next_location = next_location
        self.task_list = task_list
        self.current_path = current_path
        self._route_method = self._calc_shortest_path # allow for alternative routing methods

    def acquire_tasklist(self):
        """ When tasklist is empty, generate new tasks """
        pass
    
    def _calc_shortest_path(self, target=None):
        """ Generate path to next location. Either provide a target or utilise class property next_location """
        if not target:
            target = self.next_location
        g = self.env.sim_graph
        if g:
            try:       
                self.current_path(nx.shortest_path(g,source=self.current_location,target=target, weight='weight', method='dijkstra'))
                self.next_location = self.current_path[0]
            except Exception as e:
                raise Exception(f"Error performing shortest path for Agent {self.name}: {e}")
            
    def update_location_and_route(self, current_location):
        """ Called upon new location arrival, or with intent to recalculate route with new edge weightings/environment conditions """
        self.current_location = current_location
        if self.current_location == self.next_location:
            self.next_task()
            pds.verbose_print("Current Task list {self.current_location} completed for {self.name}")
        else:
            self._route_method(self, self.next_location)

    def next_task(self):
        if len(self.task_list)>1:
            del self.task_list[0]
            self.next_location = self.task_list[0]
            self._route_method(self, self.next_location)
            pds.verbose_print("Task list completed for {self.name}")
        else:
            self.acquire_tasklist()