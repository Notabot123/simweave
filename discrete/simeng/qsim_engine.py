from collections import deque
import numpy as np
from dataclasses import dataclass, field
from typing import Any
import threading
import queue
#import math
import networkx as nx

## TO DO: Create a terminus which deletes entities, perhaps perform garbage collection?
## Perhaps just create a delete method for entity

class print_display:
    def __init__(self,verbose=True):
        self.verbose = verbose
    def verbose_print(self, strMsg, env=None):
        if self.verbose:
            if isinstance(env,simEnvironment):
                time = env.current_time
                print("Time: "+str(time) + " --- " + strMsg)
            
            print(strMsg)

pds = print_display(True)

"""https://docs.python.org/3/library/queue.html"""
@dataclass(order=True)
class PrioritizedItem:
    """ Prioritized item wrapper, allows for entity being queued to not be comparable"""
    priority: int
    item: Any=field(compare=False)

@dataclass
class InventoryItems:
    """Class for keeping track of items in inventory."""
    names: list[str] 
    unit_price: list[float] 
    stock_level: list[int] 
    reorder_points: list[int] 
    repairable_prc: list[float] 
    repair_times: list[float]
    newbuy_leadtimes: list[float] 

    def __post_init__(self):
        """ Ensure same length """    
        prop_list = list(self.__dict__.values())
        n = len(prop_list[0])
        if any(len(x) != n for x in prop_list):
            raise Exception("All InventoryItem Properties must have same length")

class Warehouse(InventoryItems):
    """ Extend the InventoryItems DataClass with useful methods"""
    reorderTimeRemaining: list[float] = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.reorderTimeRemaining = np.zeros(len(self.names),1)
    def increment_by_idx(self, idx, increment=1):
        self.stock_level += increment
    def decrement_by_idx(self, idx, decrement=1):
        self.stock_level -= decrement



class simEnvironment:
    """
    May be used to track current time, for timestamping.
    May be used to store a graph of 'edges', for agents with intelligent routing to refer to.
    May be used to store end time of simulation, to halt processing
    ## Not yet used
    """
    def __init__(self, start_time=0, end_time=365*24, current_time=0, sim_graph=None):
        if start_time >= end_time:
            raise(ValueError,"Start time must before end time")
        else:
            self.start = start_time
            self.end_time = end_time
        if current_time < start_time or current_time > end_time:
            raise(ValueError,"Current time must be after start and before end time")
        else:
            self.current_time = current_time
        if isinstance(sim_graph,(type(None),nx.classes.digraph.Graph,nx.classes.digraph.DiGraph)):
            self.sim_graph = sim_graph
        else:
            raise(TypeError,"Sim Graph must be networkx graph, digraph or None(default)")
    

class simProperites:
    def __init__(self, entity_type = "default", service_time_dist = {"method":"exponential","params":{"mean":1}},
                  balk_param = {"q_len": None}, renage_param = {"q_len": None}):
        """ 
        service_time - could be stats based e.g. exp, uniform, or inverse sampled
        balk - did't even join, according to q_len or wait_time..
        renege - joined but leave according to q_len or wait_time.
        stu - might these be queue/ service specific?
        """
        self.entity_type = entity_type
        self.service_time_dist = service_time_dist
        self.balk_param = balk_param
        self.renage_param = renage_param
        self.dt = self.next_service_time()
    
    def next_service_time(self):
        if self.service_time_dist["method"] == "exponential":
            return np.random.exponential(self.service_time_dist["params"]["mean"], 1) 
        else:
            raise ValueError("ERROR: Next service time failed, no distribution recognised in SimProperties")
       


class simEntity:
    """
    An item which utilises queues and services. 
    It may be subclassed for resources which are consumed, or other active elements
    """
    def __init__(self,name,sim_properties):
        self.name = name
        self.sim_properties = sim_properties        
        self.remaining_service_time = 0
        self.current_wait_time = 0
        self.total_wait_time = 0
        self.age = 0

class entityGenerator:
    """ May be used for generating entities."""
    ## Not yet used
    def __init__(self,name,intergen_time = {'exp':1}):
        self.intergen_time = intergen_time

class Queue:   
    '''
    Supports multi-threading
    Double-Ended Queue basis
    # https://docs.python.org/3/library/collections.html#deque-objects
    '''

    def __init__(self, max_size = 10, next_q = "terminus", name = "None", env = None):
        '''
        Parameters
        ----------
        max_size : int
            Maximum number of items contained in this queue. Defaults to 10.
        next_q : Queue
            Allow chaining of queues, or subclasses such as Services. Defaults to terminus.
        env : simEnvironment
            Allows logging of current simulation time, or overall structure as nx.Graph
        '''
        self._queue = deque(maxlen=max_size) 
        self._next_q = next_q
        self.name = name
        self.env = env
    
    @property
    def next_q(self):
        return self._next_q

    @next_q.setter
    def next_q(self, new_q="terminus"):
        if new_q == "terminus":
            self._next_q = new_q
        elif isinstance(new_q, Queue):
            self._next_q = new_q
        else:
            print("Please enter a valid Queuing System or specify 'terminus' as per default")

    @next_q.deleter
    def next_q(self):
        self._next_q = "terminus"
        print("Next Queue reference deleted, defaulting to terminus")

    def __str__(self):
        return "Queue Class with _queue: {0}, next_q: {1}, name: {2}".format(self._queue, self.next_q, self.name)

    def __repr__(self):
        return "Queue([{0},{1},{2}])".format(self._queue, self.next_q, self.name)


    def enqueue(self, item):
        '''
        Queues the passed item (i.e., pushes this item onto the tail of this
        queue).

        If this queue is already full, the item at the head of this queue
        is silently removed from this queue *before* the passed item is
        queued.
        '''
        if isinstance(item,simEntity):
            item.current_wait_time = 0
        self._queue.append(item)


    def dequeue(self):
        '''
        Dequeues (i.e., removes) the item at the head of this queue *and*
        returns this item.

        Raises
        ----------
        IndexError
            If this queue is empty.
        '''

        return self._queue.popleft()
    
    def renege(self):
        '''
        Dequeues (i.e., removes) the item at the tail of this queue *and*
        returns this item.

        Raises
        ----------
        IndexError
            If this queue is empty.
        '''

        return self._queue.pop()
    
    def dequeue_and_forward_on(self, q=None):
        """
        Forward item to a chained queue or service.
        This defaults to class attribute next_q but may have another queue specified.
        """
        e = self.dequeue()
        
        if q == None:
            """ use next_q as defined in class property """
            if self.next_q == "terminus":
                pds.verbose_print(str(e.name) + " was dequeued from " + self.name + " and sent to terminus")
            else:
                try:
                    self.next_q.enqueue(e)
                    pds.verbose_print(str(e.name) + " was dequeued from " + self.name + "to" + self.next_q.name)
                except Exception as e:
                    raise Exception("Unable to forward item to next queue. Ensure specified queue or terminus (default) as class property." +
                          "\n" + repr(e))
        else:
            try:
                """ use next_q as passed to this function as argument (override class property setting) """
                q.enqueue(e)
                pds.verbose_print(str(e.name) + " was dequeued from " + self.name + " to " + q.name)
            except Exception as e:
                    raise Exception("Unable to forward item to next queue. Ensure specified queue or terminus (default) as input argument." +
                          "\n" + repr(e))
            
    def pending_tasks(self, elapsed=1):
        """ increment age and waiting times """
        for item in self._queue:
            item.age += elapsed
            item.current_wait_time += elapsed
            item.total_wait_time += elapsed

class ResourcePool(Queue):
    """
    A pool of Resources - entities that are utilised in services
    Resources may be released and acquired from these pools
    """
    def __init__(self, max_size = 10, next_q = "home", name="Resource"):
        Queue.__init__(self, max_size, next_q, name)
        self._queue = deque(maxlen=max_size)

    def __str__(self):
        return "ResourcePool Class with _queue: {0}, next_q: {1}, name: {2}".format(self._queue, self.next_q, self.name)

    def __repr__(self):
        return "ResourcePool([{0},{1},{2}])".format(self._queue, self.next_q, self.name)

    def _returnHome(self, resource):
        self.enqueue(resource)

    def addResource(self, resource):
        """ adds Resources to Pool """
        super().enqueue(item=resource)
        

    def acquireResource(self):
        """ Request Resource from Pool """
        r = super().dequeue()
        r.homePool = self
        return r

    def forwardResource(self, pool):
        try:
            super().dequeue_and_forward_on(self, q=pool)
        except Exception as e:
            raise Exception("ERROR: Failed to forward Resource in ResourcePool " + str(self) +
                  "\n" + repr(e))

class Resource(simEntity):
    """
    Resources - entities that are utilised in services
    They may be 'renewable' or consumable
    """
    def __init__(self,name,sim_properties,homePool="terminus"):
        super().__init__(name,sim_properties)
        self.homePool = homePool
    
    def releaseResource(self, newPool=None):
        """ Defaults to return home """
        if newPool is None:
            if isinstance(self.homePool,ResourcePool):
                try:
                    self.homePool._returnHome(self)
                except Exception as e:
                    raise(ValueError,"ERROR: Sending "+ self.name +" to Resource Pool listed for home routing." +
                        "\n" + repr(e))
            elif self.homePool == "terminus":
                pass
            else:
                raise(TypeError,"Specify a Resource Pool, terminus or None for " + self.name)
        elif isinstance(newPool,ResourcePool):
            try:
                newPool._returnHome(self)
            except Exception as e:
                raise(ValueError,"ERROR: Routing to Specified Resource Pool failed." +
                    "\n" + repr(e))
        else:
            raise(TypeError,"Must be a ResourcePool specified, or terminus.")

    


class PrioritySystem():
    def __init__(self, max_size = 10, next_q = "terminus", name="PioritySystem"):
        self._queue = queue.PriorityQueue(maxsize=max_size)
        self.next_q = next_q
    def enqueue(self, item):
        self._queue.put(item)
    def dequeue(self, item):
        self._queue.get(item)

class Service(Queue):
    """
    This class has a queue on input, and several parallel processing blocks.
    These 'work queues' will process each timestep, and move onto next items if items complete.
    It is assumed items will be forwarded, and newly pulled items will assume index 0 however,
    if Output Port is blocked, procesing will increment index (idx) according to work_q_overflow.
    When work_q_overflow is reached due to port blocking, no new items can enter the queue.
    """

    def __init__(self, max_q_size = 10, next_q = "terminus", name = "Service", work_capacity = 6, resources = None, work_q_overflow = 5, env=None):
        Queue.__init__(self, max_q_size, next_q, name, env=env)
        self.capacity = work_capacity # if someone tries to change?
        self.resources = resources
        self.work_channels = [Queue(max_size=work_q_overflow, next_q=next_q, name="workq "+str(q)) for q in range(work_capacity)]

    
    @Queue.next_q.setter
    def next_q(self, new_q="terminus"):
        """ 
        Differs from setter inherited due to work queues needing updating.
        Use of Parent.Property.fset ensures behaves as Property from Parent class
        """
        if new_q == "terminus":
            """ effectively self._next_q = new_q""" 
            Queue.next_q.fset(self, new_q)
            self.update_workq_next_q()
        elif isinstance(new_q, Queue):
            """ effectively self._next_q = new_q"""
            Queue.next_q.fset(self, new_q)
            self.update_workq_next_q()
        else:
            print("Please enter a valid Queuing System or specify 'terminus' as per default")

    def __str__(self):
        return "Service with _queue: {0}, next_q: {1}, name: {2}, capacity:{3}, resources:{4}".format(
            self._queue, self.next_q, self.name, self.capacity, self.resources)

    def __repr__(self):
        return "Service([{0},{1},{2},{4},{5}])".format(
            self._queue, self.next_q, self.name, self.capacity, self.resources)
    
    def enqueue(self, item):
        item.remaining_service_time = item.sim_properties.dt
        item.sim_properties.dt = item.sim_properties.next_service_time()
        pds.verbose_print(item.name + " enqueued to " + self.name)
        super().enqueue(item)

    

    def update_workq_next_q(self):
         """ allows these to update when property updates """
         for i in range(len(self.work_channels)):
            self.work_channels[i].next_q = self.next_q 

    def _channel_processing(self, work_q, elapsed):
        """ 
        Decrements remaining processing and 
        attempts to process leftovers from zeroth item onto any remaining items if Output Port is blocked.
        """
        leftovers = elapsed
        idx = 0
        while leftovers > 0 and idx < len(work_q._queue):
            """ process items. If completed draw the next item and begin """           
            dt = min(work_q._queue[idx].remaining_service_time, leftovers)
            work_q._queue[idx].remaining_service_time += -dt
            work_q._queue[idx].age += elapsed
            leftovers -= dt
            pds.verbose_print("processing: " + str(work_q._queue[0].name) + " service name: " + self.name +
                      " q number: " + work_q.name +
                    "\n remaining service time: " + str(work_q._queue[0].remaining_service_time),self.env)
            if work_q._queue[idx].remaining_service_time <= 0:
                # if item complete, check we can forward, then do so.
                if work_q.next_q == "terminus":
                    work_q.dequeue_and_forward_on()
                elif len(work_q.next_q._queue) == work_q.next_q._queue.maxlen:
                    pds.verbose_print(self.name + " Output port blocked on entering " + work_q.name)
                    # increment index, so if we have leftovers they are applied to next item, since idx=0 is blocked 
                    idx +=1
                else:
                    work_q.dequeue_and_forward_on()

                if self._queue:
                    # if there are things waiting, let them enter work queue, unless it's hit it's max length
                    if idx < work_q._queue.maxlen:
                        self.dequeue_and_forward_on(q=work_q)
                    else:
                        pds.verbose_print("There is working capacity, but no queue space at " + work_q.name + ". This is caused by blocked Output ports. No productive work completed this timestep")
                    

    def process_workchannel(self, work_q, elapsed=1):
        """ This method will forward completed items, if output is non-blocked.
        It will assign tasks for processing using '_channel_processing' """        
        if not(work_q._queue):
            # the work q was empty so try to draw new item
            if self._queue:
                # if there are things waiting
                self.dequeue_and_forward_on(q=work_q)
        """ There may have been items already, or we may have drawn one """
        if work_q._queue:
            if self.resources is None:
                # infinite resources
                self._channel_processing( work_q, elapsed)                 
                
            elif isinstance(self.resources, ResourcePool):
                try:
                    r = self.resources.acquireResource()
                    self._channel_processing(work_q, elapsed) 
                    r.releaseResource()
                except Exception as e:
                    pds.verbose_print("No Resources when Requesting from " + str(self.resources) + "\n issue: " + e)
            else:
                raise ValueError("RESOURCES ERROR --UNDEFINED WHEN PROCESSING " + str(work_q._queue[0]) + 
                                "\n Only ResourcePool or 'None' allowed, for infinite/unconstrained resourcing")


    def process_items(self, elapsed=1):
        """ 
        process items as a service. Parallel FIFO with channels driven by 'capacity'
        
        """
        for channel in range(self.capacity):
            # add new items to work queues, and process them
            work_q = self.work_channels[channel]        
            self.process_workchannel( work_q=work_q, elapsed=elapsed)

        # increment age and wait times for items in the pending wait queue
        self.pending_tasks(elapsed=elapsed)
                    
   
    
if __name__ == "__main__":
    queue1 = Queue()
    queue1.enqueue('Alpha')
    queue1.enqueue('Beta')
    queue1.enqueue('Gamma')
    queue1.enqueue('I want to leave please. Too long waiting')
    print(queue1.dequeue())
    print(queue1.renege())
    # could use directly..deque
    q = deque(maxlen=10)
    q.append("Alpha")
    q.append("Beta")
    q.append("Gamma")
    q.append("I want to leave please. Too long waiting")
    print(q.pop())

    # ok now we added new bits, let's see if we can fwd onwards ok.
    queue1 = Queue()
    print("new sim: chain queues")
    print("--------------")
    q2 = Queue(10,next_q=queue1)
    p = simProperites()
    entities = []
    for _ in range(4):
        e = simEntity('object'+str(_), sim_properties=p)
        entities.append(e)
        q2.enqueue(e)
    for _ in range(4):
        q2.dequeue_and_forward_on()

        print(queue1._queue.pop().name)
        #print(q2.next_q._queue.pop()) #matches
    print(len(queue1._queue))
    print("New sim - services")
    print("--------------")

    env = simEnvironment()
    s = Service(work_capacity=1, env=env)
    p = simProperites()
   
    entities = []
   
    for i in range(3):
        #e = PrioritizedItem(priority=1,item=simEntity('object'+str(i), sim_properties=p)) # to do
        e = simEntity('object'+str(i), sim_properties=p)
        s.enqueue(e)
       
        #print("q len is now:")
        #print(len(s._queue))
    elapsed = 1
    for i in range(12):
         if isinstance(env,simEnvironment):
            ## stu we will want to migrate this as there could be concurrent services
            env.current_time += elapsed
         s.process_items(elapsed=elapsed)

    print("try the print str")
    print(s)
