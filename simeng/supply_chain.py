from dataclasses import dataclass, field
import numpy as np
from collections import deque
import scipy.stats as stats
import pandas as pd
from qsim_engine import print_display, simEnvironment
from scipy.optimize import differential_evolution, NonlinearConstraint, Bounds
import matplotlib.pyplot as plt


plt.style.use('dark_background')

pds = print_display(True)

@dataclass
class InventoryItems:
    """
    Class for keeping track of items in inventory.
    This could be used in a warehouse, or attached to a system as it's breakdown structure
    
    Attributes:
    ----------
    part_names: Merely for reference
    unit_cost: For calculating inventory costs 
    stock_level: Keep track of stock on hand
    batchsize: Control reorder sizes
    reorder_points: Define point at which reorders are made
    repairable_prc: Define if an item is repairable, or not as a % 
    repair_times: Define how long it takes to repair an item
    newbuy_leadtimes: Define how long a new buy takes, at unit cost
    shelf_life: Define whether items degrade within a Warehouse
    failure_rate: Define how often an item breaks, per unit time
    """
    """
    ## If we wish to create a regular class, rather than dataclass
    self.part_names = part_names
    self.unit_cost = unit_cost 
    self.stock_level = stock_level
    self.batchsize = batchsize
    self.reorder_points = reorder_points 
    self.repairable_prc =  repairable_prc 
    self.repair_times = repair_times
    self.newbuy_leadtimes = newbuy_leadtimes
    self.shelf_life = shelf_life
    self.failure_rate = failure_rate
    """    
    part_names: list[str] 
    unit_cost: list[float] 
    stock_level: list[float|int] 
    batchsize: list[int|float] 
    reorder_points: list[int] 
    repairable_prc: np.array 
    repair_times: np.array
    newbuy_leadtimes: np.array
    shelf_life: list[float]
    failure_rate: list[float]

    def __post_init__(self):   
        prop_list = list(self.__dict__.values())
        n = len(prop_list[0])
        if any(len(x) != n for x in prop_list):
            raise Exception("All InventoryItem Properties must have same length")
        # cast these to np.array as user may enter lists
        self.repairable_prc = np.array(self.repairable_prc)
        self.repair_times = np.array(self.repair_times)
        self.newbuy_leadtimes = np.array(self.newbuy_leadtimes)
        
class Warehouse(InventoryItems):
    """ 
    Extends the InventoryItems DataClass with useful methods

    Attributes:
    ----------
    _reorders_volume - Keeps track of how many orders of a unique item were made
    _orders_placed_bool - Keeps track of whether orders of a unique item were made
    _demand_rate - Unknown at sim outset, may be deduced from total orders/sim duration
    """
    name: str
    _reorder_time_remaining: list[float] = field(init=False)
    _reorder_queue: deque = field(init=False)
    _parent_warehouse: type(InventoryItems)| str = "industry"
    #_repair_hub: type(InventoryItems)| str = "industry" # perhaps we apply this to systems not repair houses
    _reorders_volume: np.array = field(init=False)
    _orders_placed_bool: np.array = field(init=False)
    _demand_rate: list[float] = field(init=False)

    def __init__(self, name, part_names, unit_cost, stock_level, batchsize, reorder_points,
                 repairable_prc, repair_times, newbuy_leadtimes, shelf_life, failure_rate,
                 parent_warehouse='industry'):
        
        super().__init__(part_names, unit_cost, stock_level, batchsize, reorder_points,
                 repairable_prc, repair_times, newbuy_leadtimes, shelf_life, failure_rate)
        
        self.name = name
        self._reorder_time_remaining = np.zeros((len(self.part_names)))
        self._orders_placed_bool = np.zeros((len(self.part_names)))
        self._reorders_volume = np.zeros((len(self.part_names)))
        self._lifetime_total_orders = self.stock_level
        self._lifetime_backorders = np.zeros((len(self.part_names)))
        self._parent_warehouse = parent_warehouse

    def __post_init__(self):
        """ Extend __post_init__ further, append Warehouse specific attributes """
        super().__post_init__()

    def increment_by_idx(self, idx, increment=1):
        """ Increase stock level by 1 item """
        self.stock_level[idx] += increment

    def increment_vector(self,vector):
        """
        Increase stock level by vector including all items. 
        Note: += can cause issue with float64 and int64 hence operator not used
        """
        self.stock_level = self.stock_level + vector

    def decrement_by_idx(self, idx, decrement=1):        
        avail = self.stock_level[idx] >= decrement
        if avail:
            self.stock_level[idx] = self.stock_level - decrement
        else:
            pds.verbose_print(f"Item unavailable: {self.part_names[idx]} at Warehouse: {self.name}")

    def decrement_vector(self,vector):
        avail = self.check_items_available(vector)
        self.stock_level -= (vector * avail)
        pds.verbose_print(f"Items unavailable in {self.name}, total items: {np.sum(avail==False)}")

    def awaiting_stock(self,elapsed=1):        
        self._reorder_time_remaining = np.maximum(0,self._reorder_time_remaining-elapsed)
        # if we're awaiting an item, and its await time is now zero
        if any(self._reorder_time_remaining == 0) and any(self._reorders_volume>0):
            arrivals = self._reorders_volume * (self._reorder_time_remaining == 0)
            self.increment_vector(arrivals)
            # reset orders, as these are no longer awaiting
            self._reorders_volume -= arrivals
            self._orders_placed_bool[arrivals>0] = 0

    def stock_check(self):
        # if an item is already on order..
        idx_extant = self._orders_placed_bool
        # check where reorder point is breached, negate items already on order
        new_reorders_bool = self.reorder_points >= self.stock_level
        new_reorders_bool[idx_extant==1] = 0
        reorders = new_reorders_bool * self.batchsize
        
        next_w = self._parent_warehouse
        if next_w == "industry":
            # infinite replenishment stock
            pass
        else:
            if isinstance(next_w,Warehouse):
                
                # Decrement parent, if items are available
                available = next_w.check_items_available(reorders)
                next_w.decrement_vector(reorders * available)                
              
                if any(available==0):
                    pds.verbose_print(f"Some items unavailable at: {next_w.name}, requested from {self.name}")
                    pds.verbose_print(f"Total item range unavailable: {np.sum(available==0)}")
                    self._lifetime_backorders += reorders * (available==0)
                    # update attempted new orders, for items not available
                    new_reorders_bool[available==0] = 0
                    reorders[available==0] = 0
            else:
                raise(TypeError,"For multi-echelon Supply Chains, you must specify a Warehouse class, else specify 'industry' as per default")

        # update class properties, for new orders
        self._orders_placed_bool = new_reorders_bool + self._orders_placed_bool
        self._reorders_volume = self._reorders_volume + reorders
        self._reorder_time_remaining[new_reorders_bool] = self.newbuy_leadtimes[new_reorders_bool]
        # increment lifetime orders
        self._lifetime_total_orders += reorders 

    def check_items_available(self, requested):
        """
        Check items are in stock for withdrawal using vector method.
        May be requested from downstream Warehouse, before decrementing correctly and spawning wait times.
        """
        check_bool = self.stock_level >= requested
        return check_bool
    
    def process_orders(self):
        next_w = self._parent_warehouse
        
        # Increment local stock as items arrive. Decrement await times
        self.awaiting_stock()
        # Check what needs ordering, raise demands, check availability in supply-chain
        self.stock_check()

    def estimate_demand_rate(self, total_sim_time):
        """ 
        Calculate demand rate from lifetime total orders divided by total simulation time.
        Not known a-priori, call this method at the end of a simulation run
        """
        self._demand_rate = self._lifetime_total_orders / total_sim_time

    def estimate_reorder_points(self, target_availaility=0.8, return_dataframe=True,
                                 assign_reorder_points=False, assign_stock_level=False, quantize_by_batchsize=False):
        """ 
        Calculate reorder points based on Poisson distrbution.
        This may be performed to determine sensible reorder points and stock levels
        --------
        Inputs:
        target_availabilty (default 0.8) - the product of availability for each item, which will be 1/(target_available)^Num_items
        assign_reorder_points (default False) - optionally assign result to reorder_points
        assign_stock_level (default False) - optionally assign result+batchsize to stock_level
        """
        if target_availaility<0 or target_availaility>0.999:
            raise Exception("Availability should be in the range 0-0.999")
        if np.sum(self._demand_rate) == 0:
            raise Exception("User must first run sim and then call {self.name}.estimate_demand_rate(total_sim_time)")
        else:
            demands = self._demand_rate
            logistics_delay = (self.repairable_prc*self.repair_times)+((1-self.repairable_prc)*self.newbuy_leadtimes)
            lambda_value = demands * logistics_delay
            p = target_availaility**(1/len(self.part_names))

            # Calculate k
            k = stats.poisson.ppf(p, lambda_value)
            if quantize_by_batchsize:
                # in case orders really can only be multiples of batchsize, not just a min order quantity
                k = k // self.batchsize * self.batchsize

            total_cost = np.sum(self.unit_cost*(self.batchsize+k))

            if assign_reorder_points:
                self.reorder_points = k
            if assign_stock_level:
                self.stock_level = k + self.batchsize
            
            if return_dataframe:
                df = pd.DataFrame(data=np.array([self.part_names,k,self.batchsize,self.batchsize+k]).transpose(),
                                columns=["Part Names","Reorder Point","Batch-size","Starting Stock"])

                return k, total_cost, df
            else:
                return k, total_cost
        
    def cost_optimise_stock(self, target_availaility=0.8, return_dataframe=True,
                      assign_reorder_points=False, assign_stock_level=False, quantize_by_batchsize=False):
        """
        Cost optimise, which balances risk of stock unavailability against its relative cost
        i.e. If two items have equal contribution, but one item is considerably cheaper, favour that item.
        An overall availability metric across all stock is considered as constraint whilst minimising cost
        """
        if target_availaility<0 or target_availaility>0.999:
            raise Exception("Availability should be in the range 0-0.999")
        if np.sum(self._demand_rate) == 0:
            raise Exception("User must first run sim and then call {self.name}.estimate_demand_rate(total_sim_time)")
        
        x0 = self.stock_level # initial start point
        demands = self._demand_rate
        logistics_delay = (self.repairable_prc*self.repair_times)+((1-self.repairable_prc)*self.newbuy_leadtimes)
        lambda_value = demands * logistics_delay
        if quantize_by_batchsize:
            quant = self.batchsize
        else:
            quant = 1

        def objective(stock_level):
            f = np.sum(self.unit_cost * np.round(stock_level) * quant)
            return f
        
        def ineq_con(stock_level):
            """ Ensure availability is greater than target_availaility """
            avail = np.prod(stats.poisson.cdf(np.round(stock_level), lambda_value))
            return avail
        bounds = Bounds(lb=np.zeros(len(self.part_names)), ub=100 * np.ones(len(self.part_names)))
        nlc = NonlinearConstraint(ineq_con, target_availaility, 1)
        #m = minimize(objective,x0,constraints=nlc,bounds=bounds,method='SLSQP')
        ga = differential_evolution(objective, bounds, constraints=nlc,
                                seed=1)
        solution = np.round(ga.x * quant)
        total_cost = np.sum(self.unit_cost*(self.batchsize+solution))
        if assign_reorder_points:
            self.reorder_points = solution
        if assign_stock_level:
            self.stock_level = solution + self.batchsize
        if return_dataframe:
            df = pd.DataFrame(data=np.array([self.part_names,solution,self.batchsize,self.batchsize+solution,
                                             self.unit_cost*solution,self.unit_cost*(self.batchsize+solution)]).transpose(),
                              columns=["Part Names","Reorder Point","Batch-size","Starting Stock","Stock Value on Reorder","Starting Stock Value"])
            return solution, total_cost, df
        else:
            return solution, total_cost  
        
    def pareto_optimal_sweep(self, sweep_range=np.minimum(0.99,np.arange(.1,1.1,.05)), return_verbose_results=True):
        avail = []
        cost = []
        p_cost = []
        solution_array = []
        #poiss_k_array = []
        for a in sweep_range:
            avail.append(a)
            solution, total_cost = self.cost_optimise_stock(target_availaility=a, return_dataframe=False)
            poiss_k, poiss_total_cost = self.estimate_reorder_points(target_availaility=a, return_dataframe=False)
            cost.append(total_cost)
            p_cost.append(poiss_total_cost)
            solution_array.append(solution)
            #poiss_k_array.append(poiss_k)
        #plt.bar(cost, avail, label=bar_labels, color=[0.6, 0, 0.7])
        fig, ax = plt.subplots()
        plt.fill_between(cost, avail, color=[0.0, 0.8, 0.5], alpha=0.7)
        plt.plot(cost, avail, color='green', alpha=0.5, linewidth=0.9, marker='o')
        plt.fill_between(p_cost, avail, color=[0.6, 0, 0.1], alpha=0.7)
        plt.plot(p_cost, avail, color='red', alpha=0.5, linewidth=0.9, marker='o')
        ax.legend(['Cost Optimal', 'Equal Risk'])
        ax.set_ylabel('Stock Availability')
        ax.set_xlabel('Total Cost(£)')
        ax.set_title('Pareto Optimal, Cost vs Availability')
        plt.grid(True)
        plt.show()
        if return_verbose_results:
            return cost, avail
        else:
            return cost, avail, solution_array

d = Warehouse(name="store123",part_names=['a','b','c','d','e','f'],repair_times=[1,2,3,1,2,3],unit_cost=[1,1,1,2,2,30],
              stock_level=[3,3,9,3,6,9],reorder_points=[2,2,3,5,5,5],repairable_prc=[1,1,1,1,1,1],
              newbuy_leadtimes=np.array([2,2,3,4,4,6]),shelf_life=[1,2,2,1,2,2],
              batchsize=[2,2,3,4,4,7],failure_rate=[1,2,3,1,2,3])
print(d)

for i in range(10):
    d.decrement_vector(np.array([0.2,1,1,2,2,2]))
    print("stock")
    print(d.stock_level)
    print("awaiting")
    print(d._reorder_time_remaining)
    print("waiting for")
    print(d._reorders_volume)
    print("total orders")
    print(d._lifetime_total_orders)

    d.process_orders()

d.estimate_demand_rate(total_sim_time=10)
df = d.estimate_reorder_points(0.999,assign_reorder_points=True)
df2 = d.cost_optimise_stock(0.999, return_dataframe=True)
print(d.reorder_points)
print(df)
print(df2)

d.pareto_optimal_sweep()
