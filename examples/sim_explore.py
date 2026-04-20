import simeng.qsim_engine as qsim
import time
from multiprocessing import Process
import random

pds = qsim.print_display(False)

def scenarioMaker(scene, numServices, rp):
    q=[]
    if scene == "linked-chain":
        for i in range(numServices):        
            if i > 0:
                q.append( qsim.Service(5, q[i-1], "service " + str(i), 2, rp) )
                # q[i].next_q = q[i-1] # this also updates correct child queues
            else:
                q.append( qsim.Service(5, "terminus", "service " + str(i), 2, rp) )

    elif scene == "random-chain":
        for i in range(numServices):
            q.append( qsim.Service(5, "terminus", "service " + str(i), 2, rp) )
        for i in range(numServices):
            r = random.randint(0,numServices-1)
            if r == i:
                r += 1
            q[i].next_q = q[r] # updates correct child queues
    else:
        print("Scenario not recognised!")
    return q

def defineResourcesAndServices(numServices = 10, numResources = 50):
    rp = qsim.ResourcePool(10, "home", "r_pool")
    for i in range(0,numResources):
        r = qsim.Resource(name="Res"+str(i), sim_properties=[])
        rp.addResource(r)
    
    q = scenarioMaker("linked-chain", numServices, rp)
    
    return rp, q


def runsim(simTime,numEntities,p,statusIterval,q,parall=False):
    numServices = len(q)
    for _ in range(simTime):
            for i in range(numEntities):
                #e = PrioritizedItem(priority=1,item=simEntity('object'+str(i), sim_properties=p)) # to do
                e = qsim.simEntity('object_'+str(simTime)+"_"+str(i), sim_properties=p)
                q[-1].enqueue(e)
                
            else:
                for j in range(numServices):
                    q[j].process_items()

            if (_//statusIterval==_/statusIterval):
                print("Iteration" + str(_) + ". q len is now: "+str( len(q[-2]._queue) ))

if __name__ == '__main__':
    
    p = qsim.simProperites()
    rp, q = defineResourcesAndServices(numServices = 100)
    numEntities = 2 # per time step
    simTime = 24*365
    statusIterval = simTime // 4


    start = time.time()
    runsim(simTime,numEntities,p,statusIterval,q)
    end = time.time()
    print("sim took total of: ")
    print(end - start)

    # re-initialise q. Run as Parallel Sim
    rp, q = defineResourcesAndServices(numServices = 10)
    start = time.time()
    p = Process(target=runsim, args=(simTime,numEntities,p,statusIterval,q))
    p.start()
    p.join()
    end = time.time()
    print("parallel sim took total of: ")
    print(end - start)



