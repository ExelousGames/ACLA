from pyaccsharedmemory import accSharedMemory
import sched, time
asm = accSharedMemory()

print("start:")


def do_something(scheduler): 
    sm = asm.read_shared_memory()
    if  (sm is not None):
        # schedule the next call first
        scheduler.enter(1, 1, do_something, (scheduler,))
        
        print(f"Gas: {sm.Physics.gas} RPM: {sm.Physics.rpm} Gear: {sm.Physics.gear} carPosition: {sm.Graphics.normalized_car_position}")

        #print("Graphics:")
        #print(f"Strategy tyre set: {sm.Graphics.penalty.name}")

        #print("Static: ")
        #print(f"Max RPM: {sm.Static.max_rpm}")

my_scheduler = sched.scheduler(time.time, time.sleep)
my_scheduler.enter(0.01, 1, do_something, (my_scheduler,))
my_scheduler.run()




asm.close()