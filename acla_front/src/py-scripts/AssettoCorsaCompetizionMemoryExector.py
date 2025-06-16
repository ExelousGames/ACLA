from pyaccsharedmemory import accSharedMemory

asm = accSharedMemory()
sm = asm.read_shared_memory()

if (sm is not None):
    print("Physics:")
    print(f"Pad life: {sm.Physics.pad_life}")

    print("Graphics:")
    print(f"Strategy tyre set: {sm.Graphics.penalty.name}")

    print("Static: ")
    print(f"Max RPM: {sm.Static.max_rpm}")

asm.close()