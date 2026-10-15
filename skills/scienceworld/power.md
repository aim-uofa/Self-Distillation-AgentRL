### TASK: power ###

For powering an electrical component, apply these strategies:

1. Parse Device and Source Constraints: Identify the component that must operate and whether the task restricts the energy source. Separate the requested source type from the observed device state, and follow any explicitly required focus instruction before building the circuit.

2. Inspect Available Equipment: Locate the specified source, wires, and target device. Read the terminal names from observations and inspect unfamiliar components. Do not assume the wire colors or source arrangement from another episode are present here.

3. Plan a Complete Current Path: Connect one source terminal through the target device and return to the other source terminal using available wires. Both sides must be connected; a single connection does not establish a working closed circuit.

4. Check Operational Preconditions: Inspect any relevant switch or activation state and verify that the chosen energy source can operate in its current setting. If the device remains off, check missing connections before repeatedly reconnecting already-correct terminals.

5. Observe Device Operation: Allow a short wait if the simulation needs to advance, then inspect the target device. Use its observed active state as evidence rather than treating successful connection messages alone as proof that the device is powered.

6. Finish Without Breaking the Circuit: Carry out any remaining focus requirement after verifying operation. Avoid moving connected components or dismantling a working loop unless the task explicitly requests it, since these actions can undo the achieved state.
