# n-body simulation and prompt generator

ran the n-body simulation in colab using T4 GPU due to m2 macbook incompatibility with cuda

this project has two components:
- n-body simulation
- rag prompt generator

the n-body simulation generates a simulation of n particles and visualizes each particles' motion relative to the other particles, using the universal gravitation equation. 
by default, the particles are instantiated with zero velocity, so their motion is not centripetal and won't look like an orbit around any center of gravity or otherwise. 
the particles will simply move in a straight line towards each other until they collide. when running this colab, you can optionally enable centripetal motion. in the background, 
this will simply instantiate the velocity vectors with a nonzero tangential velocity. it uses cuda multithreading and global shared memory. i *partly* followed [this](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda) nvidia tutorial to do so. 

the rag prompt generator can interact with the inputs and outputs of the simulation and use context on the topic to then modify future simulation inputs and
generate interesting follow-up questions and prompts for subsequent simulation runs. more to be added here as i complete writing this lol

## how to run n-body sim colab
- you should be able to open the colab and compile everything
- the last cell will prompt you to input the number of particles, timesteps, and dt. here's what i input, as an example:
  <img width="734" height="33" alt="Screenshot 2025-10-21 at 10 41 23 PM" src="https://github.com/user-attachments/assets/25a3c367-7a33-46f0-b59e-da7939c07f51" />
- find "Files" on the left side menu bar. you can find a csv for each timestep in the `output/` folder and a visualization of the particles across all time steps in `nbody_sim.mp4`
- enjoy! and look forward to the rag description!
  
  ![nbody_sim](https://github.com/user-attachments/assets/3f7a3612-60ca-41f9-9204-e59fb40de39d)

