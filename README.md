# n-body simulation

the n-body simulation generates a simulation of n particles and visualizes each particles' motion relative to the other particles, using the universal gravitation equation. 
by default, the particles are instantiated with zero velocity, so their motion is not centripetal and won't look like an orbit around any center of gravity or otherwise. 
the particles will simply move in a straight line towards each other until they collide. when running this colab, you can optionally enable centripetal motion. in the background, 
this will simply instantiate the velocity vectors with a nonzero tangential velocity. it uses cuda multithreading and global shared memory. i *partly* followed [this](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda) nvidia tutorial to do so. 

## the architecture, in greater depth
in this problem, every particle interacts with every other particle, making it an O(N²) problem. the parallelization strategy assigns one thread per particle. each thread is responsible for computing the total gravitational force acting on its particle by iterating over all other particles. to avoid requiring every thread independently reading from global memory, the kernel uses shared memory tiling, which is where threads in a block cooperatively load a tile of particle positions into shared memory, and compute forces against that tile. this means each position is read from global memory once per tile rather than once per thread, which is the key optimization.

verlet integration is used to update positions and velocities at each timestep. compared to a naive Euler integrator, Verlet conserves energy better over long simulations, which matters here because without it, the particles would either gain or lose energy artificially over 10K timesteps, and the simulation would look physically wrong.

performance on NVIDIA A10:
| N      | time/step    | GFLOP/s  |
|--------|--------------|----------|
| 1,024  | 0.036 ms     | 590.2    |
| 4,096  | 0.141 ms     | 2,378.0  |
| 8,192  | 0.273 ms     | 4,909.8  |
| 16,384 | 0.554 ms     | 9,682.1  |

performance scales roughly linearly with N² as expected, and GFLOP/s increases with N because larger problem sizes better saturate GPU parallelism. a modern CPU running the same naive O(N²) loop typically achieves 50–100 GFLOP/s, putting the GPU roughly 20–100x faster depending on problem size!

## how to run n-body sim locally
to run the kernels and generate the position data, for 100 particles, 1000 timesteps, dt=0.05, and centripetal motion enabled:
```
nvcc -o nbody src/main.cu src/nbody.cu -O2
./nbody 100 1000 0.05 y
```

to see the animations and visualizations:
```
pip install -r requirements.txt
python scripts/visualization.py
```
then find the visualizations in the `output` folder!

to run the kernels and see the benchmarking/perf metrics, for N=8192 in this example:
```
nvcc -o benchmark src/main.cu src/nbody.cu -O2
./benchmark 8192
```

## how to run n-body sim in colab
- you should be able to open the colab and compile everything
- the last cell will prompt you to input the number of particles, timesteps, dt, and if the motion is centripetal or not. here's what i input, as an example:
  <img width="1126" height="35" alt="Screenshot 2025-10-26 at 1 25 34 AM" src="https://github.com/user-attachments/assets/fad681cb-a124-4c3f-9b22-65fe7fa83250" />
  for the "centripetal" piece, you can input "centripetal", "yes", "y", or just fail to enter anything if you want the particles to move linearly and collide with one another. you can see the result of this in the second graphic below, which was made using the parameters "100,1000,0.05".
- find "Files" on the left side menu bar. you can find a csv for each timestep in the `output/` folder and a visualization of the particles across all time steps in `nbody_sim.mp4`
- enjoy!
  
  ![nbody_sim (1)](https://github.com/user-attachments/assets/452d8697-3b27-430c-9176-630925204946)

  ![nbody_sim](https://github.com/user-attachments/assets/3f7a3612-60ca-41f9-9204-e59fb40de39d)

