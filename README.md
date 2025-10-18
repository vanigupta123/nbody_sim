# nbody_sim

ran the n-body simulation in colab using T4 GPU due to m2 macbook incompatibility with cuda

this project has two components:
- n-body simulation
- rag prompt generator

the n-body simulation generates a simulation of n particles and visualizes each particles' motion relative to the other particles, using the universal gravitation equation. 
by default, the particles are instantiated with zero velocity, so their motion is not centripetal and won't look like an orbit around any center of gravity or otherwise. 
the particles will simply move in a striaght line towards each other until they collide. when running this colab, you can optionally enable centripetal motion. in the background, 
this will simply instantiate the velocity vectors with a nonzero tangential velocity.

the rag prompt generator can interact with the inputs and outputs of the simulation and use context on the topic to then modify future simulation inputs and
generate interesting follow-up questions and prompts for subsequent simulation runs. more to be added here as i complete writing this lol
