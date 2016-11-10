Toolkit for learning driving models through maximum entropy inverse reinforcement learning, and autonomous vehicle's control through leverageing effects on human actions.

(Companion code to a paper presented at RSS 2016)

### Running

To visualize:
./vis {file_name}.pickle

To run an experiment
./run {world_name}
where world_name can be any one of the worlds defined in world.py

To run an experiment with irl_ground world:
./run irl_ground

To run the IRL algorithm:
./irl.py data/*.pickle

### Modules

- dynamics.py: This contains code for car dynamics.
- car.py: Relevant code for different car models (human-driven, autonomous, etc.)
- feature.py: Definition of features.
- lane.py: Definition of driving lanes.
- trajectory.py: Definition of trajectories.
- world.py: This code contains different scenarios (each consisting of lanes/cars/etc.).
- visualize.py: This contains the code for visualization (GUI).
