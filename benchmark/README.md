# Benchmarking - caffe
The scripts 

Starlings are small to medium-sized passerine birds in the family Sturnidae. 

Flocking behaviour is the behaviour exhibited when a group of birds, called a flock, are foraging or in flight. There are parallels with the shoaling behaviour of fish, the swarming behaviour of insects, and herd behaviour of land animals.

Computer simulations and mathematical models which have been developed to emulate the flocking behaviorus of birds can generally be applied also to the "flocking" behaviour of other species. As a result, the term "flocking" is sometimes applied, in computer science, to species other than birds.

This repository is about modelling of flocking behaviour. From the perspective of the mathematical modeller, "flocking" is the collective motion of a large number of self-propelled entities and is a collective animal behaviour exhibited by many living beings such as birds, fish, bacteria, and insects. It is considered an emergent behaviour arising from simple rules that are followed by individuals and does not involve any central coordination.

# Rules which govern their motion
Basic models of flocking behaviour are controlled by three simple rules:

* Separation - avoid crowding neighbors (short range repulsion)
* Alignment - steer towards average heading of neighbors
* Cohesion - steer towards average position of neighbors (long range attraction)

With these three simple rules, the flock moves in an extremely realistic way, creating complex motion and interaction that would be extremely hard to create otherwise.

Read the [Mathematical Modelling](https://github.com/mayanksingh2298/COP290_Starlings/blob/master/Mathematical%20Modelling/COP290__Starlings.pdf) to see what rules have I used in this project.

# Let's get a bit technical shall we?
## Installation
```bash
virtualenv venv
source venv/bin/activate 
pip install -r requirements.txt
python main.py
```

## Tune the hyperparameters
Open hyperparameters.py and then change them to tune the simulation

## What all can I do?
* use W S A D to rotate the camera
* Press P to print the physical variables in the terminal
* Press R to reset the birds


# Applications
1. In Cologne, Germany, two biologists from the University of Leeds demonstrated a flock-like behaviour in humans. The group of people exhibited a very similar behavioural pattern to that of a flock, where if 5% of the flock would change direction the others would follow suit. When one person was designated as a predator and everyone else was to avoid him, the flock behaved very much like a school of fish.
2. Flocking has also been considered as a means of controlling the behaviour of Unmanned Air Vehicles (UAVs).
3. Flocking is a common technology in screensavers, and has found its use in animation. Flocking has been used in many films to generate crowds which move more realistically. Tim Burton's Batman Returns (1992) featured flocking bats, and Disney's The Lion King (1994) included a wildebeest stampede.
4. Flocking behaviour has been used for other interesting applications. It has been applied to automatically program Internet multi-channel radio stations. It has also been used for visualizing information and for optimization tasks.

# Authors

* [**Mayank Singh Chauhan**](https://github.com/mayanksingh2298)
* [**Atishya Jain**](https://github.com/atishya-jain)