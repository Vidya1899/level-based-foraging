# Level-Based Foraging Environment

### Built With
This section should list any major frameworks that you built your project using. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.
* [Python](https://www.python.org)
* [rllib](https://github.com/ray-project/ray/tree/master/rllib)
* [OpenAI's Gym](https://gym.openai.com/)
* [pyglet](https://github.com/pyglet/pyglet)


<!-- GETTING STARTED -->
## Getting Started

### Installation

Install using pip
```sh
pip install lbforaging
```

<!-- USAGE EXAMPLES -->
## Usage

Create environments with the rllib framework.

Register your own variation using (change parameters as needed):
```python
from gym.envs.registration register

register(
    id="Foraging-{0}x{0}-{1}p-{2}f{3}-v0".format(s, p, f, "-coop" if c else ""),
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": p,
        "max_player_level": 3,
        "field_size": (s, s),
        "max_food": f,
        "sight": s,
        "max_episode_steps": 50,
        "force_coop": c,
    },
)
```

Similarly to Gym, but adapted to multi-agent settings step() function is defined as
```python
nobs, nreward, ndone, ninfo = env.step(actions)
```

Where n-obs, n-rewards, n-done and n-info are LISTS of N items (where N is the number of agents). The i'th element of each list should be assigned to the i'th agent.

actions is a LIST of N INTEGERS (one of each agent) that should be executed in that step. The integers should correspond to the Enum below:

```python
class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5
```


<!-- CITATION -->
## Citation
1. A comperative evaluation of MARL algorithms that includes this environment
```
@article{papoudakis2020comparative,
  title={Comparative Evaluation of Multi-Agent Deep Reinforcement Learning Algorithms},
  author={Papoudakis, Georgios and Christianos, Filippos and Sch{\"a}fer, Lukas and Albrecht, Stefano V},
  journal={arXiv preprint arXiv:2006.07869},
  year={2020}
}
```
2. A method that achieves state-of-the-art performance in many Level-Based Foraging tasks
```
@inproceedings{christianos2020shared,
  title={Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning},
  author={Christianos, Filippos and Sch{\"a}fer, Lukas and Albrecht, Stefano V},
  booktitle = {Advances in Neural Information Processing Systems},
  year={2020}
}
```


