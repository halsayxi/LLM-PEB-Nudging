# Understanding the Persistence of Pro-environmental Nudges through an Integrated LLM-Human Approach

## Install Environment

To set up the environment and dependencies, run the following commands:

```bash
conda create -n ProEnvironment python=3.10
pip install -r requirements.txt
```

## Running the Experiments

### Study 1: LLM Replication

#### short-term experiments
```
python run.py
```

#### long-term experiments
```
python run_long_term_0.py # Alloctt
python run_long_term_2.py # Paunov
python run_long_term_3.py # Vivek
```

#### Parameters
| **Parameter**       | **Type** | **Description**                                              |
| ------------------- | -------- | ------------------------------------------------------------ |
| `--model_name`      | `str`    | The name of the model to be used in the experiment.          |
| `--temperature`     | `float`  | Default: 0.7.                                                |
| `--num_threads`     | `int`    | The number of threads to use for parallel execution.         |
| `--max_attempts`    | `int`    | The maximum number of attempts per dialogue.                 |

> **Note**: Parameters that are shared across studies (e.g., --model_name, etc.) are only documented in Study 1 for brevity.

### Study 2: Longitudinal simulations
```
python run_long.py # participant receives only one nudge
python run_freq.py # participants receive nudges according to a schedule
```

#### Parameters
| **Parameter**  | **Type** | **Description**                                         |
| -------------- | -------- | ------------------------------------------------------- |
| `--num_rounds` | `int`    | Number of simulation rounds (experiment iterations).    |
| `--start_id`   | `int`    | Experiment ID to start from.                            |
| `--frequency`  | `int`    | Interval between nudges (in rounds).                    |


### Study 4:

```
python run_contact.py
```

#### Parameters
| **Parameter**         | **Type** | **Description**                                                                                                                                                                                                                                      |
| --------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-activation_rate`    | `float`  | Probability that an edge is activated during the simulation (edge activation rate).                                                                                                                                                                  |
| `--network_structure` | `str`    | Network type to use in the simulation: `swn`, `hcn`, `sbn`. They represent: <br>• `swn` → Watts-Strogatz (small-world networks) <br>• `hcn` → Barabási-Albert (scale-free networks) <br>• `sbn` → Stochastic-Block model (community-based networks). |
| `--target_studies`    | `str`    | IDs of the experiments to run.                                                                                                                                                                                                                       |
