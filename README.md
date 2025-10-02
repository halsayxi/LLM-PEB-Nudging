# Understanding the Persistence of Pro-environmental Nudges through an Integrated LLM-Human Approach

## Install Environment

To set up the environment and dependencies, run the following commands:

### Python Environment
```bash
conda create -n ProEnvironment python=3.10
conda activate ProEnvironment
pip install -r requirements.txt
```
### R Installation
Please install R by following the official instructions: https://cran.r-project.org/
Installation steps for different operating systems (Windows / macOS / Linux) are provided on the official page.

## Configure API

1. Edit utils.py and set your API key and optional base URL:
```python
client = OpenAI(
    base_url="https://api.openai.com",  # Replace if using a custom endpoint
    api_key="your_api_key_here"         # Replace with your OpenAI API key
)
```
2. Save the file. The gpt_res function will now use your API key to call the OpenAI API.

## Running the Experiments
You can choose between two ways to run the experiments:

1. Run all experiments together with a single command:
```bash
./run.sh
```
2. Run each experiment separately by following the instructions provided in the sections below.
   
### Study 1: LLMs recapitulate human pro-environmental nudge effects

#### Short-Term Experiments
```bash
cd study1/nudge_replication
python run.py
```

#### Long-Term Experiments
```bash
cd study1/nudge_replication
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

#### Result Analysis
```bash
cd study1/result_analysis
python result_analysis.py
python result_analysis_long.py
```


### Study 2: Longitudinal simulations reveal temporal decay of nudge effects
```bash
cd study2/longitudinal_simulation
python run_long.py # participant receives only one nudge
python run_freq.py # participants receive nudges according to a schedule
```

#### Parameters
| **Parameter**  | **Type** | **Description**                                         |
| -------------- | -------- | ------------------------------------------------------- |
| `--num_rounds` | `int`    | Number of simulation rounds (experiment iterations).    |
| `--start_id`   | `int`    | Experiment ID to start from.                            |
| `--frequency`  | `int`    | Interval between nudges (in rounds).                    |

#### Result Analysis
```bash
cd study2/result_analysis
python result_analysis.py
```

### Study 3: Human experiment validates decay and identifies dual pathways of persistence

#### Result Analysis
```bash
cd study3/result_analysis
Rscript analysis.R
```

### Study 4: Network diffusion consolidates early adoption and shapes long-term persistence

```bash
cd study4/social_simulation
python run_contact.py
```

#### Parameters
| **Parameter**         | **Type** | **Description**                                                                                                                                                                                                                                      |
| --------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-activation_rate`    | `float`  | Probability that an edge is activated during the simulation (edge activation rate).                                                                                                                                                                  |
| `--network_structure` | `str`    | Network type to use in the simulation: `swn`, `hcn`, `sbn`. They represent: <br>• `swn` → Watts-Strogatz (small-world networks) <br>• `hcn` → Barabási-Albert (scale-free networks) <br>• `sbn` → Stochastic-Block model (community-based networks). |
| `--target_studies`    | `str`    | IDs of the experiments to run.                                                                                                                                                                                                                       |

#### Result Analysis
```bash
cd study4/result_analysis
python result_analysis.py
```
