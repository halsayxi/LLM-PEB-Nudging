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
```bash
# update indices
sudo apt update -qq
# install two helper packages we need
sudo apt install --no-install-recommends software-properties-common dirmngr
# add the signing key (by Michael Rutter) for these repos
# To verify key, run gpg --show-keys /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc 
# Fingerprint: E298A3A825C0D65DFD57CBB651716619E084DAB9
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
# add the repo from CRAN -- lsb_release adjusts to 'noble' or 'jammy' or ... as needed
sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
# install R itself
sudo apt install --no-install-recommends r-base
```

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