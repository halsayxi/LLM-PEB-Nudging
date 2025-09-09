from scipy import stats
from scipy.stats import chi2_contingency
import numpy as np
from openai import OpenAI
import httpx
from scipy.stats import norm
import json
import os
import argparse
import re
import math
import random


z = norm.ppf(0.975)


def check_file_path(path):
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"{path} is not a valid file path.")
    return path


def get_system_prompt():
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(utils_dir, "prompt", "system_prompt.json")
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt_data = json.load(f)
    instruction = system_prompt_data.get("instruction", "")
    traits = system_prompt_data.get("behavioral_traits", [])
    system_prompt = instruction + " " + " ".join(traits)
    return system_prompt


system_prompt = get_system_prompt()

client = OpenAI(
    base_url="",
    api_key= "",
)

def gpt_res(role, exp, model_name, temperature):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": role},
            {"role": "user", "content": exp},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


def get_res(role, exp, model_name, temperature, system_prompt=system_prompt):
    role = role.strip()
    system_prompt = system_prompt.strip()
    role = role + " " + system_prompt
    exp = exp.strip()
    message = role + "\n" + exp
    res = gpt_res(role, exp, model_name, temperature)
    message_parts = message.split("\n")
    result = {
        "input": [part for part in message_parts],
        "output": res,
    }
    return result


def get_fixed_response(
    role,
    exp,
    model_name,
    temperature,
    cha_num,
    max_attempts,
    is_binary,
    round_num=None,
):
    attempt = 0
    while attempt < max_attempts:
        chara_res = get_res(role, exp, model_name, temperature)
        output = chara_res.get("output", "")
        if output:
            output = output.strip()
            number_match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", output)
            if number_match:
                num_str = number_match.group(0)
                if is_binary:
                    try:
                        if num_str in ["0", "1"]:
                            output_num = int(num_str)
                            if round_num:
                                return {
                                    "round": round_num,
                                    "input": chara_res["input"],
                                    "output": output,
                                    "result": output_num,
                                }
                            else:
                                return {
                                    "input": chara_res["input"],
                                    "output": output,
                                    "result": output_num,
                                }
                    except ValueError:
                        print(f"Invalid numeric output: {output}")
                else:
                    try:
                        output_num = int(num_str)
                        if round_num:
                            return {
                                "round": round_num,
                                "input": chara_res["input"],
                                "output": output,
                                "result": output_num,
                            }
                        else:
                            return {
                                "input": chara_res["input"],
                                "output": output,
                                "result": output_num,
                            }
                    except ValueError:
                        try:
                            output_num = float(num_str)
                            if round_num:
                                return {
                                    "round": round_num,
                                    "input": chara_res["input"],
                                    "output": output,
                                    "result": output_num,
                                }
                            else:
                                return {
                                    "input": chara_res["input"],
                                    "output": output,
                                    "result": output_num,
                                }
                        except ValueError:
                            print(f"Invalid numeric output: {output}")
        if attempt > 0:
            print(f"⚠️ Attempt {attempt+1} failed for chara {cha_num}, retrying...")
        attempt += 1
    print(f"⚠️ Failed to parse output after {max_attempts} attempts.")
    if round_num:
        return {
            "round": round_num,
            "input": chara_res["input"],
            "output": output,
            "result": output_num,
        }
    else:
        return {
            "input": chara_res["input"],
            "output": output,
            "result": output_num,
        }


def cohen_d_from_proportions(
    mean_intervention,
    mean_control,
    std_intervention,
    std_control,
    n_intervention,
    n_control,
    is_binary,
):
    if is_binary:
        if not (0 <= mean_intervention <= 1) or not (0 <= mean_control <= 1):
            raise ValueError("Proportions must be between 0 and 1 when is_binary=True.")
        phi_intervention = 2 * math.asin(math.sqrt(mean_intervention))
        phi_control = 2 * math.asin(math.sqrt(mean_control))
        d = phi_intervention - phi_control
        return d
    else:
        pooled_sd = math.sqrt(
            (
                (n_intervention - 1) * std_intervention**2
                + (n_control - 1) * std_control**2
            )
            / (n_intervention + n_control - 2)
        )
        d = (mean_intervention - mean_control) / pooled_sd
        return d


def p_value_and_stat_from_proportions(
    mean_intervention,
    mean_control,
    std_intervention,
    std_control,
    n_intervention,
    n_control,
    is_binary,
):
    if is_binary:
        success_intervention = mean_intervention * n_intervention
        fail_intervention = n_intervention - success_intervention
        success_control = mean_control * n_control
        fail_control = n_control - success_control
        contingency_table = [
            [success_intervention, fail_intervention],
            [success_control, fail_control],
        ]
        table = np.array(contingency_table)
        if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
            return None, None, None

        try:
            chi2, p_value, dof, expected = chi2_contingency(table)
            if (expected == 0).any():
                return None, None, None
            return chi2, p_value, dof
        except ValueError:
            return None, None, None
    else:
        if std_intervention == 0 and std_control == 0:
            if mean_intervention == mean_control:
                t_stat = 0.0
                p_value = 1.0
                df = n_intervention + n_control - 2
                return t_stat, p_value, df
            else:
                return None, None, None

        t_stat, p_value = stats.ttest_ind_from_stats(
            mean1=mean_intervention,
            std1=std_intervention,
            nobs1=n_intervention,
            mean2=mean_control,
            std2=std_control,
            nobs2=n_control,
            equal_var=False,
        )
        numerator = (
            std_intervention**2 / n_intervention + std_control**2 / n_control
        ) ** 2
        denominator = ((std_intervention**2 / n_intervention) ** 2) / (
            n_intervention - 1
        ) + ((std_control**2 / n_control) ** 2) / (n_control - 1)
        df = numerator / denominator
        return t_stat, p_value, df


def calculate_mean_std_result(res_list):
    values = []
    for item in res_list:
        result = item.get("result")
        if result is None:
            continue
        if isinstance(result, (int, float)):
            values.append(result)
    count = len(values)
    if count == 0:
        return None, None, 0
    mean_result = sum(values) / count
    variance = (
        sum((x - mean_result) ** 2 for x in values) / (count - 1) if count > 1 else 0
    )
    std_result = math.sqrt(variance)
    return mean_result, std_result, count


def compute_var_d(n1, n2, d):
    var_d = (n1 + n2) / (n1 * n2) + (d**2) / (2 * (n1 + n2))
    return var_d


def analyze_results(study_id, res_control, res_intervention, exp_data):
    mean_control, std_control, n_control_new = calculate_mean_std_result(res_control)
    mean_intervention, std_intervention, n_intervention_new = calculate_mean_std_result(
        res_intervention
    )

    d_llm = cohen_d_from_proportions(
        mean_intervention,
        mean_control,
        std_intervention,
        std_control,
        n_intervention_new,
        n_control_new,
        exp_data["binary_outcome"],
    )
    stat_llm, p_llm, df_llm = p_value_and_stat_from_proportions(
        mean_intervention,
        mean_control,
        std_intervention,
        std_control,
        n_intervention_new,
        n_control_new,
        exp_data["binary_outcome"],
    )
    test_stat_name = "chi2" if exp_data["binary_outcome"] == 1 else "t_stat"
    var_llm = compute_var_d(exp_data["n_control"], exp_data["n_intervention"], d_llm)
    ci_lower_llm = d_llm - z * np.sqrt(var_llm)
    ci_upper_llm = d_llm + z * np.sqrt(var_llm)
    if study_id in ["Dickerson_1", "Dickerson_2"]:
        d_llm = d_llm * -1
        ci_new_lower_llm = -ci_upper_llm
        ci_new_upper_llm = -ci_lower_llm
        ci_upper_llm, ci_lower_llm = ci_new_upper_llm, ci_new_lower_llm
    llm_result = {
        "mean_control": mean_control,
        "mean_intervention": mean_intervention,
        "std_control": std_control,
        "std_intervention": std_intervention,
        "cohens_d": d_llm,
        test_stat_name: stat_llm,
        "p_value": p_llm,
        "df": df_llm,
        "variance_d": var_llm,
        "ci_lower": ci_lower_llm,
        "ci_upper": ci_upper_llm,
    }

    try:
        d_human = cohen_d_from_proportions(
            exp_data["mean_intervention"],
            exp_data["mean_control"],
            exp_data["sd_intervention"],
            exp_data["sd_control"],
            exp_data["n_intervention"],
            exp_data["n_control"],
            exp_data["binary_outcome"],
        )
        if study_id in ["Dickerson_1", "Dickerson_2"]:
            d_human = d_human * -1
        if d_human is None or isinstance(d_human, float) and math.isnan(d_human):
            raise ValueError("Cohen's d is NaN or None")
    except Exception:
        d_human = exp_data.get("cohens_d", None)

    human_result = {
        "mean_control": exp_data["mean_control"],
        "mean_intervention": exp_data["mean_intervention"],
        "std_control": (
            exp_data.get("sd_control")
            if exp_data.get("sd_control") not in [None, ""]
            else None
        ),
        "std_intervention": (
            exp_data.get("sd_intervention")
            if exp_data.get("sd_intervention") not in [None, ""]
            else None
        ),
        "cohens_d": d_human,
        "variance_d": exp_data["variance_d"],
        "ci_lower": exp_data["ci_lower"],
        "ci_upper": exp_data["ci_upper"],
    }

    analysis_result = {
        "human_result": human_result,
        "llm_result": llm_result,
    }
    with open("analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)


def get_all_edges(data):
    edges = []
    for node, neighbors in data.items():
        for neighbor in neighbors:
            edge = tuple(sorted((int(node), neighbor)))
            if edge not in edges:
                edges.append(edge)
    return edges


def activate_edges(edges, activation_rate):
    num_edges_to_activate = int(len(edges) * activation_rate)
    random.seed(None)
    activated_edges = random.sample(edges, num_edges_to_activate)
    return activated_edges


def find_results(data, round_num):
    for entry in data:
        if entry["round"] == round_num:
            return entry["result"]
    return None


def load_friend_actions(activated_pairs, cha_num, round_num, data_store):
    friend_results = []
    num_activated_friends = 0
    for friend_id in activated_pairs.get(cha_num, []):
        num_activated_friends += 1
        data = data_store.get(friend_id - 1)
        result = find_results(data, round_num - 1)
        friend_results.append(result)
    return friend_results, num_activated_friends
