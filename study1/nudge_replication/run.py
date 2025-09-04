import json
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import sys
sys.path.append(os.path.abspath("../.."))
from agent_data.generate_profile import generate_and_save_population
from agent_data.attribute_to_discription import process_agent_descriptions
from utils import get_fixed_response, analyze_results


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_threads", type=int, default=15)
    parser.add_argument("--max_attempts", type=int, default=5)
    return parser


def process_chara(
    cha_num,
    role,
    model_name,
    temperature,
    max_attempts,
    exp_data,
    control,
):
    role = "Forget you are an AI model." + " " + role
    prompt_key = (
        "control_group_scenario_prompt"
        if control
        else "intervention_group_scenario_prompt"
    )
    exp = (
        exp_data[prompt_key]
        + "\n"
        + exp_data["response_instruction"]
        + " "
        + exp_data["response_options"]
        + " Only output a number."
    )
    res = get_fixed_response(
        role,
        exp,
        model_name,
        temperature,
        cha_num,
        max_attempts,
        exp_data["binary_outcome"],
    )
    return res


def run_group(
    group_name,
    chara_list,
    num_threads,
    model_name,
    temperature,
    max_attempts,
    exp_data,
    control_flag,
):
    print(f"Processing {group_name} group...")
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(
                process_chara,
                cha_num,
                role,
                model_name,
                temperature,
                max_attempts,
                exp_data,
                control_flag,
            )
            for cha_num, role in enumerate(chara_list, start=1)
        ]
        with tqdm(total=len(futures), desc=f"{group_name} group", ncols=100) as pbar:
            for future in as_completed(futures):
                results.append(future.result())
                pbar.update(1)

    with open(f"{group_name}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def agent_experiment(
    model_name,
    temperature,
    num_threads,
    control_chara,
    intervention_chara,
    max_attempts,
    exp_data,
):
    res_control = run_group(
        "control",
        control_chara,
        num_threads,
        model_name,
        temperature,
        max_attempts,
        exp_data,
        control_flag=1,
    )
    res_intervention = run_group(
        "intervention",
        intervention_chara,
        num_threads,
        model_name,
        temperature,
        max_attempts,
        exp_data,
        control_flag=0,
    )
    analyze_results(exp_data["study_id"], res_control, res_intervention, exp_data)



def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    print(args)

    model_name = args.model_name
    temperature = args.temperature
    num_threads = args.num_threads
    max_attempts = args.max_attempts

    csv_path = os.path.join("..", "..", "prompt", "meta_new_list.csv")
    df = pd.read_csv(csv_path)

    if not os.path.exists("res"):
        os.makedirs("res")
    os.chdir("res")
    if not os.path.exists(f"{model_name}_res"):
        os.makedirs(f"{model_name}_res")
    os.chdir(f"{model_name}_res")

    for index, row in df.iterrows():
        study_id = row["study_id"]
        print(f"Processing study_id {study_id}...")
        n_control = int(row["n_control"])
        n_intervention = int(row["n_intervention"])
        population = row["population"]
        if not os.path.exists(study_id):
            os.makedirs(study_id)
        os.chdir(study_id)
        if os.path.exists("control.json") and os.path.exists("intervention.json"):
            with open("control.json", "r", encoding="utf-8") as f:
                res_control = json.load(f)
            with open("intervention.json", "r", encoding="utf-8") as f:
                res_intervention = json.load(f)
            print(f"Study_id {study_id}: Results already existed.")
            analyze_results(study_id, res_control, res_intervention, row)
            os.chdir("..")
            continue

        generate_and_save_population(n_control, population, 1)
        generate_and_save_population(n_intervention, population, 0)
        control_chara = process_agent_descriptions(n_control, population, 1)
        intervention_chara = process_agent_descriptions(n_intervention, population, 0)
        agent_experiment(
            model_name,
            temperature,
            num_threads,
            control_chara,
            intervention_chara,
            max_attempts,
            row,
        )
        os.chdir("..")
    print("All studies processed.")


if __name__ == "__main__":
    main()
