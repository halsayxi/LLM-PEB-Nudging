import json
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from collections import defaultdict
import sys
sys.path.append(os.path.abspath("../.."))
from utils import get_fixed_response, load_friend_actions, get_all_edges, activate_edges
from agent_data.generate_network import (
    generate_barabasi_albert_network,
    generate_watts_strogatz_network,
    generate_stochastic_block_network,
)

default_target_studies = {
    "Bamberg_3",
    "Lofgren_2",
    "Mertens_1",
    "Hilton_1",
    "Hedlin_2",
    "Camilleri_2",
}


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_threads", type=int, default=30)
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--num_rounds", type=int, default=30)
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--activation_rate", type=float, default=0.5)
    parser.add_argument("--network_structure", type=str, default="swn")
    parser.add_argument(
        "--target_studies", type=str, default=",".join(default_target_studies)
    )
    return parser


def process_chara(
    cha_num,
    role,
    round_num,
    model_name,
    temperature,
    max_attempts,
    exp_data,
    data_store,
    activated_pairs,
    study_id,
):
    if round_num in [1, 2]:
        json_path = os.path.join("..", "..", "..", "..", "..", "..", "..", "study2", "longitudinal_simulation", "res_long", "gpt-3.5-turbo-0125_res", str(study_id), f"{cha_num}.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    all_data = json.load(f)
                    for item in all_data:
                        if item.get("round") == round_num:
                            if cha_num - 1 not in data_store:
                                data_store[cha_num - 1] = []
                            data_store[cha_num - 1].append(item)
                            return
                    print(
                        f"[Warning] No round {round_num} data found for cha_num {cha_num}"
                    )
                except json.JSONDecodeError:
                    print(
                        f"[Error] Failed to parse JSON for cha_num {cha_num} at round {round_num}"
                    )
        else:
            print(f"[Error] File not found: {json_path}")
        return

    role = "Forget you are an AI model." + " " + role
    prompt_key = (
        "control_group_scenario_prompt"
        if round_num != 2
        else "intervention_group_scenario_prompt"
    )
    exp = (
        exp_data[prompt_key]
        + "\n"
        + exp_data["response_instruction"]
        + " "
        + exp_data["response_options"]
    )

    if round_num != 1 and round_num != 2:
        historical_choices = data_store.get(cha_num - 1)
        historical_text = ""
        if historical_choices:
            historical_text = "Your historical results are as follows:\n"
            historical_text += "Short-term memory (last 7 days):\n"
            recent_choices = historical_choices[-7:]
            for entry in recent_choices:
                day = entry["round"]
                if day != 1:
                    day_text = f"Day {day}: {entry['result']}\n"
                    historical_text += day_text
            average_score = sum(item["result"] for item in historical_choices) / len(
                historical_choices
            )
            historical_text += (
                f"Long-term memory (average across all days): {average_score}\n"
            )
        exp += "\n" + historical_text
        friend_results, num_activated_friends = load_friend_actions(
            activated_pairs, cha_num, round_num, data_store
        )
        if num_activated_friends > 0:
            average_result = sum(friend_results) / len(friend_results)
            friend_count = len(friend_results)
            exp += f"Your friends' average choices from yesterday are {average_result:.2f} (based on {friend_count} friend{'s' if friend_count != 1 else ''})\n"
        exp += (
            "Previously formed habits may influence your thinking today — follow them or reassess, as you prefer."
            + " What was your result today? Only output a number."
        )
    else:
        exp += " Only output a number."
    res = get_fixed_response(
        role,
        exp,
        model_name,
        temperature,
        cha_num,
        max_attempts,
        exp_data["binary_outcome"],
        round_num,
    )
    if cha_num - 1 not in data_store:
        print(
            f"[Warning] No data found for cha_num {cha_num} in data_store. Initializing."
        )
        data_store[cha_num - 1] = []
    data_store[cha_num - 1].append(res)


def agent_experiment(
    model_name,
    temperature,
    num_threads,
    all_chara,
    max_attempts,
    exp_data,
    num_rounds,
    activation_rate,
    network_information,
    study_id,
):
    data_store = {}
    all_chara = list(all_chara)
    edges = get_all_edges(network_information)
    activated_edges_data = []
    for round_num in range(1, num_rounds + 1):
        activated_pairs = defaultdict(list)
        activated_edges = activate_edges(edges, activation_rate)
        for a, b in activated_edges:
            activated_pairs[a].append(b)
            activated_pairs[b].append(a)
        activated_edges_data_round = {
            "round": round_num,
            "num_activated_edges": len(activated_edges),
            "activated_edges": activated_edges,
        }
        activated_edges_data.append(activated_edges_data_round)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(
                    process_chara,
                    cha_num,
                    role,
                    round_num,
                    model_name,
                    temperature,
                    max_attempts,
                    exp_data,
                    data_store,
                    activated_pairs,
                    study_id,
                )
                for cha_num, role in enumerate(all_chara, start=1)
            ]
            with tqdm(
                total=len(futures), desc=f"Processing day {round_num}", ncols=100
            ) as pbar:
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)

        if round_num % 10 == 0 or round_num == num_rounds:
            print(f"Saving checkpoint at round {round_num}...")
            for cha_num, data in data_store.items():
                if len(data) % 10 == 0:
                    start_idx = len(data) - 9
                else:
                    start_idx = (len(data) // 10) * 10 + 1
                start_idx = start_idx - 1
                recent_data = data[start_idx:]
                filename = f"{cha_num + 1}.json"
                if os.path.exists(filename):
                    with open(filename, "r", encoding="utf-8") as f:
                        chara_data = json.load(f)
                else:
                    chara_data = []
                chara_data.extend(recent_data)
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(chara_data, f, ensure_ascii=False, indent=4)

    with open("activated_edges.json", "w", encoding="utf-8") as f:
        json.dump(activated_edges_data, f, indent=4, ensure_ascii=False)


def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    print(args)

    model_name = args.model_name
    temperature = args.temperature
    num_threads = args.num_threads
    max_attempts = args.max_attempts
    num_rounds = args.num_rounds
    start_id = args.start_id
    activation_rate = args.activation_rate
    network_structure = args.network_structure
    target_studies = set(
        [s.strip() for s in args.target_studies.split(",") if s.strip()]
    )

    csv_path = os.path.join("..", "..", "prompt", "meta_new_list.csv")
    df = pd.read_csv(csv_path)

    if not os.path.exists("res_contact"):
        os.makedirs("res_contact")
    os.chdir("res_contact")
    if not os.path.exists(f"{model_name}_res"):
        os.makedirs(f"{model_name}_res")
    os.chdir(f"{model_name}_res")
    for index, row in df.iterrows():
        if index < start_id:
            continue
        study_id = row["study_id"]
        if study_id not in target_studies:
            print(f"Skipping study_id {study_id}: not in target list")
            continue
        print(f"Processing study_id {study_id}...")
        if not os.path.exists(study_id):
            os.makedirs(study_id)
        os.chdir(study_id)
        if network_structure == "swn":
            if not os.path.exists("swn"):
                os.makedirs("swn")
            os.chdir("swn")
        elif network_structure == "hcn":
            if not os.path.exists("hcn_new"):
                os.makedirs("hcn_new")
            os.chdir("hcn_new")
        elif network_structure == "sbn":
            if not os.path.exists("sbn"):
                os.makedirs("sbn")
            os.chdir("sbn")
        if not os.path.exists(f"{activation_rate}_activation_rate"):
            os.makedirs(f"{activation_rate}_activation_rate")
        os.chdir(f"{activation_rate}_activation_rate")
        if any(f.endswith(".json") for f in os.listdir() if os.path.isfile(f)):
            print(f"Study_id {study_id}: Result already existed.")
            os.chdir("../../..")
            continue
        n_control = int(row["n_control"])
        n_intervention = int(row["n_intervention"])
        chara_file_path = os.path.join(
            "..", "..", "..", "..", "..", "..", "..", "study1", "nudge_replication", "res", "gpt-3.5-turbo-0125_res",
            str(study_id), "profile", f"character_control_{n_control}.json"
        )
        with open(chara_file_path, "r") as f:
            all_chara = json.load(f).values()
        if network_structure == "swn":
            generate_watts_strogatz_network(n_control)
        elif network_structure == "hcn":
            generate_barabasi_albert_network(n_control)
        elif network_structure == "sbn":
            generate_stochastic_block_network(n_control)
        network_file_path = "../profile/network_info.json"
        with open(network_file_path, "r") as f:
            data = json.load(f)
            network_information = data["Connections"]
        agent_experiment(
            model_name,
            temperature,
            num_threads,
            all_chara,
            max_attempts,
            row,
            num_rounds,
            activation_rate,
            network_information,
            study_id,
        )
        os.chdir("../../..")
    print("All studies processed.")


if __name__ == "__main__":
    main()
