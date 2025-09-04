import json
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.append(os.path.abspath("../.."))
from utils import get_fixed_response


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--num_threads", type=int, default=30)
    parser.add_argument("--max_attempts", type=int, default=5)
    return parser


def process_chara(
    cha_num,
    role,
    round_num,
    model_name,
    temperature,
    max_attempts,
    data_store,
    group,
):
    role = "Forget you are an AI model." + " " + role
    exp = "You are participating in an water usage experiment. "
    if round_num != 1:
        last_list = data_store[cha_num - 1]
        last = next((item for item in last_list if item["round"] == round_num - 1))[
            "result"
        ]
    if round_num <= 8:
        exp += f"This is week {round_num}.\n"
    elif round_num <= 21:
        exp += f"This is month {round_num-6}.\n"
    else:
        exp += f"This is quarter {round_num-16}.\n"
    if round_num <= 8 and round_num >= 4:
        if group == "base":
            exp += f"Your water usage last period is {last} liters/day.\n"
        elif group == "T1":
            last_last = next(
                (item for item in last_list if item["round"] == round_num - 2)
            )["result"]
            filtered_results = [
                item["result"] for item in last_list if item["round"] < round_num - 2
            ]
            average = (
                sum(filtered_results) / len(filtered_results)
                if filtered_results
                else None
            )
            exp += "Your Water Usage were as follows:\n"
            exp += f"Last week: {last} liters/day\n"
            exp += f"Week before last: {last_last} liters/day\n"
            exp += f"Previous average: {average} liters/day\n"
        elif group == "T2":
            exp += "The sustainable water usage target is 50-70 liters/day. Compare your usage with this target: "
            if last > 70:
                exp += "Please note that your current water usage is above the sustainable level.\n"
            else:
                exp += "Congratulations! You are approaching a sustainable level of water conservation.\n"
        elif group == "T3":
            exp += "Here are four practical and motivating tips to save water:\n"
            exp += "1. Wash fruits in a bowl instead of under running water—you can save a significant amount of water each time.\n"
            exp += "2. Turn off the tap while soaping your hands or doing laundry—each time you save 10-30 liters, which adds up quickly!\n"
            exp += "3. Try shortening your shower time—most sailors finish in under 1 minute. You'll not only save water but also have more time for other activities!\n"
            exp += f"Your water usage last period is {last} liters/day.\n"
        else:
            raise ValueError(f"No such group {group}")
    elif round_num != 1:
        exp += f"Your water usage last period is {last} liters/day.\n"
    if round_num >= 22:
        exp += (
            "We are now using marginal pricing for water consumption. "
            "This means that your water bill increases as you use more:\n"
            "the first 50 liters per person per day are charged at a low base rate, "
            "the next 50 liters at a higher rate, and any usage above 100 liters at the highest rate.\n"
            "If you reduce your daily water use, you can save money while helping the environment.\n"
        )
    exp += "Please estimate the daily household water usage (in liters per day) for this period. Give only one number. Do not add any words."
    if round_num <= 3:
        exp += "\nFor reference, typical per capita daily water usage is 80-160 liters."
    res = get_fixed_response(
        role,
        exp,
        model_name,
        temperature,
        cha_num,
        max_attempts,
        False,
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
    group,
):
    data_store = {}
    all_chara = list(all_chara)
    for round_num in range(1, 25):
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
                    data_store,
                    group,
                )
                for cha_num, role in enumerate(all_chara, start=1)
            ]
            with tqdm(
                total=len(futures), desc=f"Processing round {round_num}", ncols=100
            ) as pbar:
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)

    for cha_num, data in data_store.items():
        filename = f"{cha_num + 1}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    print(args)

    model_name = args.model_name
    temperature = args.temperature
    num_threads = args.num_threads
    max_attempts = args.max_attempts
    
    file_path_base = os.path.join("..", "..", "agent_data", "long_term_3", "character_base.json")
    file_path_T1 = os.path.join("..", "..", "agent_data", "long_term_3", "character_T1.json")
    file_path_T2 = os.path.join("..", "..", "agent_data", "long_term_3", "character_T2.json")
    file_path_T3 = os.path.join("..", "..", "agent_data", "long_term_3", "character_T3.json")
    with open(file_path_base, "r") as f:
        all_chara_base = json.load(f).values()
    with open(file_path_T1, "r") as f:
        all_chara_T1 = json.load(f).values()
    with open(file_path_T2, "r") as f:
        all_chara_T2 = json.load(f).values()
    with open(file_path_T3, "r") as f:
        all_chara_T3 = json.load(f).values()

    if not os.path.exists("res_long_term_3"):
        os.makedirs("res_long_term_3")
    os.chdir("res_long_term_3")
    if not os.path.exists("base"):
        os.makedirs("base")
    os.chdir("base")
    if any(f.endswith(".json") for f in os.listdir() if os.path.isfile(f)):
        print("Result already existed.")
    else:
        agent_experiment(
            model_name,
            temperature,
            num_threads,
            all_chara_base,
            max_attempts,
            group="base",
        )
    os.chdir("..")
    if not os.path.exists("T1"):
        os.makedirs("T1")
    os.chdir("T1")
    if any(f.endswith(".json") for f in os.listdir() if os.path.isfile(f)):
        print("Result already existed.")
    else:
        agent_experiment(
            model_name,
            temperature,
            num_threads,
            all_chara_T1,
            max_attempts,
            group="T1",
        )
    os.chdir("..")
    if not os.path.exists("T2"):
        os.makedirs("T2")
    os.chdir("T2")
    if any(f.endswith(".json") for f in os.listdir() if os.path.isfile(f)):
        print("Result already existed.")
    else:
        agent_experiment(
            model_name,
            temperature,
            num_threads,
            all_chara_T2,
            max_attempts,
            group="T2",
        )
    os.chdir("..")
    if not os.path.exists("T3"):
        os.makedirs("T3")
    os.chdir("T3")
    if any(f.endswith(".json") for f in os.listdir() if os.path.isfile(f)):
        print("Result already existed.")
    else:
        agent_experiment(
            model_name,
            temperature,
            num_threads,
            all_chara_T3,
            max_attempts,
            group="T3",
        )


if __name__ == "__main__":
    main()
