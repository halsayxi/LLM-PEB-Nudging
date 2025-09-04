import json
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import calendar
import sys
sys.path.append(os.path.abspath("../.."))
from utils import get_fixed_response



def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--num_threads", type=int, default=30)
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--num_rounds1", type=int, default=2)
    parser.add_argument("--num_rounds2", type=int, default=24)
    parser.add_argument("--num_rounds3", type=int, default=12)
    return parser


def get_month_and_year(round_num):
    month_index = (round_num - 1) % 12 + 1
    year_index = (round_num - 1) // 12 + 1
    month_name = calendar.month_name[month_index]
    return month_name, year_index


def process_chara(
    cha_num,
    role,
    round_num,
    model_name,
    temperature,
    max_attempts,
    data_store,
    num_rounds1,
    num_rounds2,
    num_rounds3,
    group,
):
    role = "Forget you are an AI model." + " " + role
    month_name, year_index = get_month_and_year(round_num)
    round_header = f"{month_name} (Year {year_index})"
    question = f"This is {round_header}. For this month, what is your household electricity consumption in kilowatt-hours (kWh)? Please respond with a single number only, without any additional explanation or units."
    if round_num <= num_rounds1 or group == "base":
        exp = "You are participating in a household electricity usage study. "
        exp += question
    else:
        user_data = data_store.get(cha_num - 1, [])
        user_last_12 = [entry["result"] for entry in user_data[-12:]]
        months = len(user_last_12)
        neighbors_sum = [0.0] * months
        neighbors_count = 0
        for other_cha_num, values in data_store.items():
            if other_cha_num == cha_num - 1:
                continue
            other_last_12 = [entry["result"] for entry in values[-months:]]
            neighbors_sum = [s + v for s, v in zip(neighbors_sum, other_last_12)]
            neighbors_count += 1
        neighbors_avg_12 = [s / neighbors_count for s in neighbors_sum]
        start_round_num = round_num - months
        month_years = [get_month_and_year(start_round_num + i) for i in range(months)]
        month_names = [f"{month} (Year {year})" for month, year in month_years]
        your_last = user_last_12[-1]
        neighbor_last = neighbors_avg_12[-1]
        higher_or_lower = "higher" if your_last > neighbor_last else "lower"
        avg_your = sum(user_last_12) / months
        avg_neighbor = sum(neighbors_avg_12) / months
        avg_diff = avg_your - avg_neighbor
        monthly_cost_extra = avg_diff * 10
        if group == "intervention":
            nudge_limit = num_rounds1 + num_rounds2 + num_rounds3
        else:
            nudge_limit = num_rounds1 + num_rounds2
        if round_num <= nudge_limit:
            prompt = (
                f"Last month ({month_names[-1]}), your household electricity consumption was {your_last:.1f} kWh, "
                f"which is {higher_or_lower} than your neighbors' average of {neighbor_last:.1f} kWh.\n\n"
                f"Here is your electricity consumption trend over the past {months} months:\n"
            )

            for m, y, n in zip(month_names, user_last_12, neighbors_avg_12):
                prompt += f"- {m}: You used {y:.1f} kWh; neighbors' average was {n:.1f} kWh.\n"

            if avg_diff > 0:
                prompt += (
                    f"\nOn average, over the past {months} months, you used {avg_diff:.1f} kWh more per month than your neighbors, "
                    f"which costs you approximately ${monthly_cost_extra:.2f} extra each month."
                )
            elif avg_diff < 0:
                prompt += (
                    f"\nOn average, over the past {months} months, you used {abs(avg_diff):.1f} kWh less per month than your neighbors, "
                    f"saving you approximately ${abs(monthly_cost_extra):.2f} each month."
                )
            else:
                prompt += f"\nOn average, over the past {months} months, your electricity usage was about the same as your neighbors."
            exp = prompt + "\n" + question
        else:
            user_history = data_store.get(cha_num - 1, [])
            first_month_value = (
                user_history[0]["result"] if len(user_history) > 0 else "N/A"
            )
            second_month_value = (
                user_history[1]["result"] if len(user_history) > 1 else "N/A"
            )
            first_month_name, first_year = get_month_and_year(1)
            second_month_name, second_year = get_month_and_year(2)
            last_month_name = month_names[-1]
            prompt = (
                f"You used to receive a monthly household energy report, "
                f"but these reports are no longer being sent.\n\n"
                f"For your reference, here is a record of your electricity consumption:\n"
                f"- {first_month_name} (Year {first_year}): {first_month_value} kWh\n"
                f"- {second_month_name} (Year {second_year}): {second_month_value} kWh\n\n"
                f"Last month ({last_month_name}), your household electricity consumption was {your_last:.1f} kWh.\n"
            )
            exp = prompt + question
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
    num_rounds1,
    num_rounds2,
    num_rounds3,
    group,
):
    data_store = {}
    all_chara = list(all_chara)
    for round_num in range(1, num_rounds1 + num_rounds2 + num_rounds3 + 1):
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
                    num_rounds1,
                    num_rounds2,
                    num_rounds3,
                    group,
                )
                for cha_num, role in enumerate(all_chara, start=1)
            ]
            with tqdm(
                total=len(futures), desc=f"Processing month {round_num}", ncols=100
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
    num_rounds1 = args.num_rounds1
    num_rounds2 = args.num_rounds2
    num_rounds3 = args.num_rounds3

    file_path_base = os.path.join("..", "..", "agent_data", "long_term_0", "character_base.json")
    file_path_control = os.path.join("..", "..", "agent_data", "long_term_0", "character_control.json")
    file_path_intervention = os.path.join("..", "..", "agent_data", "long_term_0", "character_intervention.json")
    with open(file_path_base, "r") as f:
        all_chara_base = json.load(f).values()
    with open(file_path_control, "r") as f:
        all_chara_control = json.load(f).values()
    with open(file_path_intervention, "r") as f:
        all_chara_intervention = json.load(f).values()
    if not os.path.exists("res_long_term_0"):
        os.makedirs("res_long_term_0")
    os.chdir("res_long_term_0")
    if not os.path.exists("control"):
        os.makedirs("control")
    os.chdir("control")
    if any(f.endswith(".json") for f in os.listdir() if os.path.isfile(f)):
        print("Result already existed.")
    else:
        agent_experiment(
            model_name,
            temperature,
            num_threads,
            all_chara_control,
            max_attempts,
            num_rounds1,
            num_rounds2,
            num_rounds3,
            group="control",
        )
    os.chdir("..")
    if not os.path.exists("intervention"):
        os.makedirs("intervention")
    os.chdir("intervention")
    if any(f.endswith(".json") for f in os.listdir() if os.path.isfile(f)):
        print("Result already existed.")
    else:
        agent_experiment(
            model_name,
            temperature,
            num_threads,
            all_chara_intervention,
            max_attempts,
            num_rounds1,
            num_rounds2,
            num_rounds3,
            group="intervention",
        )
    os.chdir("..")
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
            num_rounds1,
            num_rounds2,
            num_rounds3,
            group="base",
        )


if __name__ == "__main__":
    main()
