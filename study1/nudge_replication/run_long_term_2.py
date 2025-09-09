import json
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.append(os.path.abspath("../.."))
from utils import get_res


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--num_threads", type=int, default=30)
    parser.add_argument("--max_attempts", type=int, default=5)
    return parser


def get_fixed_response(
    role,
    exp,
    model_name,
    temperature,
    cha_num,
    max_attempts,
    round_num=None,
):
    attempt = 0
    while attempt < max_attempts:
        chara_res = get_res(role, exp, model_name, temperature)
        output = chara_res.get("output", "")
        if output:
            output = output.strip()
            try:
                values = [float(x.strip()) for x in output.split(",")]
                if len(values) == 7:
                    return {
                        "round": round_num,
                        "input": chara_res.get("input", ""),
                        "output": output,
                        "electricity_kWh": values[0],
                        "warm_water_m3": values[1],
                        "feedback_engagement_sec": values[2],
                        "information_engagement_count": values[3],
                        "energy_competence": values[4],
                        "self_efficacy": values[5],
                        "social_comparison": values[6],
                    }
            except Exception as e:
                pass
        if attempt > 0:
            print(output)
            print(f"⚠️ Attempt {attempt+1} failed for chara {cha_num}, retrying...")
        attempt += 1
    print(f"⚠️ Failed to parse output after {max_attempts} attempts.")
    return {
        "round": round_num,
        "input": chara_res.get("input", ""),
        "output": output,
        "electricity_kWh": None,
        "warm_water_m3": None,
        "feedback_engagement_sec": None,
        "information_engagement_count": None,
        "energy_competence": None,
        "self_efficacy": None,
        "social_comparison": None,
    }


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
    exp = f"You are participating in an experiment. This is week {round_num}.\n"
    if round_num != 1:
        cha_data = data_store[cha_num - 1]
        total_electricity = sum(week_data["electricity_kWh"] for week_data in cha_data)
        total_warm_water = sum(week_data["warm_water_m3"] for week_data in cha_data)
        cha_data_sorted = sorted(cha_data, key=lambda x: x["round"])
        cha_recent_7_weeks = cha_data_sorted[-7:]
        other_participants = [
            data_store[i] for i in range(len(data_store)) if i != cha_num - 1
        ]
        friends_avg_weeks = []
        for i in range(len(cha_recent_7_weeks)):
            week_num = cha_recent_7_weeks[i]["round"]
            week_vals = []
            for p_data in other_participants:
                week_entry = next(
                    (entry for entry in p_data if entry["round"] == week_num), None
                )
                if week_entry:
                    week_vals.append(week_entry)
            if week_vals:
                avg_elec = sum(w["electricity_kWh"] for w in week_vals) / len(week_vals)
                avg_water = sum(w["warm_water_m3"] for w in week_vals) / len(week_vals)
            else:
                avg_elec, avg_water = 0, 0
            friends_avg_weeks.append(
                {
                    "round": week_num,
                    "electricity_kWh": avg_elec,
                    "warm_water_m3": avg_water,
                }
            )

    if group == "baseline":
        exp += f"A web link library contains various energy-saving tips for your reference.\n"
        if round_num != 1:
            if round_num < 13:
                exp += f"As of this week, your cumulative electricity usage is {total_electricity} kWh.\n"
            else:
                exp += (
                    f"As of this week, your cumulative electricity usage is {total_electricity} kWh, "
                    f"and your cumulative hot water usage is {total_warm_water} m³.\n"
                )
    elif group == "boost":
        exp += (
            "Next to each device in your daily life (e.g., oven, fridge), there is a QR code you can scan to view energy-saving tips specific to that device. "
            "It is very convenient. You can scan the QR codes with your phone to receive and store the tips.\n"
        )
        if round_num != 1:
            if round_num < 13:
                exp += f"As of this week, your cumulative electricity usage is {total_electricity} kWh.\n"
            else:
                exp += (
                    f"As of this week, your cumulative electricity usage is {total_electricity} kWh, "
                    f"and your cumulative hot water usage is {total_warm_water} m³.\n"
                )
    elif group == "nudge":
        exp += "A web link library contains various energy-saving tips for your reference.\n"
        if round_num != 1:
            if round_num < 13:
                exp += "Over the past several weeks, your and your friends' weekly electricity usage were as follows:\n"
                for i in range(len(cha_recent_7_weeks)):
                    week = cha_recent_7_weeks[i]["round"]
                    your_elec = cha_recent_7_weeks[i]["electricity_kWh"]
                    friend_elec = friends_avg_weeks[i]["electricity_kWh"]
                    exp += f"Week {week}: You ({your_elec:.2f} kWh), Friends ({friend_elec:.2f} kWh)\n"
            else:
                exp += "Over the past several weeks, your and your friends' weekly electricity and hot water usage were as follows:\n"
                for i in range(len(cha_recent_7_weeks)):
                    week = cha_recent_7_weeks[i]["round"]
                    your_elec = cha_recent_7_weeks[i]["electricity_kWh"]
                    your_water = cha_recent_7_weeks[i]["warm_water_m3"]
                    friend_elec = friends_avg_weeks[i]["electricity_kWh"]
                    friend_water = friends_avg_weeks[i]["warm_water_m3"]
                    exp += f"Week {week}: You ({your_elec:.2f} kWh / {your_water:.2f} m³), Friends ({friend_elec:.2f} kWh / {friend_water:.2f} m³)\n"
        exp += "You have set an energy-saving goal for yourself (10%).\n"
    else:
        raise ValueError(f"No such group {group}")

    exp += (
        "Please answer the following questions (separate your answers with commas, in the order listed below):\n"
        "1. Electricity used this week (kWh)\n"
        "2. Hot water used this week (m³)\n"
        "3. Total time spent checking your energy dashboard this week (seconds)\n"
        "4. Number of times you accessed energy-saving tips this week\n"
        "5. Rate your energy-saving competence (0-10)\n"
        "6. Rate your confidence in your energy-saving skills (0-10)\n"
        "7. Number of times you compared your energy usage to your peers this week\n"
        "**IMPORTANT:** Output only 7 numeric values, separated by commas, on a single line. Do not include any text, explanations, or other characters."
    )

    res = get_fixed_response(
        role,
        exp,
        model_name,
        temperature,
        cha_num,
        max_attempts,
        round_num,
    )
    if cha_num - 1 not in data_store:
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
    for round_num in range(1, 30):
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
                total=len(futures), desc=f"Processing week {round_num}", ncols=100
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
    file_path_baseline = os.path.join("..", "..", "agent_data", "long_term_2", "character_baseline.json")
    file_path_nudge = os.path.join("..", "..", "agent_data", "long_term_2", "character_nudge.json")
    file_path_boost = os.path.join("..", "..", "agent_data", "long_term_2", "character_boost.json")
    with open(file_path_baseline, "r") as f:
        all_chara_baseline = json.load(f).values()
    with open(file_path_nudge, "r") as f:
        all_chara_nudge = json.load(f).values()
    with open(file_path_boost, "r") as f:
        all_chara_boost = json.load(f).values()

    if not os.path.exists("res_long_term_2"):
        os.makedirs("res_long_term_2")
    os.chdir("res_long_term_2")
    if not os.path.exists("baseline"):
        os.makedirs("baseline")
    os.chdir("baseline")
    if any(f.endswith(".json") for f in os.listdir() if os.path.isfile(f)):
        print("Result already existed.")
    else:
        agent_experiment(
            model_name,
            temperature,
            num_threads,
            all_chara_baseline,
            max_attempts,
            group="baseline",
        )
    os.chdir("..")
    if not os.path.exists("nudge"):
        os.makedirs("nudge")
    os.chdir("nudge")
    if any(f.endswith(".json") for f in os.listdir() if os.path.isfile(f)):
        print("Result already existed.")
    else:
        agent_experiment(
            model_name,
            temperature,
            num_threads,
            all_chara_nudge,
            max_attempts,
            group="nudge",
        )
    os.chdir("..")
    if not os.path.exists("boost"):
        os.makedirs("boost")
    os.chdir("boost")
    if any(f.endswith(".json") for f in os.listdir() if os.path.isfile(f)):
        print("Result already existed.")
    else:
        agent_experiment(
            model_name,
            temperature,
            num_threads,
            all_chara_boost,
            max_attempts,
            group="boost",
        )


if __name__ == "__main__":
    main()
