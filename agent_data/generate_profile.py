import random
import json
import os
import time

data_distribution = {
    "Gender": {
        "type": "categorical",
        "values": ["Male", "Female"],
        "probabilities": [0.511, 0.489],
    },
    "AgeBracket": {
        "type": "categorical",
        "values": [
            "18-19",
            "20-24",
            "25-29",
            "30-34",
            "35-39",
            "40-44",
            "45-49",
            "50-54",
            "55-59",
            "60-64",
            "65-69",
            "70-74",
            "75-79",
            "80-84",
            "85-89",
            "90-94",
            "95+",
        ],
        "probabilities": [
            0.043,
            0.061,
            0.070,
            0.094,
            0.098,
            0.084,
            0.085,
            0.105,
            0.100,
            0.070,
            0.068,
            0.054,
            0.033,
            0.020,
            0.011,
            0.004,
            0.001,
        ],
    },
    "MaritalStatus": {
        "type": "categorical",
        "values": ["Single", "Married", "Divorced", "Widowed"],
        "probabilities": [0.199, 0.712, 0.027, 0.063],
    },
    "MonthlyIncome": {
        "type": "categorical",
        "values": ["1282", "2845", "4480", "6988", "13227"],
        "probabilities": [0.2, 0.2, 0.2, 0.2, 0.2],
    },
    "EducationLevel": {
        "type": "categorical",
        "values": [
            "No Formal Education",
            "Primary",
            "JuniorHigh",
            "SeniorHigh",
            "CollegePlus",
        ],
        "probabilities": [0.04, 0.25, 0.36, 0.16, 0.19],
    },
    "Pro_Environmental_Tendency": {
        "type": "normal",
        "mean": 3,
        "std": 0.5,
        "min": 1,
        "max": 5,
    },
    "Emotional_Sensitivity": {
        "type": "normal",
        "mean": 3,
        "std": 0.5,
        "min": 1,
        "max": 5,
    },
    "Conformity_Tendency": {
        "type": "normal",
        "mean": 3,
        "std": 0.5,
        "min": 1,
        "max": 5,
    },
    "Behavioral_Inertia": {"type": "normal", "mean": 3, "std": 0.5, "min": 1, "max": 5},
}

education_by_age = {
    "0-10": {
        "values": [
            "No Formal Education",
            "Primary",
            "JuniorHigh",
            "SeniorHigh",
            "CollegePlus",
        ],
        "probabilities": [0.9, 0.1, 0, 0, 0],
    },
    "11-15": {
        "values": [
            "No Formal Education",
            "Primary",
            "JuniorHigh",
            "SeniorHigh",
            "CollegePlus",
        ],
        "probabilities": [0.05, 0.4, 0.5, 0.05, 0],
    },
    "16-20": {
        "values": [
            "No Formal Education",
            "Primary",
            "JuniorHigh",
            "SeniorHigh",
            "CollegePlus",
        ],
        "probabilities": [0.02, 0.1, 0.3, 0.4, 0.18],
    },
}


def pick_occupation_from_table():
    occupation_list = [
        ("Farmer", 0.004),
        ("Miner", 0.020),
        ("Worker", 0.219),
        ("MigrantWorker", 0.100),
        ("PowerTec", 0.022),
        ("Courier", 0.047),
        ("Programmer", 0.032),
        ("SelfEmp", 0.048),
        ("Waiter", 0.018),
        ("Trader", 0.042),
        ("HouseAgent", 0.031),
        ("Professor", 0.028),
        ("Consultant", 0.051),
        ("WaterEng", 0.016),
        ("CommunityStaff", 0.005),
        ("PrimaryTeacher", 0.119),
        ("Doctor", 0.069),
        ("Athlete", 0.009),
        ("CivilServant", 0.121),
    ]
    values = [o[0] for o in occupation_list]
    probs = [o[1] for o in occupation_list]
    return random.choices(values, weights=probs, k=1)[0]


def get_occupation(age):
    if age < 18:
        return "Student"
    elif age > 60:
        return "Retired"
    elif random.random() < 0.052:
        return "Unemployed"
    else:
        return pick_occupation_from_table()


def get_age_from_bracket(bracket):
    if bracket == "95+":
        return random.randint(95, 100)
    parts = bracket.split("-")
    return random.randint(int(parts[0]), int(parts[1]))


def generate_person(person_id, is_adult):
    person = {}
    person["id"] = person_id
    for attr, meta in data_distribution.items():
        if meta["type"] == "categorical":
            person[attr] = random.choices(
                meta["values"], weights=meta["probabilities"], k=1
            )[0]
        elif meta["type"] == "normal":
            while True:
                sample = random.gauss(meta["mean"], meta["std"])
                if meta["min"] <= sample <= meta["max"]:
                    person[attr] = round(min(max(sample, meta["min"]), meta["max"]), 2)
                    break

    age = get_age_from_bracket(person["AgeBracket"])
    person["Age"] = age

    if not is_adult:
        person["Age"] = random.randint(0, 20)
        age = person["Age"]
        if age <= 10:
            bracket = "0-10"
        elif age <= 15:
            bracket = "11-15"
        elif age <= 20:
            bracket = "16-20"
        person["AgeBracket"] = bracket
        person["MaritalStatus"] = "Single"
        edu_dist = education_by_age[bracket]
        person["EducationLevel"] = random.choices(
            edu_dist["values"], weights=edu_dist["probabilities"], k=1
        )[0]

    person["Occupation"] = get_occupation(age)
    return person


def generate_population_with_ids(num_samples, is_adult):
    population = []
    for person_id in range(1, num_samples + 1):
        population.append(generate_person(person_id, is_adult))
    return population


def save_population_to_file(population, filename):
    with open(filename, "w") as f:
        json.dump(population, f, indent=4, ensure_ascii=False)


def wait_for_file(path, timeout=2):
    start = time.time()
    while not os.path.exists(path):
        if time.time() - start > timeout:
            raise TimeoutError(f"File {path} not created in time.")
        time.sleep(0.1)


def generate_and_save_population(num_agents, is_adult, control):
    folder_name = f"profile"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if control:
        file_path = os.path.join(folder_name, f"agent_data_control_{num_agents}.json")
    else:
        file_path = os.path.join(
            folder_name, f"agent_data_intervention_{num_agents}.json"
        )
    if os.path.exists(file_path):
        print(f"{file_path} has existed")
    else:
        population_data = generate_population_with_ids(num_agents, is_adult)
        save_population_to_file(population_data, file_path)
        wait_for_file(file_path)
        print(f"Data has been saved to {file_path} file.")


if __name__ == "__main__":
    files = [
        "long_term_0/agent_data_base.json",
        "long_term_0/agent_data_control.json",
        "long_term_0/agent_data_intervention.json",
        ]

    for file_path in files:
        population_data = generate_population_with_ids(250, True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(population_data, f, indent=4, ensure_ascii=False)
