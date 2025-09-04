import random
import json
import os
import numpy as np

NUM_SAMPLES = 65
GROUP_SIZES = [21, 22, 22]
GROUP_NAMES = ["baseline", "nudge", "boost"]


def generate_population(num_samples):
    population = []

    genders = (["Female"] * 31) + (["Male"] * 27) + (["Other"] * 3) + (["NoAnswer"] * 4)
    random.shuffle(genders)

    ages = np.random.normal(25.5, 4.21, NUM_SAMPLES)
    ages = [int(max(18, min(age, 40))) for age in ages]

    edu_levels = []
    num_masters = int(NUM_SAMPLES * 0.721)
    edu_levels += ["Master"] * num_masters
    remaining = NUM_SAMPLES - num_masters
    edu_levels += random.choices(["Bachelor", "PhD"], k=remaining)
    random.shuffle(edu_levels)

    income_levels = []
    num_low = int(NUM_SAMPLES * 0.869)
    income_levels += ["1282"] * num_low
    remaining = NUM_SAMPLES - num_low
    income_levels += random.choices(["2845", "4480", "6988", "13227"], k=remaining)
    random.shuffle(income_levels)

    marital_probs = [0.85, 0.10, 0.03, 0.02]
    marital_options = ["Single", "Married", "Divorced", "Widowed"]
    marital_statuses = random.choices(
        marital_options, weights=marital_probs, k=NUM_SAMPLES
    )

    for i in range(num_samples):
        person = {
            "id": int(i + 1),
            "Gender": genders[i],
            "Age": int(ages[i]),
            "EducationLevel": edu_levels[i],
            "MonthlyIncome": income_levels[i],
            "MaritalStatus": marital_statuses[i],
            "Occupation": "Student",
            "Pro_Environmental_Tendency": round(
                min(max(random.gauss(3, 0.5), 1), 5), 2
            ),
            "Emotional_Sensitivity": round(min(max(random.gauss(3, 0.5), 1), 5), 2),
            "Conformity_Tendency": round(min(max(random.gauss(3, 0.5), 1), 5), 2),
            "Behavioral_Inertia": round(min(max(random.gauss(3, 0.5), 1), 5), 2),
        }
        population.append(person)

    return population


def save_population(population, filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(population, f, indent=4, ensure_ascii=False)
    print(f"✅ Data saved to {filename}")


if __name__ == "__main__":
    start_idx = 0
    for size, group in zip(GROUP_SIZES, GROUP_NAMES):
        pop = generate_population(size)
        save_population(pop, f"long_term_2/agent_data_{group}.json")
