import json
import os
import time

DESCRIPTION_TEMPLATE = """You are a {Gender} aged {Age}, {MaritalStatus}, {Occupation} with {EducationLevel}. With a monthly income of ${MonthlyIncome}, you belong to the {IncomeQuintile}. You naturally exhibit {ProEnvironmentalTendencyDesc} concern for the environment (your score: {ProEnvironmentalTendencyScore}, social average: {ProEnvironmentalTendencyAvg}). You show {EmotionalSensitivityDesc} sensitivity to emotional cues (your score: {EmotionalSensitivityScore}, social average: {EmotionalSensitivityAvg}). You show {ConformityTendencyDesc} alignment with group norms (your score: {ConformityTendencyScore}, social average: {ConformityTendencyAvg}). You uphold {BehavioralInertiaDesc} consistency in habits (your score: {BehavioralInertiaScore}, social average: {BehavioralInertiaAvg})."""

DESCRIPTION_TEMPLATE2 = """You are a {Gender} aged {Age}, {MaritalStatus}, {Occupation} with {EducationLevel}. You naturally exhibit {ProEnvironmentalTendencyDesc} concern for the environment (your score: {ProEnvironmentalTendencyScore}, social average: {ProEnvironmentalTendencyAvg}). You show {EmotionalSensitivityDesc} sensitivity to emotional cues (your score: {EmotionalSensitivityScore}, social average: {EmotionalSensitivityAvg}). You show {ConformityTendencyDesc} alignment with group norms (your score: {ConformityTendencyScore}, social average: {ConformityTendencyAvg}). You uphold {BehavioralInertiaDesc} consistency in habits (your score: {BehavioralInertiaScore}, social average: {BehavioralInertiaAvg})."""


INCOME_QUINTILE_MAPPING = {
    "1282": "lowest income group (bottom 20%)",  
    "2845": "lower-middle income group (20%-40%)", 
    "4480": "middle income group (40%-60%)", 
    "6988": "upper-middle income group (60%-80%)",  
    "13227": "highest income group (top 20%)",  
}


def map_income_quintile(monthly_income):
    """Map the MonthlyIncome to its corresponding income quintile description."""
    return INCOME_QUINTILE_MAPPING.get(monthly_income, "unknown income group")


EDUCATION_LEVEL_MAPPING = {
    "No Formal Education": "no formal education",
    "Primary": "primary school education",
    "JuniorHigh": "junior high school education",
    "SeniorHigh": "senior high school education",
    "CollegePlus": "college education or higher",
    "Bachelor": "bachelor's degree",
    "Master": "master's degree",
    "PhD": "doctoral degree",
}


def map_education_level(education_level):
    """Map the EducationLevel to its descriptive phrase."""
    return EDUCATION_LEVEL_MAPPING.get(education_level, "unknown education level")


traits_info = {
    "Pro_Environmental_Tendency": {"mean": 3, "std": 0.5, "min": 1, "max": 5},
    "Emotional_Sensitivity": {"mean": 3, "std": 0.5, "min": 1, "max": 5},
    "Conformity_Tendency": {"mean": 3, "std": 0.5, "min": 1, "max": 5},
    "Behavioral_Inertia": {"mean": 3, "std": 0.5, "min": 1, "max": 5},
}


def map_trait_description(score, trait_info):
    """Map a numerical trait score to its descriptive phrase based on statistical info."""
    if not isinstance(score, (int, float)):
        return "unknown"
    mean = trait_info.get("mean", 0)
    std = trait_info.get("std", 0)
    min_value = trait_info.get("min", 0)
    max_value = trait_info.get("max", 0)
    if score < min_value or score > max_value:
        return "out of range"
    if score < mean - 2 * std:
        return "very low"
    elif mean - 2 * std <= score < mean - std:
        return "low"
    elif mean - std <= score < mean + std:
        return "moderate"
    elif mean + std <= score < mean + 2 * std:
        return "high"
    else:
        return "very high"


def generate_description(agent, is_adult):
    """Generate the descriptive text for a single agent."""
    income_quintile = map_income_quintile(agent.get("MonthlyIncome", "unknown"))

    education_level = map_education_level(agent.get("EducationLevel", "unknown"))

    pro_environmental_tendency_desc = map_trait_description(
        agent.get("Pro_Environmental_Tendency", "unknown"),
        traits_info["Pro_Environmental_Tendency"],
    )

    emotional_sensitivity_desc = map_trait_description(
        agent.get("Emotional_Sensitivity", "unknown"),
        traits_info["Emotional_Sensitivity"],
    )

    conformity_tendency_desc = map_trait_description(
        agent.get("Conformity_Tendency", "unknown"), traits_info["Conformity_Tendency"]
    )

    behavioral_inertia_desc = map_trait_description(
        agent.get("Behavioral_Inertia", "unknown"), traits_info["Behavioral_Inertia"]
    )

    occupation = agent.get("Occupation", "Unknown")
    if occupation in ["Student", "Retired", "Unemployed"]:
        occupation_phrase = occupation.lower()
    else:
        occupation_phrase = f"working as a {occupation.lower()}"

    if is_adult:
        description = DESCRIPTION_TEMPLATE.format(
            Gender=agent.get("Gender", "Unknown").lower(),
            Age=agent.get("Age", "Unknown"),
            MaritalStatus=agent.get("MaritalStatus", "Unknown").lower(),
            Occupation=occupation_phrase,
            MonthlyIncome=agent.get("MonthlyIncome", "Unknown"),
            IncomeQuintile=income_quintile,
            EducationLevel=education_level,
            ProEnvironmentalTendencyScore=agent.get(
                "Pro_Environmental_Tendency", "unknown"
            ),
            ProEnvironmentalTendencyAvg=traits_info["Pro_Environmental_Tendency"][
                "mean"
            ],
            ProEnvironmentalTendencyDesc=pro_environmental_tendency_desc,
            EmotionalSensitivityScore=agent.get("Emotional_Sensitivity", "unknown"),
            EmotionalSensitivityAvg=traits_info["Emotional_Sensitivity"]["mean"],
            EmotionalSensitivityDesc=emotional_sensitivity_desc,
            ConformityTendencyScore=agent.get("Conformity_Tendency", "unknown"),
            ConformityTendencyAvg=traits_info["Conformity_Tendency"]["mean"],
            ConformityTendencyDesc=conformity_tendency_desc,
            BehavioralInertiaScore=agent.get("Behavioral_Inertia", "unknown"),
            BehavioralInertiaAvg=traits_info["Behavioral_Inertia"]["mean"],
            BehavioralInertiaDesc=behavioral_inertia_desc,
        )
    else:
        description = DESCRIPTION_TEMPLATE2.format(
            Gender=agent.get("Gender", "Unknown").lower(),
            Age=agent.get("Age", "Unknown"),
            MaritalStatus=agent.get("MaritalStatus", "Unknown").lower(),
            Occupation=occupation_phrase,
            EducationLevel=education_level,
            ProEnvironmentalTendencyScore=agent.get(
                "Pro_Environmental_Tendency", "unknown"
            ),
            ProEnvironmentalTendencyAvg=traits_info["Pro_Environmental_Tendency"][
                "mean"
            ],
            ProEnvironmentalTendencyDesc=pro_environmental_tendency_desc,
            EmotionalSensitivityScore=agent.get("Emotional_Sensitivity", "unknown"),
            EmotionalSensitivityAvg=traits_info["Emotional_Sensitivity"]["mean"],
            EmotionalSensitivityDesc=emotional_sensitivity_desc,
            ConformityTendencyScore=agent.get("Conformity_Tendency", "unknown"),
            ConformityTendencyAvg=traits_info["Conformity_Tendency"]["mean"],
            ConformityTendencyDesc=conformity_tendency_desc,
            BehavioralInertiaScore=agent.get("Behavioral_Inertia", "unknown"),
            BehavioralInertiaAvg=traits_info["Behavioral_Inertia"]["mean"],
            BehavioralInertiaDesc=behavioral_inertia_desc,
        )
    return description


def wait_for_file(path, timeout=2):
    start = time.time()
    while not os.path.exists(path):
        if time.time() - start > timeout:
            raise TimeoutError(f"File {path} not created in time.")
        time.sleep(0.1)


def process_agent_descriptions(num_agents, is_adult, control):
    folder_name = f"profile"
    if control:
        input_file = f"agent_data_control_{num_agents}.json"
        output_file = f"character_control_{num_agents}.json"
    else:
        input_file = f"agent_data_intervention_{num_agents}.json"
        output_file = f"character_intervention_{num_agents}.json"

    input_path = os.path.join(folder_name, input_file)
    output_path = os.path.join(folder_name, output_file)

    if not os.path.exists(folder_name):
        print(f"Folder {folder_name} does not exist.")
        return
    if not os.path.exists(input_path):
        print(f"{input_file} does not exist.")
        return
    if os.path.exists(output_path):
        print(f"{output_file} has existed.")
        with open(output_path, "r", encoding="utf-8") as f:
            character_data = json.load(f)
        return list(character_data.values())
    with open(input_path, "r", encoding="utf-8") as f:
        agents = json.load(f)
    character_data = {}
    for idx, agent in enumerate(agents, start=1):
        description = generate_description(agent, is_adult)
        print(f"Processed Agent {idx}:")
        print(description)
        print("-" * 50)
        character_data[str(idx)] = description
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(character_data, f, ensure_ascii=False, indent=4)
    wait_for_file(output_path)
    print(f"All agent descriptions have been saved to '{output_file}'.")
    return list(character_data.values())


if __name__ == "__main__":
    input_file_name = "long_term_0/agent_data_intervention.json"
    output_file_name = "long_term_0/character_intervention.json"
    with open(input_file_name, "r", encoding="utf-8") as f:
        agents = json.load(f)
    character_data = {}
    for idx, agent in enumerate(agents, start=1):
        description = generate_description(agent, True)
        character_data[str(idx)] = description
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(character_data, f, ensure_ascii=False, indent=4)
