import json

def generate_cornella_config(n_houses=20, school_peak_gen=15.0, school_battery=50.0):
    """
    Generates a configuration file for the Cornellà energy community.
    Prosumer 1: School (generates power + has battery)
    Consumers 2 to N+1: Houses (only consume)
    """
    houses_list = list(range(2, n_houses + 2))
    
    # 1. Structure the types (School is type 3, Houses are split between type 1 and 2 randomly for variety)
    mid_point = len(houses_list) // 2
    type_1 = houses_list[:mid_point]
    type_2 = houses_list[mid_point:]
    type_3 = [1] # School
    
    prosumers = [1] # Only the school has solar panels
    stashers = [1]  # Only the school has a battery
    
    users = {
        "type_1": type_1,
        "type_2": type_2,
        "type_3": type_3,
        "prosumers": prosumers,
        "stashers": stashers
    }
    
    # 2. Define the individual profiles
    profiles = []
    
    # User 1: School Profile
    school_profile = [1, {
        "kW_base": 1.5,
        "max_cons_time": ["8:00", "11:30", "15:00"],
        "peak_kW": 15.0,
        "peak_gen": school_peak_gen,
        "pan_inc": 0,
        "max_stash": school_battery,
        "use": "work"
    }]
    profiles.append(school_profile)
    
    # Users 2 to N+1: Houses Profile
    import random
    for i in houses_list:
        house_profile = [i, {
            "kW_base": round(random.uniform(0.3, 0.8), 2),
            "max_cons_time": ["7:00", "14:00", "20:00"], # Approx times
            "peak_kW": round(random.uniform(3.5, 7.5), 2),
            "peak_gen": 0, # No solar
            "pan_inc": 0,
            "max_stash": 0, # No battery
            "use": "dwelling"
        }]
        profiles.append(house_profile)
        
    # 3. Define the general community constraints (copied from Cornellà)
    constraints = {
        "basics": [0.3, 20, [0.95, 0.3, 0.8], 60, [0.6, 0.3, 0.5]],
        "peaks": [[0.7, 0.3, 0.2], [0.05, 0.1, 0.015, 0.2, 0.25, 0.3]],
        "dur": [[15, 60], [20, 90], [90, 240]],
        "pk": [[1, 25], [5, 45], [30, 90]]
    }
    
    # Final structure expected by the simulator
    cornella_data = [ [users, profiles], constraints ]
    
    # Write to file
    with open('cornella_community.txt', 'w') as f:
        json.dump(cornella_data, f)
        
    print(f"Generated cornella_community.txt with 1 school and {n_houses} houses.")

if __name__ == "__main__":
    import os
    import sys
    
    # Añadimos la carpeta src al PYTHONPATH para usar la misma fuente de parámetros
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(root_dir, "src"))
    import params

    generate_cornella_config(
        n_houses=params.N_HOUSES,
        school_peak_gen=params.SCHOOL_PEAK_GEN_KW,
        school_battery=params.E_MAX_KWH
    )
