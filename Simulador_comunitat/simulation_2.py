from functions_0 import *
import os
file = 'cornella_community'

#canviar arxiu generació!!!
gen_path = r"generation.csv" #canviar arxiu generació


gen_data = []; gen_day = []
time = [n for n in range(24)]; gen = []
n = 0; m = 0; r = 1440

with open(gen_path, 'r') as csvfile:
  csv_reader = csv.reader(csvfile)
  for row in csv_reader:
    n += 1
    if n > 11:
      gen_day.append(round(float(row[1]), 2))
      m += 1
      if m == 24:
        gen_data.append([row[0][0:8], gen_day])
        m = 0; gen_day = []

# Repetir los datos de generación solar para simular N años
# El consumo es estocástico, así que cada repetición genera perfiles distintos
gen_data_1y = gen_data.copy()

#TUNE SIMULATION
formula = 'const'#'div1'
start_day = 7; week = 0    
e_s = 6; e_i = 3 #e in pool to activate AMM pricing
trig = 1 #trigger to gen/cons e kWh
rel = 300 # relation €/MWh / m or max day pvpc
ac = 1

t_csv = 5 # details in excel in min
import os, sys

# Añadimos la carpeta src al PYTHONPATH para usar la misma fuente de parámetros
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(root_dir, "src"))
import params

precision = int(params.GRAN_MIN)
bess_efficiency = 1
price_sell_eur = params.PRICE_SELL_EUR
price_buy_eur = params.PRICE_BUY_EUR
max_stash = params.E_MAX_KWH
n_years = params.N_YEARS

# Construir gen_data multi-año
gen_data = []
for yr in range(n_years):
    base_year = 2023 + yr
    for day_entry in gen_data_1y:
        # Reemplazar el año original por el año simulado
        original_date = day_entry[0]          # ej: '20230115'
        new_date = f"{base_year}{original_date[4:]}"  # ej: '20240115'
        gen_data.append([new_date, day_entry[1]])

print(f"Simulando {n_years} año(s) = {len(gen_data)} días totales.", flush=True)

time = get_time_list(1, precision)

info_total = import_info(file)
cond = info_total[1]

a = cond['basics'][0]; b = cond['basics'][1]; c = cond['basics'][2]; b_2 = cond['basics'][3]; c_2 = cond['basics'][4]
prob = cond['peaks'][0]; props = cond['peaks'][1]

dur_morn = cond['dur'][0]; morn_pk = cond['pk'][0]
dur_mid = cond['dur'][1]; mid_pk = cond['pk'][1]
dur_night = cond['dur'][2]; night_pk = cond['pk'][2]

durs = [dur_morn, dur_mid, dur_night]
pks = [morn_pk, mid_pk, night_pk]

info_buses = info_total[0]
prosumers = info_total[0][0]['prosumers']
total_buses = len(info_total[0][0]['type_1']) + len(info_total[0][0]['type_2']) + len(info_total[0][0]['type_3'])
chars = info_buses[1]
stash = 0
cons_final = []; gen_final = []
next_day_c = [[] for n in range(total_buses)]




#SIMULACIÓ, 1 any default, canviar for per menys dies




output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

total_dias = len(gen_data)
for i, day in enumerate(gen_data): #gen_data[0:32] per un mes
        if (i+1) % 10 == 0 or i == 0:
            print(f"🔥 Simulando Día {i+1} / {total_dias}...", flush=True)
        list_users = []; gen_total = []; cons_total = []; balance = []; capacity = []
        user_cons = []; user_gen = []
        weather = []; w_dur = 0; w_mult = 0
        cost = []; es = []; ms = []

        #GENERATION DATA

        date = [day[0][6:8], day[0][4:6], day[0][0:4]]

        data = get_day_prices(date, "prices.csv")
        pvpc = data[0]
        surplus = data[1]

        steps = gen_plateau(day[1])
        gen_int = interpolate_plateaus(steps)

        for bus in chars:   
            generation = []
            peak_gen = bus[1]['peak_gen']
            for t in time:
                pos = time.index(t)
                gen = peak_gen*gen_int[pos]/1000
                generation.append(round(gen,2))

            list_users.append({'bus': bus[0], 'gen': generation})  
            user_gen.append(sum(generation))

        #CONSUMPTION 

        summs = summary_day(chars, time, a, b, c, int(date[1][1]), start_day, b_2, c_2)

        for summ in summs:
            bus = summ[0]
            one_day = get_one_day_consume(precision, summ, prob, props, a, durs, pks, next_day_c[bus-1])
            consume = one_day[0]; next_day_c[bus-1] = one_day[1]
            list_users[bus-1]['cons'] = consume.tolist()
            user_cons.append(sum(consume))

        #CAPACITY 
        for t in range(int(1440/precision)):
            gen_t = round(sum([bus['gen'][t] for bus in list_users]),2)
            gen_total.append(gen_t)

            cons_t = round(sum([bus['cons'][t] for bus in list_users]),2)
            cons_total.append(cons_t)

            dif = round(gen_t - cons_t,2)
            balance.append(dif)

            # BESS efficiency implementation
            BESS_EFF = bess_efficiency
            if dif > 0:
                energy_change = (dif * BESS_EFF) * precision / 60
            else:
                energy_change = (dif / BESS_EFF) * precision / 60

            stash += energy_change

            if stash < 0:
                stash = 0

            elif stash > max_stash:
                stash = max_stash

            capacity.append(round(stash,2))

        # EXPORT CLEAN DATA FOR RL ENVIRONMENT
        # We need a clean CSV with: Time, Gen_Escuela, Dem_Escuela, Dem_Casas
        # From the configuration, School is bus=1 (assuming Prosumers has id 1)
        
        # 1. Obtenemos el consumo y generación de la escuela (bus local 1, index 0)
        school_gen = list_users[0]['gen']
        school_cons = list_users[0]['cons']
        
        # 2. Obtenemos el consumo total de todas las casas (resto de la comunidad)
        # Sumamos los consumos de todos los usuarios excepto el [0]
        houses_cons = []
        for t in range(int(1440/precision)):
            h_cons = round(sum([bus['cons'][t] for bus in list_users[1:]]), 2)
            houses_cons.append(h_cons)
            
        # 3. Guardamos los datos purificados del día
        # Límites físicos máximos — protege contra valores aberrantes del simulador estocástico
        _MAX_GEN     = params.SCHOOL_PEAK_GEN_KW          # 15 kW
        _MAX_DEM_ESC = params.SCHOOL_PEAK_GEN_KW * 2      # 30 kW (escuela)
        _MAX_DEM_CAS = params.N_HOUSES * 15.0              # 300 kW (casas, conservador)

        with open(f'{output_dir}/{date[2]}-{date[1]}-{date[0]}_datos_cornella.csv', mode="w", newline="", encoding="utf-8") as file_csv:
            writer = csv.writer(file_csv)
            writer.writerow(["Time_Min", "Gen_Escuela_kW", "Dem_Escuela_kW", "Dem_Casas_kW", "Precio_Compra", "Precio_Venta"])
            for t in range(int(1440/precision)):
                h = int((t * precision)/60)
                gen_val  = min(max(float(school_gen[t]),  0.0), _MAX_GEN)
                dem_esc  = min(max(float(school_cons[t]), 0.0), _MAX_DEM_ESC)
                dem_cas  = min(max(float(houses_cons[t]), 0.0), _MAX_DEM_CAS)
                writer.writerow([t * precision, round(gen_val, 4), round(dem_esc, 4), round(dem_cas, 4), pvpc[h], surplus[h]])

        # Avanzar día lógica
        if start_day < 7:
            start_day += 1
        else:
            start_day = 1; week += 1

print(f"✅ ¡Simulación completada! Datos crudos para el Entorno RL guardados en '{output_dir}'.")




 
