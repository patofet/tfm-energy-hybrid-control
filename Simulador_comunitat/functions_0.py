from random import randint, choice, uniform, random, randrange
import numpy as np
from numpy import zeros
import json
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from math import exp
import pandas as pd

def get_time_list(days, precision):
    time = ['0:00']
    max_mins = int((60 * 24 * days/precision) - 1)
    min_lim = 60 - precision   

    for _ in range (0, max_mins):
        a, b = map(int, time[-1].split(":"))

        if b == min_lim:
            b = 0
            if a == 23:
                a = 0
            else:
                a += 1
        else:
            b += precision

        if b < 10:
            B = '0' + str(b)
        else:
            B = str(b)

        A = str(a)

        t = A + ':' + B
        time.append(t)

    return time

def interpolate_plateaus(data):
    data = np.array(data, dtype=np.float64)
    n = len(data)

    i = 0
    while i < n - 1:
        # Find start of plateau
        start = i
        while i < n - 1 and data[i] == data[i+1]:
            i += 1
        end = i

        if end > start and start > 0 and end < n - 1:
            # interpolate between data[start - 1] and data[end + 1]
            interp = np.linspace(data[start - 1], data[end + 1], end - start + 3)[1:-1]
            data[start:end+1] = interp
        i += 1

    return data

def gen_plateau(data):
    steps = []; r = 1440; o = 0; t = 0
    for var in range(r):
            o += 1
            x = data[t]
            steps.append(x)
            if o == 60:
                o = 0; t += 1

    return steps

#CONSUMPTION

def rand_distribution(buses, cons_types, gen, stash, user_types):
    buses_1 = buses[:]
   
    total_buses = sum(cons_types)
    users = [[i] for i in range(1, total_buses + 1)]
    i = 0; j = 0; k = 0

    prosumers = []
    stashers = []

    t1 = []
    t2 = []
    t3 = []

    if gen == [-1, -1, -1]:
        prosumers = buses[:]
        while buses_1:
            n = choice(buses_1)
            buses_1.remove(n)
        
            if i < cons_types[0]: 
                users[n-1].append(user_types[0])
                t1.append(n)
                if i < stash[0]:
                    stashers.append(n)
                i += 1

            elif i >= cons_types[0] and j < cons_types[1]:
                users[n-1].append(user_types[1])
                t2.append(n)
                if j < stash[1]:
                    stashers.append(n)
                j += 1        
            
            elif (j >= cons_types[1] and i >= cons_types[0]) and k < cons_types[2]: #  and k <= cons_profiles[2]:
                users[n-1].append(user_types[2])
                t3.append(n)
                if k < stash[2]:
                    stashers.append(n)
                k += 1
    
    else:
        while buses_1:
            n = choice(buses_1)
            buses_1.remove(n)
        
            if i < cons_types[0]: 
                users[n-1].append(user_types[0])
                t1.append(n)

                if i < gen[0]:
                    prosumers.append(n)

                if i < stash[0]:
                    stashers.append(n)

                i += 1

            elif i >= cons_types[0] and j < cons_types[1]:
                users[n-1].append(user_types[1])
                t2.append(n)

                if j < gen[1]:
                    prosumers.append(n)

                if j < stash[1]:
                    stashers.append(n)

                j += 1        
            
            elif (j >= cons_types[1] and i >= cons_types[0]) and k < cons_types[2]: #  and k <= cons_profiles[2]:
                users[n-1].append(user_types[2])
                t3.append(n)

                if k < gen[2]:
                    prosumers.append(n)

                if k < stash[2]:
                    stashers.append(n)

                k += 1

        
    buses_o = {'type_1': sorted(t1), 'type_2': sorted(t2), 'type_3': sorted(t3), 'prosumers': sorted(prosumers), 'stashers': sorted(stashers)}
    output = [users, buses_o]

    return buses_o

def average_measures(buses, users, user_types, time, panel_peak,gen):
    info_buses =  []
    for bus in buses:
        info_buses.append([bus])

    for key, value in users.items():
        if key == 'type_1':
            type_1 = value

        elif key == 'type_2':
            type_2 = value

        elif key == 'type_3':
            type_3 = value
   
    for n in type_1:
        base = user_types[0]
        profile = get_profile(base, n, users, time, panel_peak,gen)        
        info_buses[n-1].append(profile) 
    
    for n in type_2:
        base = user_types[1]
        profile = get_profile(base, n, users, time, panel_peak,gen)        
        info_buses[n-1].append(profile) 

    for n in type_3:
        base = user_types[2]
        profile = get_profile(base, n, users, time, panel_peak,gen)        
        info_buses[n-1].append(profile) 
    
    return info_buses 

def get_profile(base, n, users, time, panel_peak,gen):
        const = round(uniform(base['const'][0],base['const'][1]), 2)
        peak = round(choice(base['peak'])*uniform(base['fact'][0],base['fact'][1]),2)

        if n in users['prosumers'] and gen > 0:
            gen = randint(base['gen'][0][0],base['gen'][0][1])*panel_peak
            inc = choice(base['gen'][1])
        else:
            gen = 0
            inc = 0

        if n in users['stashers']:
            stash = round(uniform(base['stash'][0],base['stash'][1]), 2)
        else:
            stash = 0
       
        pos_1 = time.index(base['times'][0])
        pos_2 = time.index(base['times'][1])
        pos_3 = time.index(base['times'][2])
        pos_4 = time.index(base['times'][3])
        pos_5 = time.index(base['times'][4])
        pos_6 = time.index(base['times'][5])

        time_1 = randint(pos_1, pos_2)
        time_2 = randint(pos_3, pos_4)
        time_3 = randint(pos_5, pos_6)

        times = [time[time_1], time[time_2], time[time_3]]

        profile = {'kW_base': const, 'max_cons_time': times, 'peak_kW': peak, 'peak_gen': gen, 'pan_inc': inc, 'max_stash': stash, 'use': base['use']}

        return profile

def gen_smart_distribution(profiles, gen_com):
    rated_p = []; shared_pv = []

    for bus in profiles:
        rated_p.append(bus[1]['peak_kW'])

    peak_p = sum(rated_p)
    print(peak_p)
    print(rated_p)
    for user in rated_p:
        share = round(user*(gen_com/peak_p)/1000,3)
        shared_pv.append(share)

    print(shared_pv)
    print(sum(shared_pv))

    for bus in profiles:
        bus[1]['peak_gen'] = shared_pv[bus[0]-1]


    return profiles


    
def summary_day(info_buses, time, a, b, c, month, day, b_2, c_2):
    general_sum = []

    for bus in info_buses:
        profile = bus[1] 
        #print(profile)    

        if profile['use'] == 'dwelling' and (month >= 6 and month <= 9 or day >= 6): #summer/weekend consume variation
            peak_morn = c_2[0]
            peak_mid = c_2[1]
            peak_night = c_2[2]
            b = b_2

        elif profile['use'] == 'work' and (day >= 6 or (month >= 6 and month <= 9)):#free time work
            peak_morn = 0
            peak_mid = 0
            peak_night = 0   

        elif profile['use'] == 'work': #same prob all day (> than dwellings)
            peak_morn = c[0]*1.05
            peak_mid = c[0]*1.05
            peak_night = c[0]*1.05

        else:
            peak_morn = c[0]
            peak_mid = c[1]
            peak_night = c[2]

        n = round(uniform(1-a,1+a),2)
        m = randint(-b,b)
        o = randint(-b,b)
        p = randint(-b,b)

        const = round(n*profile['kW_base'],2)# change to constant conss
        max_lim = profile['peak_kW']

        time_1 = time.index(profile['max_cons_time'][0]) + m
        time_2 = time.index(profile['max_cons_time'][1]) + o
        time_3 = time.index(profile['max_cons_time'][2]) + p

        if random() < peak_morn:          
            morn = True

        else:
            morn = False

        if random() < peak_mid and not(profile['use'] == 'work' and morn == False):          
            mid = True

        else:
            mid = False


        if random() < peak_night and not(profile['use'] == 'work' and morn == False):          
            night = True

        else:
            night = False

        times = [time[time_1], time[time_2], time[time_3]]
        probs = [morn, mid, night]

        summary = [bus[0], const, max_lim, times, probs, profile['use']]
        general_sum.append(summary)

    return general_sum

def get_peak(n_peak, peak_dur, peak_val, duration, prob, prop, a, precision):
    
    if n_peak == 1:
                
        if peak_dur[n_peak-1] == precision:
            n_peak = 0
            peak_val = [0, 0, 0]
            peak_dur = [0, 0, 0]
            
        else:
            peak_dur[n_peak-1] -= precision

    elif n_peak == 2:  

            if peak_dur[n_peak-2] == precision and peak_dur[n_peak-1] == precision:
                n_peak = 0
                peak_val = [0, 0, 0]
                peak_dur = [0, 0, 0]

            elif peak_dur[n_peak-2] == precision:
                peak_val = [peak_val[n_peak-1], 0, 0]
                peak_dur = [peak_dur[n_peak-1]-precision, 0, 0]
                n_peak -= 1

            elif peak_dur[n_peak-1] == precision:           
                peak_val[n_peak-1] = 0
                peak_dur[n_peak-1] = 0
                peak_dur[n_peak-2] -= precision
                n_peak -= 1
        
            else:
                peak_dur[n_peak-1] -= precision
                peak_dur[n_peak-2] -= precision

    elif n_peak == 3:
            
            if peak_dur[n_peak-1] == precision:           
                peak_val[n_peak-1] = 0
                peak_dur[n_peak-1] = 0
                
            if peak_dur[n_peak-2] == precision:           
                peak_val = [peak_val[n_peak-3], peak_val[n_peak-1], 0]
                peak_dur = [peak_dur[n_peak-3], peak_dur[n_peak-1], 0]          

            if peak_dur[n_peak-3] == precision:       
                peak_val = [peak_val[n_peak-2], peak_val[n_peak-1], 0]
                peak_dur = [peak_dur[n_peak-2], peak_dur[n_peak-1], 0]
                
            dif = 0
            peaks = []
            for dur in peak_dur:
                if dur != 0:
                    dur -= precision
                    peaks.append(dur)

                else:
                    dif +=  1 
                    peaks.append(0)
            
            n_peak -= dif
            peak_dur = peaks[:]

    if n_peak == 0 and random() < prob[0]:
        peak_dur[n_peak] = duration
        peak_val[n_peak] = choice(prop) * round(uniform(1-a,1+a),2)

        n_peak += 1

    elif n_peak == 1 and random() < prob[1]:
        peak_dur[n_peak] = duration
        peak_val[n_peak] = choice(prop) * round(uniform(1-a,1+a),2)

        n_peak += 1

    elif n_peak == 2 and random() < prob[2]:
        peak_dur[n_peak] = duration
        peak_val[n_peak] = choice(prop) * round(uniform(1-a,1+a),2)

        n_peak += 1
        
        #cons = sum(peak_val)

    details = [n_peak, peak_dur, peak_val]
    
    return details

def get_one_day_consume(precision, summ, prob, props, a, durs, pks, next_consume):

    time = get_time_list(1, precision)

    if summ[-1] == 'work':
        dur_morn = [200, 300]; morn_pk = pks[0] #duració fase i pics lloc de feina
        dur_mid = [60, 90]; mid_pk = pks[1]
        dur_night = [150, 250]; night_pk = pks[2]
        props = [pr * 1.25 for pr in props]

        
        cons_rand = (0.05, 0.075, 0.1, 0.125)
        duration = randrange(dur_morn[0], dur_morn[1], precision)
        duration_1 = randrange(dur_mid[0], dur_mid[1], precision)
        duration_2 = randrange(dur_night[0], dur_night[1], precision)

    else:
        dur_morn = durs[0]; morn_pk = pks[0]
        dur_mid = durs[1]; mid_pk = pks[1]
        dur_night = durs[2]; night_pk = pks[2]

        
        cons_rand = (0.025, 0.05, 0.075, 0.1)
        duration = randrange(dur_morn[0], dur_morn[1], precision)
        duration_1 = randrange(dur_mid[0], dur_mid[1], precision)
        duration_2 = randrange(dur_night[0], dur_night[1], precision)


    pmax = summ[2]
    cons = zeros(len(time)); n_peak = 0
    peak_dur = [0, 0, 0]; peak_val = [0, 0, 0]
    lim = -2; lim_1 = -2; lim_2 = -2
    m = [1, 25]
    still_active = False; act = 0
    consume = []


    for t in time:
        pos = time.index(t)
        if (t == summ[3][0] or pos <= lim) and summ[4][0] == True:
                if t == summ[3][0]:
                    for n in range(randint(m[0], m[1])):
                        idx = pos - 1 - n
                        if idx >= 0 and idx < len(consume):
                            consume[idx] = round(choice(cons_rand) * uniform(1-a,1+a) ,2) 

                lim = duration/precision + time.index(summ[3][0])
                dur = randrange(morn_pk[0], morn_pk[1] + 1, precision)
                prop = props[:]
                details = get_peak(n_peak, peak_dur, peak_val, dur, prob, prop, a, precision)
                n_peak = details[0]; peak_dur = details[1]; peak_val = details[2]
                cons = round(pmax * sum(peak_val), 2)
                

        elif (t == summ[3][1] or pos <= lim_1) and summ[4][1] == True:
                if t == summ[3][1]:
                    for n in range(randint(m[0], m[1])):
                        idx = pos - 1 - n
                        if idx >= 0 and idx < len(consume):
                            consume[idx] = round(choice(cons_rand) * uniform(1-a,1+a), 2)
                    
                lim_1 = duration_1/precision + time.index(summ[3][1])
                dur = randrange(mid_pk[0], mid_pk[1] + 1, precision)
                prop = props[0:4]
                details = get_peak(n_peak, peak_dur, peak_val, dur, prob, prop, a, precision)
                n_peak = details[0]; peak_dur = details[1]; peak_val = details[2]
                cons = round(pmax * sum(peak_val), 2)

        elif (t == summ[3][2] or pos <= lim_2) and summ[4][2] == True:
                if t == summ[3][2]:
                    for n in range(randint(m[0], m[1])):
                        idx = pos - 1 - n
                        if idx >= 0 and idx < len(consume):
                            consume[idx] = round(choice(cons_rand) * uniform(1-a,1+a), 2)
                lim_2 = duration_2/precision + time.index(summ[3][2])
                dur = randrange(night_pk[0], night_pk[1] + 1, precision)
                prop = props[:]
                details = get_peak(n_peak, peak_dur, peak_val, dur, prob, prop, a, precision)
                n_peak = details[0]; peak_dur = details[1]; peak_val = details[2]
                cons = round(pmax * sum(peak_val), 2)
            
        elif (pos == lim_2 + 1) or (pos == lim_1 + 1) or (pos == lim + 1) or still_active == True:
                still_active = True
                cons = round(choice(cons_rand) * uniform(1-a,1+a), 2)
                act += 1
                if act >= 10:
                    still_active = False
                    act = 0
                
        else:
                n_peak = 0; peak_dur = [0, 0, 0]; peak_val = [0, 0, 0]
                cons = round(pmax * sum(peak_val), 2)

        info = [t, n_peak, peak_dur, peak_val]       
        consume.append(round(cons,2))

    
    if next_consume != []:
            n = 0
            for cons_min in next_consume:
                consume[n] = cons_min
                n += 1

    consume = interpolate_plateaus(consume)

    for t in time:    
        pos = time.index(t)
        consume[pos] = consume[pos] + round(summ[1] * uniform(1-(a-0.2),1+(a-0.2)), 2)
        if consume[pos] > pmax:
            consume[pos] = pmax

    next_consume = []
    for t in time:
            if lim_2 > time.index(time[-1]): #millora a fer: passar aquest consum al dia següent.
                pos_n = lim_2 - time.index(time[-1])
                pos_2 = time.index(t)        
                if pos_2 < pos_n:
                    details = get_peak(n_peak, peak_dur, peak_val, dur, prob, prop, a, precision)
                    n_peak = details[0]; peak_dur = details[1]; peak_val = details[2]
                    next_consume.append(round(pmax * sum(peak_val), 2))
    
    output = [consume, next_consume]

    return output

#AMM

def m_calc(data):
    if (len(data.pool[0]) >= data.e_th_s and data.check == False) or (len(data.pool[0]) > data.e_th_i and data.check == True):
        
        if data.form == 'exp' :
            data.pm = 1 - exp(-data.a*(data.pool[1]/(len(data.pool[0]))))
        
        elif data.form == 'div1' :#formula used
            data.pm = 1/((1+data.a*(len(data.pool[0])/data.pool[1])))

        elif data.form == 'div2':
            data.pm = 1 / (1 + exp(-data.a*((data.m/data.e)-1)))

        elif data.form == 'div3':
            data.pm = 1/(1 + ((data.e/data.m)**data.a))

        elif data.form == 'const':
            data.pm = 0.5
            
    else:
        data.pm = 1
    

    if data.e_th_i != 0:
        if len(data.pool[0]) >= data.e_th_s and data.check == False:
            data.check = True
        elif data.check == True and len(data.pool[0]) <= data.e_th_i:
            data.check = False
    else:
        data.check = False
    
    return data.pm

def get_price(data, t):
    pvpc_n = data.buy[data.h]; surplus_n = data.sell[data.h]

    if data.summary['cap'][t] == data.max_cap and data.summary['gen'][t] > data.summary['cons'][t]:
        real_price = surplus_n
    
    else:
        real_price = round(surplus_n + (pvpc_n - surplus_n)*data.pm, 2) #data.pm considers if there's enough gen (e in pool)

    output = [real_price, data.h]
    return output

def sum_trade(trade, data):
    if trade['bus'] != 'net' and trade['profit'] != 0: 
        bus = str(trade['bus'])
        data.profit[bus] += trade['profit']/1000
        round(float(data.profit[f'{bus}']),4)
        if trade['token in'] == 'e':
            data.real[bus] += trade['real']/1000
            round(data.real[f'{bus}'],4)
        
def com_gen(t, i, data):
    taken = round(get_price(data,t)[0]/data.euros,4)
    given = [1, 'e', i, t]#tokens traded
        
    data.pool[0].append(given)#modify pool
    data.pool[1] -= taken

    trade = {'time': t, 'bus': i, 'token in': 'e', 'info in': given, 'token out': 'm', 'info out': taken, 'price': data.last_paid[0],
              'real': data.sell[data.last_paid[1]], 'profit': round(abs(data.last_paid[0]-data.sell[data.last_paid[1]]),2)}

    return trade

def com_cons(t, j, data):
    taken = data.pool[0].pop()#takes eldest e token
    given = round(get_price(data,t)[0]/data.euros,4)

    data.pool[1] += given

    trade = {'time': t, 'bus': j, 'token in': 'm', 'info in': given, 'token out': 'e', 'info out': taken, 'price': data.last_paid[0],
             'real': data.buy[data.last_paid[1]], 'profit': round(abs(data.last_paid[0]-data.buy[data.last_paid[1]]),2)}

    return trade

def trade_net2c(t, data):
    given = [1, 'e', 'net', t]
    taken = round(get_price(data,t)[0]/data.euros,4)

    data.pool[0].append(given)
    data.pool[1] -= taken

    trade = {'time': t, 'bus': 'net', 'token in': 'e', 'info in': given, 'token out': 'm', 'info out': taken, 'price': data.last_paid[0],
             'real': data.buy[data.last_paid[1]], 'profit': 0}
    
    return trade

def trade_g2net(t, data):
    given = round(get_price(data,t)[0]/data.euros,4)
    taken = data.pool[0].pop(-1)

    data.pool[1] += given

    trade = {'time': t, 'bus': 'net', 'token in': 'm', 'info in': given, 'token out': 'e', 'info out': taken, 'price': data.last_paid[0],
             'real': data.sell[data.last_paid[1]], 'profit': 0}
    
    return trade

def k_flag(acc_gen, acc_con, k_gen, k_con, precision, list_users, trig, t, prosumers, next_day):#flag when reach 1kWh of gen or cons per bus

    for bus in list_users:
        #bus =  list_users[0]
        n = bus['bus']-1

        if t == 0:
             acc_con[n] = next_day[n][0]
             acc_gen[n] = next_day[n][1]

        acc_con[n] += (bus['cons'][t]/60)*precision
            
        if acc_con[n] >= trig:
                acc_con[n] -= trig
                k_con[n] = True 

        if bus['bus'] in prosumers:
                acc_gen[n] += (bus['gen'][t]/60)*precision

                if acc_gen[n] >= trig:
                    acc_gen[n] -= trig
                    k_gen[n] = True 

        if t == (1440 - precision):
                next_day[n][0] = acc_con[n]
                next_day[n][1] = acc_gen[n] 

        response = [acc_gen, acc_con, k_gen, k_con, next_day]

    return response

class DataHolder:
    def __init__(self, e_0, m_0, e_s, e_i, a, pvpc, surplus, dict_total, max_stash, profit, real, formula):
        self.e = e_0
        self.m = m_0
        self.e_th_s = e_s
        self.e_th_i = e_i
        self.a = a
        self.check = False
        self.pool = [[[1, 'e', 'net', 0]], self.m]#list of e tokens, amount of ms
        self.trans = []
        self.form = formula
        self.pm = 1
        self.buy = pvpc
        self.sell = surplus
        self.h = 0; self.day_mkt = []
        self.summary = dict_total
        self.last_paid = surplus[0]
        self.max_cap = max_stash
        self.profit = profit
        self.euros = max(pvpc)
        self.real = real

def get_day_prices(date, prices_file="prices.csv"):
    day = date[0]; month = date[1]; year = date[2]
    date_2 = f'{year}-{month}-{day}'

    pvpc = []
    surplus = []

    try:
        with open(prices_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader, None) # Saltar cabeceras
            
            for row in reader:
                # Modificado para tolerar espacios u otras variaciones
                if len(row) >= 4 and date_2 in row[0]:
                    pvpc.append(round(float(row[2]), 2))
                    
                    value = float(row[3])
                    if value >= 1:
                        value = value / 1000
                    elif value < 0:
                        value = 0
                    surplus.append(value)
                    
    except FileNotFoundError:
         print(f"Error: No se ha encontrado el archivo de precios {prices_file}.")
         
    # Fallback con perfil horario sintético estilo PVPC español (cuando la fecha no está en el CSV).
    # Tiene valle nocturno y pico vespertino para que el RL pueda aprender arbitraje temporal.
    if not pvpc or not surplus:
        pvpc = [
            0.09, 0.08, 0.08, 0.08, 0.09, 0.10,  # 00-05h  valle nocturno
            0.12, 0.15, 0.19, 0.22, 0.23, 0.20,  # 06-11h  rampa matutina
            0.16, 0.14, 0.13, 0.13, 0.15, 0.19,  # 12-17h  mediodía / tarde
            0.25, 0.29, 0.30, 0.27, 0.22, 0.14,  # 18-23h  pico vespertino
        ]
        surplus = [round(p * 0.38, 3) for p in pvpc]

    return [pvpc, surplus]

#DATA EXPORTING
    
def import_info(file):
    path = Path(f'{file}.txt')
    contents = path.read_text()
    info_buses = json.loads(contents)
    
    return info_buses

def export_info(info, file):
    content = json.dumps(info)
    path = Path(f'{file}.txt')
    path.write_text(content)

def export_csv(precision, d, data, name, t, prosumers, output_dir="."):

    with open(f'{output_dir}/{d}_gen_{name}.csv', mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id"] + list(n for n in range(t,1441,precision*t))) 
        #for row in data:
        for row in data:
            if row['bus'] in prosumers:
                i = 0; j = 0
                acc_gen = []
                t_gen = 0
                for gen in row['gen']:
                    t_gen += gen/60
                    j += 1
                    i += 1
                    if i == t or j == len(row['gen']):
                        acc_gen.append(t_gen)
                        t_gen = 0 
                        i = 0
                writer.writerow([round(n,4) for n in [row["bus"]] + list(acc_gen)])

    with open(f'{output_dir}/{d}_cons_{name}.csv', mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["id"] + list(n for n in range(t,1441,precision*t))) 
            for row in data:
                i = 0; j = 1
                acc_con = []
                t_con = 0
                for con in row['cons']:
                    t_con += con/60
                    i += 1
                    j += 1
                    if i == t or j == len(row['cons']):
                        acc_con.append(t_con)
                        t_con = 0 
                        i = 0
                writer.writerow([round(n,4) for n in [row["bus"]] + list(acc_con)])

#IMPROVE PLOTTING

def export_triplot_week(time, cons_total, gen_total, capacity, file, day):
    fig, ax1 = plt.subplots(figsize=(10, 5)) 
    
   
    ax1.set_xlabel('Time (Days)') 
    ax1.set_ylabel('Power (kW)') 
    lns1 = ax1.plot(cons_total, label="Aggregate Consumption (Power)", color = 'r') 
    lns2 = ax1.plot(gen_total, label="Aggregate Generation (Power)", color = 'b')
    #plt.xticks(ticks=np.arange(0, 1441, 60), labels=np.arange(0, 25, 1))
    plt.xticks(ticks=np.arange(0, 10801, 1440), labels=np.arange(0, 8, 1))

    ax2 = ax1.twinx() 
    ax2.set_ylabel('Energy (kWh)') 
    lns3 = ax2.plot(capacity, label="Aggregate Capacity (Energy)", color= 'g')
    
    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    plt.savefig('test.png')
    
    plt.close()

def export_triplot_pool(e, m, cost, file):
    fig, ax1 = plt.subplots(figsize=(10, 5)) 
    
    ax1.set_xlabel('Time (h)') 
    ax1.set_ylabel('Quantity (tokens)') 
    lns1 = ax1.plot(e, label="e currency", color = 'r') 
    lns2 = ax1.plot(m, label="m currency", color = 'b')
    plt.xticks(ticks=np.arange(0, 1441, 60), labels=np.arange(0, 25, 1))

    ax2 = ax1.twinx() 
    ax2.set_ylabel('e price (m)') 
    lns3 = ax2.plot(cost, label="Cost of 1 e", color= 'g')
    
    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    plt.savefig(f'{file}.png')
    plt.close()
    
    # Show plot
    #plt.show()

def export_triplot_price(pvpc, surplus, mkt, file):
  r = 1440

  plt.plot(range(r), mkt, label= 'AMM', color = 'g')
  plt.plot(range(r), pvpc, label= 'PVPC', color = 'r')
  plt.plot(range(r), surplus, label= 'Surplus', color = 'b')
  
  plt.xticks(ticks=np.arange(0, 1441, 60), labels=np.arange(0, 25, 1))
  plt.xlabel('Time of the day (h)')
  plt.ylabel('Price (€/MWh)')
  plt.grid()
  plt.legend()
  plt.savefig(f'{file}.png')
  plt.close()
  #plt.show()
   
def export_triplot_day(time, cons_total, gen_total, capacity, file, day, user_cons):
    fig, ax1 = plt.subplots(figsize=(10, 5)) 
    
   
    ax1.set_xlabel('Time (h)') 
    ax1.set_ylabel('Power (kW)') 
    lns1 = ax1.plot(time, cons_total, label="Aggregate Consumption (Power)", color = 'r') 
    lns2 = ax1.plot(time, gen_total, label="Aggregate Generation (Power)", color = 'b')
    lns3 = ax1.plot(time, user_cons, label="User Consumption (Power)", color = 'y')
    plt.xticks(ticks=np.arange(0, 1441, 60), labels=np.arange(0, 25, 1))

    ax2 = ax1.twinx() 
    ax2.set_ylabel('Energy (kWh)') 
    lns4 = ax2.plot(time, capacity, label="Aggregate Capacity (Energy)", color= 'g')
    
    lns = lns1+lns2+lns3+lns4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    plt.savefig(f'{day}_{file}.png')
    plt.close()
    # Show plot
    #plt.show()