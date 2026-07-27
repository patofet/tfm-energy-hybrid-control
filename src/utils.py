import numpy as np
from params import (
    DT_H, E_MAX_KWH, P_MAX_KW, SOC_MIN, SOC_MAX, 
    SOC_DEG_MIN, SOC_DEG_MAX, DEG_COST_EUR_KWH
)


def filtro_seguridad(p_desired_kw: float, soc_actual: float) -> tuple[float, float]:
    # Máxima potencia de carga antes de tocar SOC_MAX
    p_max_charge_kw = (SOC_MAX - soc_actual) * E_MAX_KWH / DT_H
    # Máxima potencia de descarga antes de tocar SOC_MIN (valor negativo)
    p_max_discharge_kw = -(soc_actual - SOC_MIN) * E_MAX_KWH / DT_H
    
    # Límite superior: mínimo entre capacidad física y potencia del inversor
    upper = min(p_max_charge_kw, P_MAX_KW)
    # Límite inferior: máximo entre capacidad física y potencia del inversor (negativa)
    lower = max(p_max_discharge_kw, -P_MAX_KW)
    
    p_safe_kw = np.clip(p_desired_kw, lower, upper)
    
    # Cuánto hemos recortado (normalizado por P_MAX_KW para que sea comparable)
    clipping_amount = abs(p_desired_kw - p_safe_kw) / P_MAX_KW
    
    return float(p_safe_kw), float(clipping_amount)


def actualizar_dinamica_bateria(p_safe_kw: float, soc_actual: float) -> tuple[float, float, float]:
    energia_real_kwh = p_safe_kw * DT_H
    delta_soc = energia_real_kwh / E_MAX_KWH
    nuevo_soc = np.clip(soc_actual + delta_soc, SOC_MIN, SOC_MAX)
    return float(nuevo_soc), float(p_safe_kw), float(energia_real_kwh)


def calcular_beneficio_sin_bateria(dem_esc_kw: float, gen_kw: float, dem_cas_kw: float, precio_c: float, precio_v: float) -> float:
    balance_esc = dem_esc_kw - gen_kw
    balance_cas = dem_cas_kw
    if balance_esc < 0:
        compartido = min(abs(balance_esc), balance_cas)
        balance_esc += compartido
        balance_cas -= compartido
    kwh_esc = balance_esc * DT_H
    kwh_cas = balance_cas * DT_H
    if kwh_esc > 0:
        return -kwh_esc * precio_c - kwh_cas * precio_c
    else:
        return abs(kwh_esc) * precio_v - kwh_cas * precio_c


def calcular_economia(dem_esc_kw: float, gen_kw: float, p_bat_kw: float, dem_cas_kw: float, precio_c: float, precio_v: float, soc_actual: float, energia_real_kwh: float) -> tuple[float, float]:  # noqa: ARG001 soc_actual reserved for future use
    # Balance = Demanda - Generación + Lo que consume/suelta la batería
    balance_esc_kw = dem_esc_kw - gen_kw + p_bat_kw
    balance_cas_kw = dem_cas_kw

    # Si a la escuela le sobra energía (balance negativo), cubre primero la demanda de las casas.
    # Los pagos internos (escuela→casas) se cancelan a nivel de comunidad;
    # solo importan los flujos reales con la red exterior.
    if balance_esc_kw < 0:
        compartido_kw = min(abs(balance_esc_kw), balance_cas_kw)
        balance_esc_kw += compartido_kw
        balance_cas_kw -= compartido_kw

    escuela_red_kwh = balance_esc_kw * DT_H
    casas_red_kwh   = balance_cas_kw * DT_H

    # Flujos reales con la red exterior
    beneficio = 0.0

    # 1. Escuela: compra o vende a la red lo que no cubre internamente
    if escuela_red_kwh > 0:
        beneficio -= escuela_red_kwh * precio_c   # Compra
    else:
        beneficio += abs(escuela_red_kwh) * precio_v  # Vende

    # 2. Casas: compran a la red lo que no recibieron de la escuela
    beneficio -= casas_red_kwh * precio_c

    # 3. Desgaste interno de la batería
    coste_deg = DEG_COST_EUR_KWH * abs(energia_real_kwh)

    return beneficio, coste_deg
