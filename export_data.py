import pandas as pd
import numpy as np
from datetime import datetime

def generate_synthetic_data():
    """Generates realistic synthetic data with exact labels requested by the user."""
    districts = ['Rajkot', 'Jamnagar', 'Junagadh', 'Amreli', 'Bhavnagar', 'Porbandar', 'Morbi', 'Dwarka']
    
    AVG_RECHARGE_RATE = 0.05 
    AVG_EXTRACTION_RATE = 0.12
    
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    data = []
    
    for district in districts:
        if district in ['Junagadh', 'Amreli']:
            ext_wells = np.random.randint(150, 300)
            rech_wells = np.random.randint(50, 100)
        else:
            ext_wells = np.random.randint(300, 500)
            rech_wells = np.random.randint(20, 60)
            
        for date in dates:
            month = date.month
            is_monsoon = 6 <= month <= 9
            
            if is_monsoon:
                daily_rain = np.random.gamma(shape=2, scale=10) if np.random.rand() > 0.3 else 0
            else:
                daily_rain = np.random.gamma(shape=1, scale=2) if np.random.rand() > 0.9 else 0
            
            gw_level = 15 + (np.sin(date.dayofyear / 365 * 2 * np.pi) * 5) + np.random.normal(0, 0.5)
            gw_level = max(2, gw_level)
            
            res_level = 40 + (30 if is_monsoon else -30) * (date.dayofyear % 365 / 365) + np.random.normal(0, 5)
            res_level = max(0, min(100, res_level))
            
            # Calculations for the CSV
            nat_rech = daily_rain * 0.1 # Real-time rain contribution
            art_rech = rech_wells * AVG_RECHARGE_RATE
            total_rech = nat_rech + art_rech
            extraction = ext_wells * AVG_EXTRACTION_RATE
            net_change = total_rech - extraction
            
            # Stress Classification Logic
            if gw_level < 12 and net_change >= 0:
                stress_label = "Safe"
            elif gw_level > 20 and net_change < -5:
                stress_label = "Critical"
            else:
                stress_label = "Warning"

            data.append({
                'Date': date,
                'District': district,
                'Rainfall_mm': round(daily_rain, 1),
                'Groundwater_Level_mbgl': round(gw_level, 2),
                'Reservoir_Level_pct': round(res_level, 1),
                'Extraction Borewells': ext_wells,
                'Recharge Borewells': rech_wells,
                'Groundwater Stress Classification': stress_label,
                'Net Groundwater Change MLD': round(net_change, 2),
                'Groundwater Recovery MLD': round(total_rech, 2),
                'Groundwater Extraction MLD': round(extraction, 2)
            })
            
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("Regenerating refined dataset with exact labels...")
    df = generate_synthetic_data()
    df.to_csv('saurashtra_water_data.csv', index=False)
    print("Done! CSV updated with 'Extraction Borewells', 'Recharge Borewells', and 'Groundwater Stress Classification'.")
