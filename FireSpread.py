import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage
import os
from scipy.ndimage import convolve, gaussian_filter, binary_closing, binary_dilation
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
map_path = os.path.join(BASE_DIR, "VegMap_NoBG_cp.png")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
GLOBAL_MIN = 0.0
GLOBAL_MAX = 1.0

def generate_risk_map(image_path, temp, humidity):
    
    dryness = temp * (1.0 - humidity)

    legend = {
        "Salt_Pan": {"rgb": np.array([225, 225, 225]), "fuel": 0.0}, 
        "Grass":    {"rgb": np.array([230, 152, 0]),   "fuel": 1.0}, 
        "Shrub":    {"rgb": np.array([164, 196, 0]),   "fuel": 0.7}, 
        "Tree":     {"rgb": np.array([0, 114, 75]),    "fuel": 0.5}
    }

    img = Image.open(map_path).convert('RGB')
    img_array = np.array(img)

    # Rasterization
    distances = []
    fuel_values = []
    for name, data in legend.items():
        dist = np.sqrt(np.sum((img_array - data["rgb"]) ** 2, axis=2))
        distances.append(dist)
        fuel_values.append(data["fuel"])

    distances = np.stack(distances, axis=-1)
    min_distances = np.min(distances, axis=-1)
    closest_color_indices = np.argmin(distances, axis=-1)
    fuel_values = np.array(fuel_values)
    fuel_matrix = fuel_values[closest_color_indices]

    threshold = 130 
    mask = min_distances > threshold
    
    ignite_matrix = dryness * fuel_matrix.astype(float)
    ignite_matrix[mask] = np.nan
    
    temp_fill = np.nan_to_num(ignite_matrix, nan=0.0)
    ignite_smooth = gaussian_filter(temp_fill, sigma=1.5)
    # Spread Matrix
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]) / 9.0
    spread_matrix = convolve(ignite_smooth, kernel, mode='constant', cval=0.0)

    # Risk Score
    w_ig = 0.5
    w_sp = 0.5
    risk_matrix = w_ig * spread_matrix + w_sp * ignite_matrix
    
    '''
    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Original Image
    
    axes[0].imshow(img_array)
    axes[0].set_title("1. Original Vegetation Map", fontsize=12)
    axes[0].axis('off')
    

    # Raster
    im2 = axes[1].imshow(fuel_matrix, cmap='YlGn')
    axes[1].set_title("2. Fuel Load Raster", fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], label='Fuel Score', fraction=0.046, pad=0.04)
    
    '''

    return risk_matrix

def scale_to_1km_grid(high_res_matrix, real_width_km, real_height_km):

    mask = np.isnan(high_res_matrix)
    filled_matrix = np.nan_to_num(high_res_matrix, nan=0.0)
    

    new_shape = (int(real_height_km), int(real_width_km))
    
    zoom_factors = (new_shape[0] / high_res_matrix.shape[0], 
                    new_shape[1] / high_res_matrix.shape[1])
    

    grid_1km = scipy.ndimage.zoom(filled_matrix, zoom_factors, order=1)
    
    new_mask = scipy.ndimage.zoom(mask.astype(float), zoom_factors, order=0)
    grid_1km[new_mask > 0.5] = np.nan
    
    return grid_1km


ETOSHA_WIDTH_KM = 300.0  
ETOSHA_HEIGHT_KM = 95.0

current_temp = .85     # 0-1, Farenheit/100
current_humidity = .1 # % Humidity

final_risk_matrix = generate_risk_map(map_path, current_temp, current_humidity)

km_risk_grid = scale_to_1km_grid(final_risk_matrix, ETOSHA_WIDTH_KM, ETOSHA_HEIGHT_KM)

# Fire Risk Map
cmap = plt.cm.get_cmap("RdYlGn_r").copy()
cmap.set_bad(color='#222222')
'''
plt.figure(figsize=(20,7), dpi = 150)
sns.heatmap(km_risk_grid, 
                # ax=axes[2], 
                cmap=cmap, 
                cbar_kws={'label': 'Risk Intensity'},
                vmin=GLOBAL_MIN,
                vmax=GLOBAL_MAX,
                xticklabels=50, 
                yticklabels=20)
    # axes[2].set_title(f"Wildfire Risk Heatmap\n(Temp: {temp}, Hum: {humidity})", fontsize=12)
plt.title(f"Wildfire Risk Heatmap\n(Temp: {current_temp}, Hum: {current_humidity})", fontsize=12)
plt.xlabel("Distance East (km)")
plt.ylabel("Distance South (km)")
plt.plot()
plt.show()
'''

# Dynamic Response Time
def calculate_time_limits(risk_grid, road_mask):
    # Max time allowed for lowest risk is 120 mins
    # Max penalty for highest risk (1.0) is 75 mins
    base_time = 120
    max_penalty = 75 
    min_allowable_time = 30
    
    time_thresholds = base_time - (max_penalty * risk_grid)
    
    # Floor
    time_thresholds = np.clip(time_thresholds, min_allowable_time, base_time)
    
    # Mask roads
    time_thresholds[road_mask] = np.nan
    time_thresholds[np.isnan(risk_grid)] = np.nan
    
    return time_thresholds

road_grid_1km = np.isnan(km_risk_grid)

time_limit_grid = calculate_time_limits(km_risk_grid, road_grid_1km)
'''
plt.figure(figsize=(20, 7), dpi=100)
cmap_time = plt.cm.get_cmap("jet_r").copy() # Red = Urgent (low time), Blue = Safe (high time)
cmap_time.set_bad(color='#222222')

sns.heatmap(time_limit_grid, cmap=cmap_time, 
            cbar_kws={'label': 'Max Allowed Response Time (mins)'},
            xticklabels=50, yticklabels=20)
plt.title("Operational Time Limits ($T_{max}$) per $1km^2$ Cell")
plt.xlabel("Distance East (km)")
plt.ylabel("Distance South (km)")
plt.show()
'''

# Burn

def apply_burn_strategy(risk_grid, road_mask, strategy='none', param=20, thickness=1, effectiveness=0.85):
    new_risk = np.copy(risk_grid)
    burn_mask = np.zeros_like(risk_grid, dtype=bool)
    rows, cols = risk_grid.shape
    
    valid_vegetation = ~np.isnan(risk_grid)
    
    if strategy == 'none':
        return new_risk, 0.0, burn_mask
        
    elif strategy == 'grid_lines':
        y, x = np.indices((rows, cols))
        
        h_lines = (y % param) < thickness
        v_lines = (x % param) < thickness
        
        burn_mask = (h_lines | v_lines) & valid_vegetation

    elif strategy == 'high_risk_target':
        valid_values = risk_grid[valid_vegetation]
        if len(valid_values) > 0:
            threshold = np.nanpercentile(valid_values, param)
            y, x = np.indices(risk_grid.shape)
            checkerboard = (y + x) % 2 == 0
            burn_mask = (risk_grid >= threshold) & checkerboard & valid_vegetation

    new_risk[burn_mask] = risk_grid[burn_mask] * (1.0 - effectiveness)
    
    area_burned = np.sum(burn_mask)
    return new_risk, area_burned, burn_mask

#Greedy Alg Post Placements
def calculate_travel_time(y_start, x_start, grid_shape, road_mask):

    y, x = np.indices(grid_shape)
    distance_km = np.sqrt((y - y_start)**2 + (x - x_start)**2)
    

    speed_offroad = 30.0 / 60.0  # km/min
    speed_road = 60.0 / 60.0     # km/min
    

    speed_grid = np.where(road_mask, speed_road, speed_offroad)
    return distance_km / speed_grid

#Add Max Radius to check less tiles
def calculate_travel_time_fast(y_start, x_start, grid_shape, road_mask, max_radius=120):
    rows, cols = grid_shape

    r_min, r_max = max(0, y_start - max_radius), min(rows, y_start + max_radius)
    c_min, c_max = max(0, x_start - max_radius), min(cols, x_start + max_radius)
    
    y, x = np.ogrid[r_min:r_max, c_min:c_max]
    dist_sq = (y - y_start)**2 + (x - x_start)**2
    distance_km = np.sqrt(dist_sq)
    
    local_road = road_mask[r_min:r_max, c_min:c_max]
    speed_grid = np.where(local_road, 1.0, 0.5) 
    
    times = np.full(grid_shape, np.inf)
    times[r_min:r_max, c_min:c_max] = distance_km / speed_grid
    return times

def optimize_station_placement(risk_grid, baseline_risk_grid, time_limits, weight_grid, num_stations):
    rows, cols = risk_grid.shape
    road_mask = np.isnan(risk_grid)
    
    clean_baseline = np.nan_to_num(baseline_risk_grid, nan=0.0)
    
    uncovered_value = np.copy(clean_baseline)
    
    total_potential_risk = np.sum(clean_baseline)
    
    if total_potential_risk <= 0:
        return [], 0.0, 0.0

    valid_placement_mask = (~np.isnan(risk_grid)) & (weight_grid > 0.05)
    valid_coords = np.argwhere(valid_placement_mask)
    
    final_locations = []
    
    for station_idx in range(num_stations):
        best_score = -1
        best_loc = None
        best_coverage_mask = None
        
        for r, c in valid_coords:
            travel_times = calculate_travel_time_fast(r, c, (rows, cols), road_mask)
            
            is_covered = (travel_times <= time_limits) & (uncovered_value > 0)
            
            score = np.sum(uncovered_value[is_covered])
            
            if score > best_score:
                best_score = score
                best_loc = (r, c)
                best_coverage_mask = is_covered
                    
        if best_loc is None or best_score <= 0:
            break

        final_locations.append(best_loc)
        uncovered_value[best_coverage_mask] = 0.0 
        
    remaining_risk = np.sum(uncovered_value)
    
    covered_risk = total_potential_risk - remaining_risk
    
    coverage_percentage = (covered_risk / total_potential_risk) * 100
    coverage_percentage = max(0.0, min(100.0, coverage_percentage))
    
    return final_locations, remaining_risk, coverage_percentage

'''
TOTAL_STATIONS = 10

station_locations = optimize_station_placement(km_risk_grid, time_limit_grid, km_risk_grid, num_stations=TOTAL_STATIONS)


plt.figure(figsize=(30, 11), dpi=300) 


sns.heatmap(km_risk_grid, 
                cmap=cmap, 
                cbar_kws={'label': 'Wildfire Risk Intensity'},
                vmin=GLOBAL_MIN,
                vmax=GLOBAL_MAX,
                xticklabels=50, 
                yticklabels=20)

# Plotting the Stations
for idx, (y, x) in enumerate(station_locations):
    plt.scatter(x, y, color='blue', marker='*', s=700, 
                edgecolor='white', linewidth=1.5, zorder=10)
    
    plt.text(x + 2, y - 2, f"Post {idx+1}", color='white', fontsize=16, fontweight='bold',
             bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=2))


plt.title(f"Etosha National Park: Optimal Wildfire Response Strategy\n"
          f"Temp: {current_temp}, Humidity: {current_humidity} | {TOTAL_STATIONS} Proposed Posts")

plt.xlabel("Distance East (km)")
plt.ylabel("Distance South (km)")

plt.tight_layout()
plt.show()
    


# print(km_risk_grid.shape)
# np.savetxt("wildfire_risk_results.csv", final_risk_matrix, delimiter=",")
'''  
# '''
#strategy analysis
road_grid_1km = np.isnan(km_risk_grid)
absolute_baseline_risk = np.nansum(km_risk_grid)

strategies = [
    {"name": "A: Baseline (No Burns)",  "burn_strat": "none", "param": 0},
    {"name": "B: 30km Firebreaks",  "burn_strat": "grid_lines", "param": 30},
    {"name": "C: 15km Road Firebreaks",  "burn_strat": "grid_lines", "param": 15},
    {"name": "D: Top 5% Mosaic Target", "burn_strat": "high_risk_target", "param": 95},
    {"name": "E: Top 10% Mosaic Target","burn_strat": "high_risk_target", "param": 90}
]

all_scenario_results = []
optimal_summary = []
marginal_risk_history = {}
optimal_data_per_strat = {}

GAIN_THRESHOLD = 2.5

for strat in strategies:
    print(f"\nProcessing {strat['name']}")
    
    # Burn
    mod_risk, burn_area, b_mask = apply_burn_strategy(
        km_risk_grid, road_grid_1km, strategy=strat['burn_strat'], param=strat['param']
    )

    mod_limits = calculate_time_limits(mod_risk, road_grid_1km)
    initial_burn_coverage = ((absolute_baseline_risk - np.nansum(mod_risk)) / absolute_baseline_risk) * 100
    
    current_cumulative_coverage = initial_burn_coverage
    strat_history = [np.nansum(mod_risk)] 

    optimal_data_per_strat[strat['name']] = {
        "map": mod_risk, "stations": [], "mask": b_mask, "n": 0, "coverage": initial_burn_coverage
    }

    veg_weight_grid = scale_to_1km_grid(km_risk_grid, ETOSHA_WIDTH_KM, ETOSHA_HEIGHT_KM)

    for n in range(1, 11):
        stations, remaining_risk, coverage = optimize_station_placement(
            risk_grid=mod_risk, 
            baseline_risk_grid=km_risk_grid, 
            time_limits=mod_limits, 
            weight_grid=km_risk_grid,  
            num_stations=n
        )
        
        total_coverage_now = ((absolute_baseline_risk - remaining_risk) / absolute_baseline_risk) * 100
        marginal_gain = max(0.0, total_coverage_now - current_cumulative_coverage)

        all_scenario_results.append({
            "Strategy": strat['name'], 
            "Stations": n,
            "Total_Protection_%": round(total_coverage_now, 4),
            "Marginal_Gain_%": round(marginal_gain, 4),
            "Burn_Area_km2": int(burn_area)
        })

        if marginal_gain >= GAIN_THRESHOLD:
            optimal_data_per_strat[strat['name']] = {
                "map": mod_risk, 
                "stations": stations, 
                "mask": b_mask,
                "n": n, 
                "coverage": total_coverage_now
            }
            
        current_cumulative_coverage = total_coverage_now
        strat_history.append(remaining_risk)
        
        print(f"Station {n}: Total {total_coverage_now:.2f}% | Gain {marginal_gain:.2f}%")

    marginal_risk_history[strat['name']] = strat_history

    res = optimal_data_per_strat[strat['name']]
    optimal_summary.append({
        "Strategy": strat['name'], 
        "Opt_Stations": res['n'], 
        "Total_Protection_%": round(res['coverage'], 2),
        "Burn_Area_km2": int(burn_area)
    })

#Summary Table
print("Optimization Summary")
summary_df = pd.DataFrame(optimal_summary)
print(summary_df.to_string(index=False))

# Compare Eff
plt.figure(figsize=(12, 6), dpi=150)
x_axis = np.arange(0, 11)

for name, risk_values in marginal_risk_history.items():
    initial_val = risk_values[0]
    protection_pct = [100 - (r / (initial_val + 1e-9) * 100) for r in risk_values]
    
    line = plt.plot(x_axis, protection_pct, marker='o', label=name, linewidth=2)
    
    # Mark optimal point
    opt_n = optimal_data_per_strat[name]['n']
    plt.scatter(opt_n, protection_pct[opt_n], s=180, edgecolors='black', color=line[0].get_color(), zorder=5)

plt.title("Comparative Efficiency: Total Protection Coverage", fontsize=14)
plt.xlabel("Number of Ranger Stations")
plt.ylabel("Protection Coverage (%)")
plt.ylim(0, 100)
plt.xticks(x_axis)
plt.grid(True, alpha=0.2)
plt.legend(loc='lower right')
plt.show()

# Marginal Efficiency
plt.figure(figsize=(12, 6), dpi=150)

for name, risk_values in marginal_risk_history.items():
    initial_val = risk_values[0]

    gains = [(risk_values[i-1] - risk_values[i]) / initial_val * 100 for i in range(1, len(risk_values))]
    
    plt.plot(np.arange(1, 11), gains, marker='s', label=name, linewidth=2)

plt.axhline(y=GAIN_THRESHOLD, color='r', linestyle='--', label=f'Threshold ({GAIN_THRESHOLD}%)')
plt.title("Operational Efficiency: Marginal Gain per Additional Station", fontsize=14)
plt.xlabel("Station Number (Added)")
plt.ylabel("Additional Risk Covered (%)")
plt.ylim(0, 100)
plt.xticks(np.arange(1, 11))
plt.grid(True, alpha=0.2)
plt.legend()
plt.show()

# Map Plotting
print("Generating Optimal Maps")

for strat_name, data in optimal_data_per_strat.items():
    plt.figure(figsize=(20, 7), dpi=150)
    ax = plt.gca()
    
    im = ax.imshow(data["map"], cmap=cmap, vmin=GLOBAL_MIN, vmax=GLOBAL_MAX, aspect='equal')
    
    # Burn Mask
    burn_overlay = np.zeros((*data["mask"].shape, 4))
    burn_overlay[data["mask"]] = [0.0, 0.0, 0.0, 0.65]
    ax.imshow(burn_overlay, aspect='equal', zorder=5)
    
    # Plot Stations
    for idx, (y, x) in enumerate(data["stations"]):
        ax.scatter(x, y, color='blue', marker='*', s=700, 
                   edgecolor='white', linewidth=1.5, zorder=10)
        
        # Label station
        ax.text(x + 2, y - 2, f"P{idx+1}", color='white', weight='bold', fontsize=12,
                 bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=1), zorder=11)

    plt.title(f"Optimal Management Strategy: {strat_name}\n"
              f"Recommended Stations: {data['n']} | Total Protection: {data['coverage']:.2f}%", 
              fontsize=18, pad=20)
    
    plt.xlabel("Distance East (km)", fontsize=12)
    plt.ylabel("Distance South(km)", fontsize=12)
    
    cbar = plt.colorbar(im, fraction=0.015, pad=0.04)
    cbar.set_label("Risk Intensity", fontsize=12)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    plt.tight_layout()
    plt.show()


# file saving
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
detail_filename = os.path.join(RESULTS_DIR, f"fire_strategy_detailed_{timestamp}.csv")
summary_filename = os.path.join(RESULTS_DIR, f"fire_strategy_summary_{timestamp}.csv")

df_details = pd.DataFrame(all_scenario_results)
df_details.to_csv(detail_filename, index=False)

df_summary = pd.DataFrame(optimal_summary)
df_summary.to_csv(summary_filename, index=False)

print("\n Data Saved")
print(f"Detailed Log: {detail_filename}")
print(f"Optimal Summary: {summary_filename}")


print("done")
