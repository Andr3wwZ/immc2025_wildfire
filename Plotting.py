import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


file_path = "fire_strategy_detailed_20260406_1428.csv"
df = pd.read_csv(file_path)


strategies = df['Strategy'].unique()
colors = plt.cm.gist_rainbow(np.linspace(0, .8, len(strategies)))
GAIN_THRESHOLD = 2.5

#Comparitive Efficiency
plt.figure(figsize=(12, 6), dpi=150)

for i, strat in enumerate(strategies):
    subset = df[df['Strategy'] == strat].sort_values('Stations')
    

    plt.plot(subset['Stations'], subset['Total_Protection_%'], 
             marker='o', label=strat, color=colors[i], linewidth=2)
    

    subset['gain'] = subset['Total_Protection_%'].diff().fillna(subset['Total_Protection_%'])

    optimal_row = subset[subset['gain'] >= GAIN_THRESHOLD].iloc[-1:]
    
    if not optimal_row.empty:
        plt.scatter(optimal_row['Stations'], optimal_row['Total_Protection_%'], 
                    s=200, edgecolors='black', color=colors[i], zorder=5)

plt.title("Total Protection Coverage", fontsize=14)
plt.xlabel("Number of Ranger Stations", fontsize=11)
plt.ylabel("Total Risk Protected (%)", fontsize=11)
plt.ylim(0, 105)
plt.xticks(range(0, 11))
plt.grid(True, alpha=0.2)
plt.legend(loc='lower right')
plt.tight_layout()
# plt.savefig("comparative_efficiency_final.png")
plt.show()

#Marginal Gain
plt.figure(figsize=(12, 6), dpi=150)

for i, strat in enumerate(strategies):
    subset = df[df['Strategy'] == strat].sort_values('Stations')
    
    #Calculate Gain
    gains = subset['Total_Protection_%'].diff().fillna(subset.iloc[0]['Total_Protection_%'])
    
    plt.plot(subset['Stations'], gains, marker='s', label=strat, color=colors[i], linewidth=2)

plt.axhline(y=GAIN_THRESHOLD, color='r', linestyle='--', label=f'Threshold ({GAIN_THRESHOLD}%)')
plt.title("Marginal Gain per Additional Station", fontsize=14)
plt.xlabel("Number of Ranger Stations")
plt.ylabel("Additional Risk Covered (%)")
plt.ylim(0, max(df['Total_Protection_%'].diff().max(), 40)) # Dynamic Y-limit
plt.xticks(range(1, 11))
plt.grid(True, alpha=0.2)
plt.legend(loc='upper right')
plt.tight_layout()
# plt.savefig("marginal_gain_final.png")
plt.show()