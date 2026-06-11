import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Fake cell viability data (like your CCK-8 results)
data = {
    "Concentration": [0, 10, 50, 100, 200, 500],
    "HA":            [100, 98, 95, 90, 80, 60],
    "GLY_HA":        [100, 97, 93, 85, 70, 45],
    "PDA_GLY_HA":    [100, 95, 88, 75, 55, 30]
}

df = pd.DataFrame(data)
print(df)

# Plot
sns.lineplot(data=df, x="Concentration", y="HA", label="HA")
sns.lineplot(data=df, x="Concentration", y="GLY_HA", label="GLY-HA")
sns.lineplot(data=df, x="Concentration", y="PDA_GLY_HA", label="PDA-GLY-HA")

plt.xlabel("Concentration (µg/mL)")
plt.ylabel("Cell Viability (%)")
plt.title("CCK-8 Cell Viability")
plt.legend()
plt.tight_layout()
plt.show()