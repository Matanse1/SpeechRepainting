import matplotlib.pyplot as plt
import pandas as pd

# טעינת הנתונים שהורדו מ-TensorBoard
df = pd.read_csv("final_tranning_alon_matan_test_loss.csv")  # שנה לשם הקובץ שלך

# הגדרת סגנון אקדמי נקי
plt.style.use("seaborn-v0_8-paper")
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

# שרטוט הנתונים (מניחים שהעמודות הן Step ו-Value)
ax.plot(df["Step"], df["Value"], label="Training Loss", linewidth=1.5)

# עיצוב הצירים והתוויות
ax.set_xlabel("Steps", fontsize=11)
ax.set_ylabel("Loss", fontsize=11)
ax.set_title("Test Loss Convergence", fontsize=12, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(frameon=True)

plt.tight_layout()

# שמירה באיכות גבוהה עבור Word
plt.savefig("final_tranning_alon_matan_test_loss.png", dpi=300)
plt.show()