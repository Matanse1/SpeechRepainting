import numpy as np
import matplotlib.pyplot as plt

# Parameters
dim_model = 192.
warmup_steps = 4000.

# Function definition
def learning_rate(step_num, dim_model, warmup_steps):
    return (np.power(dim_model, -0.5) *
            np.minimum(np.power(step_num, -0.5), step_num * np.power(warmup_steps, -1.7)))

# Generate data
step_nums = np.arange(1, 20000, 1)
learning_rates = learning_rate(step_nums, dim_model, warmup_steps)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(step_nums, learning_rates, label="Learning Rate Schedule")
plt.axvline(x=warmup_steps, color='r', linestyle='--', label="Warmup Steps")
plt.title("Learning Rate Schedule")
plt.xlabel("Step Number")
plt.ylabel("Learning Rate")
plt.legend()
plt.grid()
plt.savefig("/home/dsi/moradim/SpeechRepainting/glow-tts/learning_rate_schedule_1p7.png")
plt.close()
