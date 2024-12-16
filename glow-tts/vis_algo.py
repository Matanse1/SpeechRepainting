import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time

# Define the matrix size and initialize the values matrix
t_x, t_y = 10, 10  # Dimensions of the matrix
values = np.random.randint(1, 10, (t_x, t_y))  # Random values in the matrix
max_neg_val = -1e09  # Negative infinity for invalid moves

# Prepare a figure for interactive visualization
fig, ax = plt.subplots(figsize=(8, 8))
cmap = ListedColormap(["white", "lightblue", "blue", "darkblue"])

# Function to update the plot dynamically
def update_plot(matrix, path=None, title=""):
    ax.clear()
    ax.matshow(matrix, cmap="viridis", alpha=0.8)
    if path is not None:
        for y, x in enumerate(path):
            ax.text(x, y, "X", ha="center", va="center", color="red", fontsize=12, fontweight="bold")
    ax.set_xticks(range(t_y))
    ax.set_yticks(range(t_x))
    ax.set_title(title, fontsize=16)
    plt.draw()
    plt.pause(0.5)  # Pause for a brief moment

# Forward pass
for y in range(t_y):
    for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
        v_cur = max_neg_val  # Stay move
        v_diag = max_neg_val  # Diagonal move
        v_jump = max_neg_val  # Jump move

        if y > 0:
            v_cur = values[x, y - 1]  # Stay
        if x > 0 and y > 0:
            v_diag = values[x - 1, y - 1]  # Diagonal
        if x > 1 and y > 0:
            v_jump = values[x - 2, y - 1]  # Jump

        values[x, y] = max(v_cur, v_diag, v_jump) + values[x, y]  # Update the cell

        # Visualize the update
        update_plot(values, title=f"Forward Pass: y={y}, x={x}")

# Backward pass to trace the optimal path
path = [-1] * t_y  # Path initialization
index = np.argmax(values[:, -1])  # Start from the max value in the last column

for y in range(t_y - 1, -1, -1):
    path[y] = index  # Mark the current position in the path

    move_jump = max_neg_val
    move_diagonal = max_neg_val
    move_stay = max_neg_val

    if index > 1 and y > 0:
        move_jump = values[index - 2, y - 1]  # Jump
    if index > 0 and y > 0:
        move_diagonal = values[index - 1, y - 1]  # Diagonal
    if y > 0:
        move_stay = values[index, y - 1]  # Stay

    if move_jump >= max(move_diagonal, move_stay):
        index -= 2
    elif move_diagonal >= move_stay:
        index -= 1

    # Visualize the path tracing
    update_plot(values, path, title=f"Backward Pass: y={y}, index={index}")

plt.show()