import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# Initialize figure and axis
fig, ax = plt.subplots()

# Set up the axes limits and labels
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-1, 1)
ax.set_xlabel('X')
ax.set_ylabel('Sine Wave')

# Initialize an empty line object that will be updated during the animation
line, = ax.plot([], [], lw=2)

# Define the initialization function
def init():
    line.set_data([], [])
    return line,

# Define the animation function
def animate(i):
    x = np.linspace(0, 2 * np.pi, 1000)
    y = np.sin(x + 0.1 * i)
    line.set_data(x, y)
    return line,

# Create the animation
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)

# Display the animation
plt.show()
