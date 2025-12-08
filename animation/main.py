import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from benchmarks.benchmark_functions import rosenbrock, schwefel
import pickle

######### USTAW RODZAJ, FUNKCJE I KRYTERIUM
TYPE = "sequential"
# TYPE = "parallel"
FUNCTION_NAME = "Schwefel"
# FUNCTION_NAME = "Rosenbrock"
# CRITERIUM = "kryt_1"
CRITERIUM = "kryt_2"
#########

######### USTAW WIELKOŚĆ PUNKTÓW I WEKTORÓW
POINT_SCALE = 5
VECTOR_SCALE = 10000
#########


METADATA_PATH = f"data/{TYPE}_{FUNCTION_NAME}_{CRITERIUM}_metadata.txt"
POINTS_PATH = f"data/{TYPE}_{FUNCTION_NAME}_{CRITERIUM}_points.txt"

file = open(METADATA_PATH, 'rb')
metadata = pickle.load(file)
file.close()

FUNCTIONS = {"rosenbrock": rosenbrock, "schwefel": schwefel}
ITERATIONS = metadata["Iteracje"]
NUMBER_OF_POINTS = metadata["Liczba punktów"]
FUNCTION_TYPE = metadata["Funkcja"]
BOUNDS = metadata["Ograniczenia"]
DIMENSTIONS = 2

print(f"ITERACJE: {ITERATIONS}" )
print(f"LICZBA PUNKTÓW: {NUMBER_OF_POINTS}" )
print(f"FUNKCJA: {FUNCTION_TYPE}")
print(f"Ograniczenia: {BOUNDS}")

POSITIONS_LIST = []

file = open(POINTS_PATH, 'rb')
for i in range(ITERATIONS):
  POSITIONS_LIST.append(np.load(file))
file.close()

DISTANCE_BETWEEN_PLOT_POINTS = (BOUNDS[1] - BOUNDS[0]) / 100.0

low = np.full((DIMENSTIONS), BOUNDS[0])
high = np.full((DIMENSTIONS), BOUNDS[1])

x = np.arange(BOUNDS[0], BOUNDS[1], DISTANCE_BETWEEN_PLOT_POINTS)
y = np.arange(BOUNDS[0], BOUNDS[1], DISTANCE_BETWEEN_PLOT_POINTS)
z = np.empty((x.shape[0], x.shape[0]))

for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        z[i, j] = FUNCTIONS[FUNCTION_TYPE](np.array([x[i], y[j]]))

fig, ax = plt.subplots()
ax.set_xlim([BOUNDS[0], BOUNDS[1]])
ax.set_ylim([BOUNDS[0], BOUNDS[1]])

image = plt.contourf(x, y, z)
scat = plt.scatter(POSITIONS_LIST[0][:,0], POSITIONS_LIST[0][:,1], c='red', s = POINT_SCALE)

vector_value = POSITIONS_LIST[1] - POSITIONS_LIST[0]
vec = plt.quiver(POSITIONS_LIST[0][:,0], POSITIONS_LIST[0][:,1], vector_value[:,0], vector_value[:,1], color = 'b', scale = VECTOR_SCALE)
def update(frame):
  scat.set_offsets(POSITIONS_LIST[frame + 1])
  vec.set_offsets(POSITIONS_LIST[frame + 1])
  vector_value = POSITIONS_LIST[frame + 1] - POSITIONS_LIST[frame]
  vec.set_UVC(vector_value[:,0], vector_value[:,1])
  return image

animation = animation.FuncAnimation(fig, update, frames=ITERATIONS-1, interval=100, repeat = False)
animation.save(f"output/{TYPE}_{FUNCTION_NAME}_{CRITERIUM}_animation.gif")
plt.show()