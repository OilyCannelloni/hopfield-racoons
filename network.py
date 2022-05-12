from hopfieldnetwork import HopfieldNetwork, images2xi, image2numpy_array, plot_network_development
from PIL import Image
import numpy as np
import os
from random import randint


TARGET_PATH = ".\\pigs_raw"
PREDICT_PATH = ".\\pigs_predict_raw"
NETWORK_PATH = ".\\networks"

NETWORK_NAME = "pigsonly"
OVERWRITE = False
N = 100 ** 2

if __name__ == '__main__':
    network_file = f"{NETWORK_PATH}\\{NETWORK_NAME}"
    network_exists = os.path.isfile(network_file)

    # if not network_exists:
    network = HopfieldNetwork(N=N)
    path_list = [f"{TARGET_PATH}\\{os.fsdecode(filename)}"
                 for filename in os.listdir(os.fsencode(TARGET_PATH)) if randint(1, 100) > 50]

    xi = images2xi(path_list, N)

    network.train_pattern(xi)
    network.save_network(network_file)

    # network = HopfieldNetwork(filepath=f"{network_file}.npz")

    # predict_dir = os.fsencode(PREDICT_PATH)
    # for file in os.listdir(predict_dir):
    #     if os.fsdecode(file).startswith("result"):
    #         continue
    #     filename = f"{PREDICT_PATH}\\{os.fsdecode(file)}"

    # filename = f"{PREDICT_PATH}\\images.jfif"
    # vector = image2numpy_array(filename, (100, 100))
    # network.set_initial_neurons_state(np.copy(vector))
    # plot_network_development(
    #     network,
    #     3,
    #     "async",
    #     np.zeros((100, 100)),
    #     "test.pdf"
    # )

    pig = np.copy(xi[:, 0])
    half_pig = np.copy(xi[:, 0])
    half_pig[: int(N/4)] = -1
    network.set_initial_neurons_state(np.copy(half_pig))
    plot_network_development(
        network,
        6,
        "async",
        pig,
        "test.pdf"
    )







