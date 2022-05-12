import os
import sys
from PIL import Image, ImageFilter
from time import process_time
import numpy as np
from hopfieldnetwork import HopfieldNetwork
from hopfieldnetwork import images2xi, plot_network_development, DATA_DIR, image2numpy_array


def result_to_png(result):
    for i in range(len(result)):
        if result[i] == -1:
            result[i] = 255
        else:
            result[i] = 0

    result = np.reshape(result, (100, 100)).astype(np.uint8)
    print(result.tolist())

    img = Image.fromarray(result, "L")
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    img = img.filter(ImageFilter.SHARPEN)
    img = img.filter(ImageFilter.SMOOTH_MORE)
    img = img.filter(ImageFilter.SHARPEN)
    return img




N = 100 ** 2
path_list = [
    os.path.join(".\\animals", f)
    for f in [
        #"guinea.jpg",
        "hedgehog.jpg",
        "racoon.png"
    ]
]

xi = images2xi(path_list, N)

hopfield_network = HopfieldNetwork(N=N)
hopfield_network.train_pattern(xi)


hedgehog_drawing = image2numpy_array(".\\animals_predict\\hedgehog1.PNG", (100, 100)).flatten()
hopfield_network.set_initial_neurons_state(np.copy(hedgehog_drawing))
plot_network_development(
    hopfield_network, 3, "async", hedgehog_drawing, "hedgehog.pdf"
)

racoon_drawing = image2numpy_array(".\\animals_predict\\racoon1.PNG", (100, 100)).flatten()
hopfield_network.set_initial_neurons_state(np.copy(racoon_drawing))
plot_network_development(
    hopfield_network, 3, "async", racoon_drawing, "racoon.pdf"
)
result_to_png(hopfield_network.S).save("racoon_res.png")


hedgehog_drawing = image2numpy_array(".\\animals_predict\\hedgehog_60s.png", (100, 100)).flatten()
hopfield_network.set_initial_neurons_state(np.copy(hedgehog_drawing))
plot_network_development(
    hopfield_network, 3, "async", hedgehog_drawing, "hedgehog.pdf"
)
result_to_png(hopfield_network.S).save("hedgehog2_res.png")

