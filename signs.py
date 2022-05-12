import os
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



timer = process_time()

N = 100 ** 2
path_list = [
    os.path.join(".\\hands", f)
    for f in [
        #"o.jfif",
        "h.png",
        "v.png"
    ]
]

xi = images2xi(path_list, N)

# Speichere/Trainiere Bilder bekannter Physiker im Netzwerk
hopfield_network = HopfieldNetwork(N=N)
hopfield_network.train_pattern(xi)

v1 = image2numpy_array(".\\hands_raw\\v1.png", (100, 100)).flatten()
hopfield_network.set_initial_neurons_state(np.copy(v1))
plot_network_development(
    hopfield_network, 3, "async", v1, "v1.pdf"
)

print("\nProcess time: {:.3f} s".format(process_time() - timer))
