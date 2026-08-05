# main.py

import numpy as numpy
import matplotlib.pyplot as plt
from svm_class import LinearSVM


def example2d2class():

    print(f"\nLinear SVM with 2D data, 2 class\n")

    # class 0: (1,2), (2,3)
    # class 1: (7,9), (8,5)

    x = numpy.array([
        [66,40],
        [50,27],
        [68,47],
        [50,45],
        [68,31],
        [51,46],
        [44,36],
        [48,34],
        [62,44],
        [64,39],
        [45,45],
        [50,29],
        [41,46],
        [67,34],
        [44,33],
        [59,31],
        [53,35],
        [42,41],
        [57,39],
        [63,39],
        [60,28],
        [40,35],
        [64,44],
        [55,39],
        [55,32],
        [65,47],
        [61,37],
        [47,46],
        [50,36],
        [62,27],
        [62,38],
        [44,32],
        [43,46],
        [42,32],
        [47,27],
        [43,27],
        [51,44],
        [49,46],
        [51,42],
        [47,41],
        [51,32],
        [69,33],
        [54,29],
        [59,44],
        [56,44],
        [46,38],
        [59,37],
        [47,28],
        [70,27],
        [54,37],
        [67,32],
        [83,26],
        [92,25],
        [87,31],
        [94,30],
        [94,26],
        [84,25],
        [93,34],
        [91,26],
        [84,35],
        [86,25],
        [66,25],
        [91,27],
        [87,28],
        [81,28],
        [93,26],
        [94,26],
        [82,35],
        [74,26],
        [88,26],
        [75,35],
        [80,33],
        [78,27],
        [65,33],
        [67,33],
        [89,28],
        [89,33],
        [92,34],
        [89,30],
        [89,28],
        [70,25],
        [69,29],
        [76,25],
        [69,30],
        [67,30],
        [92,35],
        [86,34],
        [93,32],
        [86,34],
        [83,25],
        [91,28],
        [79,25],
        [86,27],
        [79,35],
        [87,28],
        [78,25],
        [80,26],
        [75,35],
        [92,28],
        [89,25],
    ], dtype = float)
    y = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    model = LinearSVM(learningRate = 1e-3, lambdaParam = 1e-2, iteration = 1000)
    model.fit(x, y)

    print("w =", model.w)
    print("b =", model.b)
    pred = model.predict(x)
    print("predicted:", pred, " | acutal:", y)


    # plot with matplotlib

    fig, ax = plt.subplots(figsize = (6, 6))

    for cls, marker, color in [(0, "o", "#1D9E75"), (1, "s", "#D85A30")]:

        pts = x[y == cls]
        ax.scatter(pts[:, 0], pts[:, 1], marker = marker, s = 120,
        color = color, edgecolor = "black", label = f"class {cls}", zorder = 3)

    
    # decision boundary: w0 * x1 + w1 * x2 + b = 0 -> x2 = -(w0 * x1 + b) / w1

    x1val = numpy.linspace(x[:, 0].min() - 2, x[:, 0].max() + 2, 100)
    w0, w1 = model.w
    b = model.b

    if abs(w1) > 1e-9:

        x2val = -(w0 * x1val + b) / w1
        ax.plot(x1val, x2val, "k-", linewidth = 2, label = "decision boundary")


        # margin line
        # w.x + b = +1 and -1
        
        x2plus = -(w0 * x1val + b - 1) / w1
        x2minus = -(w0 * x1val + b + 1) / w1

        ax.plot(x1val, x2plus, "k--", linewidth = 1, alpha = 0.5)
        ax.plot(x1val, x2minus, "k--", linewidth = 1, alpha = 0.5)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("linear SVM 2 classes")
        ax.legend()
        ax.grid(alpha = 0.3)
        fig.tight_layout()

        outpath = "svm2d2class100.png"
        fig.savefig(outpath, dpi = 150)
        plt.close(fig)
        print(f"saved plot to '{outpath}'")

        return model



if __name__ == "__main__":

    example2d2class()