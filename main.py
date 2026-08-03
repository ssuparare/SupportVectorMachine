# main.py

import numpy as numpy
import matplotlib.pyplot as plt
from svm_class import LinearSVM


def example2d2class():

    print(f"\n Linear SVM with 2D data, 2 class \n")

    # class 0: (1,2), (2,3)
    # class 1: (7,9), (8,5)

    x = numpy.array([
        [1,2],
        [2,3],
        [7,9],
        [8,5],
    ], dtype = float)
    y = numpy.array([0, 0, 1, 1])

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

        outpath = "svm2d2class.png"
        fig.savefig(outpath, dpi = 150)
        plt.close(fig)
        print(f"saved plot to '{outpath}'")

        return model



if __name__ == "__main__":

    example2d2class()