# main.py

import numpy as numpy
import matplotlib.pyplot as plt
from svm_class import LinearSVM, MultiClassSVM


def example2d2class():

    print(f"\nLinear SVM with 2D data, 2 class\n")

    x = numpy.array([
        [65,47],
        [61,37],
        [47,46],
        [50,36],
        [62,27],
        [86,25],
        [66,25],
        [91,27],
        [87,28],
        [81,28],
    ], dtype = float)
    y = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    model = LinearSVM(learningRate = 1e-3, lambdaParam = 1e-2, iteration = 1000)
    model.fit(x, y)

    print("w =", model.w)
    print("b =", model.b)
    pred = model.predict(x)
    print("predicted:", pred, " | acutal:", y)

    # plot with matplotlib

    fig, ax = plt.subplots(figsize = (6, 6))

    for cls, marker, color in [(0, "o", "#D85A30"), (1, "s", "#1D9E75")]:

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
        ax.set_xlabel("Humidity (%)")
        ax.set_ylabel("Temperature (℃)")
        ax.set_title("linear SVM 2 classes")
        ax.legend()
        ax.grid(alpha = 0.3)
        fig.tight_layout()

        outpath = "svm2d2class.png"
        fig.savefig(outpath, dpi = 300)
        plt.close(fig)
        print(f"saved plot to '{outpath}'")

        return model

#########################################################################################
#########################################################################################
#########################################################################################

def example2d3class():

    x = numpy.array([
        [65,47],
        [61,37],
        [87,28],
        [81,28],
        [0,0],
        [0,0],
    ], dtype = float)
    y = numpy.array([0, 0 ,1 ,1, 2, 2])

    model = MultiClassSVM(learningRate=1e-3, lambdaParam=1e-2, iteration=3000)
    model.fit(x, y)

    for cls, m in model.model.items():

        print(f"group {cls} vs rest\nw = {m.w}, b = {m.b:.4f}")

    predict = model.predict(x)
    print(f"predicted: {predict}\nactual: {y}")

    # predict a new point

    xNew = numpy.array([[80,25]])

    score = model.decisionScore(xNew)
    predictNew = model.predict(xNew)[0]
    print(f"\nnewpoint: {xNew}\nscore: {score}\npredicted group: {predictNew}")

    # plot with matplotlib

    fig, ax = plt.subplots(figsize = (7, 7))

    color = {0: "#1d9e75", 1: "#d85a30", 2: "#3a6fd8"}
    marker = {0: "o", 1: "s", 2: "^"}

    for cls in model.classes:

        pts = x[y == cls]
        ax.scatter(pts[:, 0], pts[:, 1], marker = marker[cls], s = 130, color = color[cls], edgecolor = "black", label = f"{cls}", zorder = 3)

    ax.scatter(xNew[:, 0], xNew[:, 1], marker = "*", s = 300, color = "#f9c74f", edgecolor = "black", label = f"{int(xNew[0,0])}, {int(xNew[0,1])}", zorder = 4)

    # draw one-vs-rest boundary

    x1val = numpy.linspace(x[:, 0].min() - 2, x[:, 0].max() + 2, 100)
    lineStyle = {0: "-", 1: "--", 2: ":"}

    for cls in model.classes:

        w0, w1 = model.model[cls].w
        b = model.model[cls].b

        if abs(w1) > 1e-9:

            x2val = -(w0 * x1val + b) / w1
            ax.plot(x1val, x2val, lineStyle[cls], color = color[cls], linewidth = 2, label = f"boundary: {cls} vs rest")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("title")

    ax.legend(fontsize = 9)
    ax.grid(alpha = 0.3)

    ax.set_xlim(x[:, 0].min() - 2, x[:, 0].max() + 2)
    ax.set_ylim(x[:, 0].min() - 2, x[:, 0].max() + 2)
    fig.tight_layout()

    # outputting

    outpath = "svm2d3class.png"

    fig.savefig(outpath, dpi = 150)
    plt.close(fig)

    print(f"saved to '{outpath}'")

    return model

#########################################################################################
#########################################################################################
#########################################################################################

if __name__ == "__main__":

    # example2d2class()

    example2d3class()