# 3class.py

# 2d3class

import numpy as numpy
import matplotlib.pyplot as plt
from svm_class import MultiClassSVM



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

    # data = numpy.genfromtxt("data2var.csv", delimiter = ",", skip_header = 1)
    # print(f"ข้อมูล X2: {data.shape[0]} แถว,{data.shape[1]} ตัวแปร")
    # x = data[:, 0:2]
    # y = data[:, 2].astype(int)
    # print(x.shape)
    # print(y.shape)

    model = MultiClassSVM(learningRate=1e-3, lambdaParam=1e-2, iteration=3000)
    model.fit(x, y)

    for cls, m in model.model.items():

        print(f"group {cls} vs rest\nw = {m.w}, b = {m.b:.4f}")

    predict = model.predict(x)
    print(f"predicted: {predict}\nactual: {y}")

    # predict a new point 1

    xNew = numpy.array([[60,35]])
    score = model.decisionScore(xNew)
    predictNew = model.predict(xNew)[0]
    print(f"\nnewpoint: {xNew}\nscore: {score}\npredicted group: {predictNew}")

    # predict a new point 2
    
    yNew = numpy.array([[80,25]])
    score = model.decisionScore(yNew)
    yPredictNew = model.predict(yNew)[0]
    print(f"\nnewpoint: {yNew}\nscore: {score}\npredicted group: {yPredictNew}")

    # plot with matplotlib 1

    fig, ax = plt.subplots(figsize = (7, 7))

    color = {0: "#1d9e75", 1: "#d85a30", 2: "#3a6fd8"}
    marker = {0: "o", 1: "s", 2: "^"}

    for cls in model.classes:

        pts = x[y == cls]
        ax.scatter(pts[:, 0], pts[:, 1], marker = marker[cls], s = 130, color = color[cls], edgecolor = "black", label = f"{cls}", zorder = 3)

    ax.scatter(xNew[:, 0], xNew[:, 1], marker = "*", s = 300, color = "#f9c74f", edgecolor = "black", label = f"{int(xNew[0,0])}, {int(xNew[0,1])}", zorder = 4)
    ax.scatter(yNew[:, 0], yNew[:, 1], marker = "*", s = 300, color = "#f9c74f", edgecolor = "black", label = f"{int(yNew[0,0])}, {int(yNew[0,1])}", zorder = 4)

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

    outpath = "TEST2d3c.png"

    fig.savefig(outpath, dpi = 150)
    plt.close(fig)

    print(f"saved to '{outpath}'")

    return model



if __name__ == "__main__":

    example2d3class()