# Copy from: https://github.com/gary9987/tensorflow-get-started/blob/main/gcn/transformation.py
import numpy as np
from spektral.data import Graph


class ReshapeYTransform:
    def __call__(self, graph):
        if graph.y is not None:
            # Remove the columns of features include parameters and FLOPs
            graph.y = graph.y.reshape(graph.y.shape[0] * graph.y.shape[1])

        return graph

class OnlyValidAccTransform:
    def __call__(self, graph):
        if graph.y is not None:
            graph.y = graph.y[1, :]

        return graph

class RemoveParAndFlopTransform:
    def __call__(self, graph):
        if graph.x is not None:
            # Remove the columns of features include parameters and FLOPs
            graph.x = np.delete(graph.x, [7, 8], 1)

        return graph


class OneHotToIndexTransform:
    def __call__(self, graph):
        if graph.x is not None:
            try:
                tmp = np.delete(graph.x, [7, 8], 1)
            except:
                tmp = graph.x

            # set position 0 as the feature index
            graph.x[:, 0] = np.argmax(tmp, axis=1)
            # remove one-hot (position 1 to 6)
            graph.x = np.delete(graph.x, range(1, 7), 1)

        return graph


class NormalizeParAndFlop:
    def __call__(self, graph):
        if graph.x is not None:
            flops = graph.x[:, 7] - 28819043.233719405
            flops /= 68531284.19735347
            graph.x[:, 7] = flops

            params = graph.x[:, 8] - 98277.40047462686
            params /= 332440.6417713961
            graph.x[:, 8] = flops

        return graph


class NormalizeParAndFlop_NasBench101:
    def __call__(self, graph):
        if graph.x is not None:
            flops = graph.x[:, 7] - 28108567.14472483
            flops /= 67398823.71203184
            graph.x[:, 7] = flops

            params = graph.x[:, 8] - 95841.84206601226
            params /= 326745.84386388084
            graph.x[:, 8] = flops

        return graph


class SelectLabelQueryIdx_NasBench101:
    def __init__(self, idx):
        self.idx = idx

    def __call__(self, graph):
        if graph.y is not None:
            new_y = np.array([graph.y[self.idx][i] for i in range(graph.y.shape[1])])
            graph.y = new_y

        return graph


class SelectNoneNanData_NasBench101:
    def __call__(self, graph):
        if graph.y is not None:
            new_y = np.array([0 for _ in range(graph.y.shape[1])])
            for idx in range(graph.y.shape[0]):
                if not np.isnan(graph.y[idx][0]):
                    # Select the highest test acc
                    if graph.y[idx][2] > new_y[2]:
                        new_y = np.array([graph.y[idx][i] for i in range(graph.y.shape[1])])

            graph.y = new_y

        return graph


class LabelScale:
    def __init__(self, scale=100):
        self.scale = scale

    def __call__(self, graph):
        if graph.y is not None:
            graph.y *= self.scale

        return graph


class RemoveTrainingTime_NasBench101:
    def __call__(self, graph):
        if graph.y is not None:
            graph.y = np.delete(graph.y, [3], 1)

        return graph


class NormalizeLayer_NasBench101:
    def __call__(self, graph):
        if graph.x is not None:
            flops = graph.x[:, 9] - 30.528941727204867
            flops /= 17.807043336964252
            graph.x[:, 9] = flops

        return graph


class Normalize_x_10to15_NasBench101:
    def __call__(self, graph):
        if graph.x is not None:
            flops = graph.x[:, 10] / 32
            graph.x[:, 10] = flops

            flops = graph.x[:, 11] / 32
            graph.x[:, 11] = flops

            flops = graph.x[:, 12] / 512
            graph.x[:, 12] = flops

            flops = graph.x[:, 13] / 32
            graph.x[:, 13] = flops

            flops = graph.x[:, 14] / 32
            graph.x[:, 14] = flops

            flops = graph.x[:, 15] / 256
            graph.x[:, 15] = flops

        return graph


class NormalizeEdgeFeature_NasBench101:
    def __call__(self, graph):
        if graph.e is not None:
            graph.e /= 512
        return graph


class RemoveEdgeFeature_NasBench101:
    def __call__(self, graph):
        ret = Graph(x=graph.x, a=graph.a, y=graph.y)
        return ret

class OnlyFinalAcc:
    def __call__(self, graph):
        if graph.y is not None:
            graph = Graph(x=graph.x, a=graph.a, y=graph.y[-1:])
        return graph