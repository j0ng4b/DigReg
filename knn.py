import numpy as np
from typing import List


class KNN:
    def __init__(self, k: int = 3):
        self.k = k

        self.x_train = None
        self.y_train = None

    def fit(self, x: List[List[float]], y: List[int]) -> None:
        self.x_train = x
        self.y_train = y

    def predict(self, x: List[List[float]]) -> List[int]:
        return [self.__predict(sample) for sample in x]

    def __predict(self, x: List[float]) -> int:
        distances = []
        for i in range(len(self.x_train)):
            distances.append((self.__euclidean_distance(self.x_train[i], x), self.y_train[i]))
        
        distances.sort(key=lambda x: x[0])
        k_nearest_labels = [distances[i][1] for i in range(self.k)]
        
        label_count = {}
        for label in k_nearest_labels:
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1
        
        max_count = 0
        most_common_label = None
        for label, count in label_count.items():
            if count > max_count:
                max_count = count
                most_common_label = label
        
        return most_common_label

    def __euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum(np.square(point1 - point2)))