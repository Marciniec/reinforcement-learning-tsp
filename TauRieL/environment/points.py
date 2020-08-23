from train.distance import route_distance


class Points:
    def __init__(self, indices, permutation, original_permutation, probabilities=None):
        self.indices = indices
        self.permutation = permutation
        self.original_permutation = original_permutation
        self.probabilities = probabilities
        self.length = None

    def calculate_length(self):
        self.length = route_distance(self.permutation)
