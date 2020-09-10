import argparse

parser = argparse.ArgumentParser(description='Runs reinforcement learning algorithm for TSP')
parser.add_argument('--cities',
                    help='File containing cities coordinates permutations with solutions as ordered indexes',
                    type=argparse.FileType('r'))

parser.add_argument('--steps', 's', type=int, help='number of steps')
parser.add_argument('k', type=int, help='number of steps after transition matrix should be updated')

