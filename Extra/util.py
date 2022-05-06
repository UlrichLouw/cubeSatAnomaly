import itertools

def createParams(*args):
    allLists = []

    for arg in args:
        allLists.append(arg)

    return itertools.product(*allLists)