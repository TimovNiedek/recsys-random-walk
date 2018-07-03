from recsysrw.graph import MPDGraph
import multiprocessing
from multiprocessing import Array
import ctypes


def init(graph_file, graph_seed, n_jobs):
    print("Loading full graph")
    global graph
    graph = MPDGraph(graph_file, seed=graph_seed)

    global arr_enabled
    arr_enabled = Array(ctypes.c_bool, len(graph._adjacency), lock=False)

    global pool
    if n_jobs > 0:
        print("Initializing pool")
        pool = multiprocessing.Pool(processes=n_jobs)
    else:
        pool = None
