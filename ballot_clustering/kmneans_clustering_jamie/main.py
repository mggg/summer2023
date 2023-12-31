from parser import parse
from clustering import *
from tqdm import tqdm


test_num = 11

if test_num == 1:
    election, names, location = parse("Data/edinburgh17-16.blt")
    print(len(election))
    print(election[(1, 2)])
    print(names)
    print(location)
elif test_num == 2:
    bg = ballot_graph(3, 3)
    draw(bg)
elif test_num == 3:
    election = {(1, 2, 3, 4, 5, 6): 20, (1, 2, 3): 30, (1, 2): 7}
    bg = ballot_graph(5, 3, election)
    for n in bg.nodes(data=True):
        print(n)
elif test_num == 4:
    election, names, location = parse("Data/edinburgh17-16.blt")
    n = len(names)
    bg = ballot_graph(n, 3, election)
    for n in bg.nodes(data=True):
        print(n)
elif test_num == 5:
    election, names, location = parse("Data/edinburgh17-16.blt")
    n = len(names)
    for i in range(n):
        print(f"{i + 1}: {names[i]}")
    bg = ballot_graph(n, 2, election)
    print("\nk-means centers:")
    print(k_means(bg, n, 2, 2))
elif test_num == 6:
    election, names, location = parse("Data/edinburgh17-16.blt")
    n = len(names)
    bg = ballot_graph(4, 2)
    draw(bg)
elif test_num == 7:
    election, names, location = parse("Data/edinburgh17-16.blt")
    n = len(names)
    bg = ballot_graph(n, 2, election)
    draw(bg)
elif test_num == 8:
    bg = ballot_graph(10, 10)
    print("Got graph")
elif test_num == 9:
    for n in tqdm(range(4, 10)):
        pickle_distance_matrix(n, 4)
elif test_num == 10:
    election, names, location = parse("Data/edinburgh17-16.blt")
    n = len(names)
    for i in range(n):
        print(f"{i + 1}: {names[i]}")
    bg = ballot_graph(n, 4, election)
    print("\nk-means centers:")
    print(k_means(bg, n, 4, 2))
elif test_num == 11:
    for election_num in range(1, 18):
        print(f"\n\nElection {election_num}:\n")
        election, names, location = parse(f"Data/edinburgh17-{election_num:02}.blt")
        n = len(names)
        for i in range(n):
            print(f"{i + 1}: {names[i]}")
        bg = ballot_graph(n, 4, election)
        print("\nk-means centers:")
        try:
            print(k_means(bg, n, 4, 2))
        except:
            print("Can't do it, too many candidates.")
