import networkx as nx
from tqdm import tqdm
from sys import argv
import pickle
import matplotlib.pyplot as plt


USAGE = "Usage: 'python3 metagraph.py generate|analyze m n k' (where m*n is divisible by k)"


def generate_blocks(block_size):
    if block_size == 1:
        return {frozenset([(0, 0)])}
    else:
        new_blocks = set()
        old_blocks = generate_blocks(block_size - 1)
        for block in old_blocks:
            for old_cell in block:
                x_0, y_0 = old_cell
                for new_cell in [(x_0 + 1, y_0), (x_0 - 1, y_0), (x_0, y_0 + 1), (x_0, y_0 - 1)]:
                    row, col = new_cell
                    if new_cell not in block and (row > 0 or (row == 0 and col >= 0)):
                        new_blocks.add(block.union([new_cell]))
        return new_blocks


def generate_partitions(m, n, k, blocks):
    num_vertices = m*n
    return fill(m, n, k, blocks, ([0 for _ in range(num_vertices)], 0))


def try_to_add(m, n, p, district_number, cells):
    for row, col in cells:
        if row < 0 or row >= n or col < 0 or col >= m:
            return False
        i = m*row + col
        if p[i] == 0:
            p[i] = district_number
        else:
            return False
    return True


def canonize(p, k):
    s = {0}
    renaming = [0 for _ in range(k + 1)]
    new_name = 0
    for old_name in p:
        if not(old_name in s):
            s.add(old_name)
            new_name += 1
            renaming[old_name] = new_name
    return tuple(renaming[old_name] for old_name in p)


def fill(m, n, k, blocks, root):
    stack = [root]
    partitions = []
    while len(stack) > 0:
        p_1, num_parts = stack.pop()
        if num_parts == k:
            partitions.append(tuple(p_1))
        else:
            num_parts += 1
            i = p_1.index(0)
            row_base = int(i / m)
            col_base = i % m
            for cells in blocks:
                p_2 = p_1.copy()
                if try_to_add(m, n, p_2, num_parts,
                              [(row_base + cell[0], col_base + cell[1]) for cell in cells]):
                    stack.append((p_2, num_parts))
    return partitions


def get_element_of_target_size(collection, target_size):
    for element in collection:
        if len(element) == target_size:
            return element
    raise Exception(f"No element found of target size {target_size}.")


def print_nicely(p):
    for row in range(6):
        s = ""
        for col in range(6):
            district_number = p[6*row + col]
            s += chr(ord('@') + district_number) if district_number > 0 else "O"
        print(s)


if __name__ == "__main__":
    try:
        _, command, m_str, n_str, k_str = argv
        m = int(m_str)
        n = int(n_str)
        if n < m:
            t = n_str
            n_str = m_str
            m_str = t
            m = int(m_str)
            n = int(n_str)
        filename = f"metagraph_{m_str}_{n_str}_{k_str}.p"
        k = int(k_str)
        num_vertices = m*n
        block_size = int(num_vertices/k)
        if block_size*k != num_vertices:
            raise Exception()
        if command not in ["generate", "analyze"]:
            raise Exception()
    except:
        command = ""
    if command == "generate":
        blocks = generate_blocks(block_size)
        print("Enumerating partitions...")
        partitions = generate_partitions(m, n, k, blocks)
        print(f"There are {len(partitions)} partitions of a {m}x{n} grid graph into districts of size {block_size}.")
        metagraph = nx.Graph()
        metagraph.add_nodes_from(partitions)
        for p_1 in tqdm(partitions):
            for name_1 in range(1, k + 1):
                for name_2 in range(1, k + 1):
                    if name_1 != name_2:
                        p_1_with_2_districts_removed = list(p_1)
                        for i in range(num_vertices):
                            x = p_1_with_2_districts_removed[i]
                            if x == name_1 or x == name_2:
                                p_1_with_2_districts_removed[i] = 0
                        neighbors = fill(m, n, k, blocks, (list(canonize(p_1_with_2_districts_removed, k)), k - 2))
                        if len(neighbors) > 1:
                            canonical_neighbors = map(lambda p: canonize(p, k), neighbors)
                            for p_2 in canonical_neighbors:
                                if p_1 != p_2:
                                    metagraph.add_edge(p_1, p_2)
        with open(filename, 'wb') as f:
            pickle.dump(metagraph, f)
    elif command == "analyze":
        with open(filename, 'rb') as f:
            metagraph = pickle.load(f)
        print(f"The metagraph has {len(metagraph.nodes())} vertices and {len(metagraph.edges())} edges.")
        components = list(nx.connected_components(metagraph))
        print(f"\nThere are {len(components)} connected components.")
        print("\nSizes of components:")
        component_sizes = list(sorted(map(len, components), reverse=True))
        component_sizes.append(0)
        last_component_size = metagraph.order() + 1
        num_copies = 0
        for c in component_sizes:
            if c == last_component_size:
                num_copies += 1
            else:
                if num_copies > 0:
                    print(f"{last_component_size} x {num_copies}")
                last_component_size = c
                num_copies = 1
    else:
        print(USAGE)

