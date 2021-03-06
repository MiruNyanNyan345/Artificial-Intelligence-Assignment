import time


class Node:
    def __init__(self, node=None, targets=None, parent=None, cost=None):
        self.node = node  # current node (stored in string)
        self.parent = parent  # the parent node of current node
        self.targets = targets  # child node of current node
        self.cost = cost  # traveling cost

        self.g = 0  # the cost we have paid to enter the current state
        self.h = 0  # the lower-bound cost to travel from the current to goal states
        self.f = 0  # the lower-bound overall cost


def reverseSearch(close_list, map_list, start_node, end_node):
    # get the end node
    end = end_node.node
    # pick the node that is the smallest lower-bound to the target node
    nodeToTarget = [node for node in close_list if end_node.node in node.targets]
    for node in nodeToTarget:
        pathCost = None
        for p_idx, path in enumerate(map_list):
            if p_idx > 0:
                if path[0] == node.node and path[1] == end:
                    pathCost = path[2]
        node.cost = ((node.node + end) * pathCost) + node.cost

    path = [end_node]
    currNode = min(nodeToTarget, key=lambda x: x.cost)
    path.append(currNode)
    while currNode.parent:
        currNode = currNode.parent
        path.append(currNode)
    # reverse the order the path list
    path.reverse()
    # construct the final result
    result = ""
    for index, node in enumerate(path):
        if index == len(path) - 1:
            result += str(node.node)
        else:
            result += str(node.node) + " -> "
    # for i in close_list:
    #     print("Node: {}, Parent: {}, Target: {}, Cost: {}, g: {}, h: {}, f: {}".format(
    #         i.node, i.parent, i.targets, i.cost, i.g, i.h, i.f
    #     ))
    return result


def main():
    # [0] = target node
    # [n][0] = start, [n][1] = end, [n][2] = cost
    map_list = [[7],
                [0, 1, 1],
                [0, 2, 3],
                [0, 3, 2],
                [1, 4, 5],
                [1, 6, 3],
                [2, 4, 4],
                [2, 5, 3],
                [3, 5, 2],
                [3, 6, 7],
                [4, 7, 4],
                [5, 7, 1],
                [6, 7, 1]]

    # storing open nodes / un-visit
    openList = []
    # storing closed / visited nodes
    closeList = []

    # node of graph
    nodes = set(sum(map_list, []))
    # get start node from graph
    # initialize the node object of start node
    start = min(nodes)
    start_targets = []
    for path_idx, path in enumerate(map_list):
        if path_idx > 0:
            if path[0] == start:
                start_targets.append(path[1])
    start_node = Node(node=start, targets=start_targets, parent=None)
    start_node.g = 0
    start_node.h = 0
    start_node.f = 0
    start_node.cost = 0
    # get start node from graph
    # initialize the node object of start node
    end = map_list[0][0]
    end_node = Node(node=end, parent=None)
    end_node.g = None
    end_node.h = None
    end_node.f = None

    # append start node into open list
    openList.append(start_node)

    # when the open-list still contain the nodes
    while len(openList) > 0:
        # pick the smallest overall lower-bound node from open list to be the current node
        curr_idx, curr_node = min(enumerate(openList), key=lambda x: x[1].f)
        # remove that current node from open list
        openList.pop(curr_idx)
        # added current node to close list, since it's visited
        closeList.append(curr_node)

        # get the children of current node
        children = []
        for p_idx, path in enumerate(map_list):
            if p_idx > 0:
                if path[0] == curr_node.node:
                    targets = []
                    for subP_idx, subP in enumerate(map_list):
                        if subP_idx > 0:
                            if subP[0] == path[1]:
                                targets.append(subP[1])

                    new_node = Node(node=path[1], targets=targets, parent=curr_node)
                    new_node.g = path[2]
                    children.append(new_node)

        for child in children:
            # calculating the cost of the path from its parent to current child
            child.g += curr_node.g
            children_of_child = []
            for p_idx, path in enumerate(map_list):
                if p_idx > 0:
                    if path[0] == child.node:
                        children_of_child.append(path[2])

            if len(children_of_child) < 1:
                # calculate the final cost of arriving the end node
                if len(openList) == 0:
                    shortest_path = reverseSearch(close_list=closeList, map_list=map_list, start_node=start_node,
                                                  end_node=end_node)
                    print("The shortest path calculated by A* is: {}".format(shortest_path))
                break

            # get the minimum cost of the child path of current child
            min_childPath_cost = min(children_of_child)
            child.h = min_childPath_cost
            # the lower-bound overall cost is the sum of .h and .g
            child.f = child.g + child.h
            # compute the traveling cost
            pathCost = None
            for p_idx, path in enumerate(map_list):
                if p_idx > 0:
                    if path[0] == child.parent.node and path[1] == child.node:
                        pathCost = path[2]
            child.cost = ((child.parent.node + child.node) * pathCost) + curr_node.cost
            openList.append(child)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("Running Time: {}".format(end - start))
