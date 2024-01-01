from matamp.agents.obstacle import Vertice
from matamp.tools.utils import sqr, left_of, det, absSq


class AgentTreeNode(object):
    def __init__(self):
        self.begin = 0
        self.end = 0
        self.minX = 0.0
        self.maxX = 0.0
        self.minY = 0.0
        self.maxY = 0.0
        self.left = 0
        self.right = 0


class ObstacleTreeNode(object):
    def __init__(self):
        self.begin = 0
        self.end = 0
        self.minX = 0.0
        self.maxX = 0.0
        self.minY = 0.0
        self.maxY = 0.0
        self.left = 0
        self.right = 0


class TargetTreeNode(object):
    def __init__(self):
        self.begin = 0
        self.end = 0
        self.minX = 0.0
        self.maxX = 0.0
        self.minY = 0.0
        self.maxY = 0.0
        self.left = 0
        self.right = 0


class PolyObstacleTreeNode(object):
    def __init__(self):
        self.obstacle_ = None
        self.left_ = None
        self.right_ = None


class FloatPair(object):
    """
    Defines a pair of scalar values.
    """

    def __init__(self, a, b):
        self.a_ = a
        self.b_ = b

    def __lt__(self, other):
        """
        Returns true if the first pair of scalar values is less than the second pair of scalar values.
        """
        return self.a_ < other.a_ or not (other.a_ < self.a_) and self.b_ < other.b_

    def __le__(self, other):
        """
        Returns true if the first pair of scalar values is less than or equal to the second pair of scalar values.
        """
        return (self.a_ == other.a_ and self.b_ == other.b_) or self < other

    def __gt__(self, other):
        """
        Returns true if the first pair of scalar values is greater than the second pair of scalar values.
        """
        return not (self <= other)

    def __ge__(self, other):
        """
        Returns true if the first pair of scalar values is greater than or equal to the second pair of scalar values.
        """
        return not (self < other)


class KdTree(object):
    """
    Defines k-D trees for agents and static obstacles in the simulation.
    """

    def __init__(self, agents, obstacles, targets, polyobs_vertices):
        self.agents = agents
        self.obstacles = obstacles
        self.targets = targets
        self.polyobs_vertices = polyobs_vertices

        # agent kdtree
        self.agentTree = []
        for i in range(2 * len(agents)):
            self.agentTree.append(AgentTreeNode())

        # obstacle kdtree
        self.obstacleTree = []
        for i in range(2 * len(obstacles)):
            self.obstacleTree.append(ObstacleTreeNode())

        # target kdtree
        self.targetTree = []
        for i in range(2 * len(targets)):
            self.targetTree.append(TargetTreeNode())

        self.polyObstacleTree = None

        self.max_leaf_size = 10
        self.epsilon = 1e-5

    def buildAgentTree(self):
        """
        Builds an agent k-D tree.
        """
        if len(self.agents) != 0:
            self.buildAgentTreeRecursive(0, len(self.agents), 0)

    def computeAgentNeighbors(self, agent, rangeSq):
        """
        Computes the agent neighbors of the specified agent.

        Args:
            agent (Agent): The agent for which agent neighbors are to be computed.
            rangeSq (float): The squared range around the agent.
        """
        self.queryAgentTreeRecursive(agent, rangeSq, 0)

    def buildAgentTreeRecursive(self, begin, end, node):
        """
        Recursive method for building an agent k-D tree.

        Args:
            begin (int): The beginning agent k-D tree node node index.
            end (int): The ending agent k-D tree node index.
            node (int): The current agent k-D tree node index.
        """
        self.agentTree[node].begin = begin
        self.agentTree[node].end = end
        self.agentTree[node].minX = self.agentTree[node].maxX = self.agents[begin].pos_global_frame[0]
        self.agentTree[node].minY = self.agentTree[node].maxY = self.agents[begin].pos_global_frame[1]

        for i in range(begin + 1, end):
            self.agentTree[node].maxX = max(self.agentTree[node].maxX, self.agents[i].pos_global_frame[0])
            self.agentTree[node].minX = min(self.agentTree[node].minX, self.agents[i].pos_global_frame[0])
            self.agentTree[node].maxY = max(self.agentTree[node].maxY, self.agents[i].pos_global_frame[1])
            self.agentTree[node].minY = min(self.agentTree[node].minY, self.agents[i].pos_global_frame[1])

        if end - begin > self.max_leaf_size:
            # No leaf node.
            isVertical = self.agentTree[node].maxX - self.agentTree[node].minX > self.agentTree[node].maxY - \
                         self.agentTree[node].minY
            splitValue = 0.5 * (
                self.agentTree[node].maxX + self.agentTree[node].minX if isVertical else self.agentTree[
                                                                                             node].maxY +
                                                                                         self.agentTree[
                                                                                             node].minY)
            left = begin
            right = end

            while left < right:
                while left < right and (
                        self.agents[left].pos_global_frame[0] if isVertical else self.agents[left].pos_global_frame[
                            1]) < splitValue:
                    left += 1

                while right > left and (self.agents[right - 1].pos_global_frame[0] if isVertical else self.agents[
                    right - 1].pos_global_frame[1]) >= splitValue:
                    right -= 1

                if left < right:
                    tempAgent = self.agents[left]
                    self.agents[left] = self.agents[right - 1]
                    self.agents[right - 1] = tempAgent
                    left += 1
                    right -= 1

            leftSize = left - begin

            if leftSize == 0:
                leftSize += 1
                left += 1
                right += 1

            self.agentTree[node].left = node + 1
            self.agentTree[node].right = node + 2 * leftSize

            self.buildAgentTreeRecursive(begin, left, self.agentTree[node].left)
            self.buildAgentTreeRecursive(left, end, self.agentTree[node].right)

    def queryAgentTreeRecursive(self, agent, rangeSq, node):
        """
        Recursive method for computing the agent neighbors of the specified agent.

        Args:
            agent (Agent): The agent for which agent neighbors are to be computed.
            rangeSq (float): The squared range around the agent.
            node (int): The current agent k-D tree node index.
        """
        if self.agentTree[node].end - self.agentTree[node].begin <= self.max_leaf_size:
            for i in range(self.agentTree[node].begin, self.agentTree[node].end):
                agent.insertAgentNeighbor(self.agents[i], rangeSq)
        else:
            distSqLeft = sqr(max(0.0, self.agentTree[self.agentTree[node].left].minX - agent.pos_global_frame[0])) + \
                         sqr(max(0.0, agent.pos_global_frame[0] - self.agentTree[self.agentTree[node].left].maxX)) + \
                         sqr(max(0.0, self.agentTree[self.agentTree[node].left].minY - agent.pos_global_frame[1])) + \
                         sqr(max(0.0, agent.pos_global_frame[1] - self.agentTree[self.agentTree[node].left].maxY))
            distSqRight = sqr(max(0.0, self.agentTree[self.agentTree[node].right].minX - agent.pos_global_frame[0])) + \
                          sqr(max(0.0, agent.pos_global_frame[0] - self.agentTree[self.agentTree[node].right].maxX)) + \
                          sqr(max(0.0, self.agentTree[self.agentTree[node].right].minY - agent.pos_global_frame[1])) + \
                          sqr(max(0.0, agent.pos_global_frame[1] - self.agentTree[self.agentTree[node].right].maxY))

            if distSqLeft < distSqRight:
                if distSqLeft < rangeSq:
                    self.queryAgentTreeRecursive(agent, rangeSq, self.agentTree[node].left)

                    if distSqRight < rangeSq:
                        self.queryAgentTreeRecursive(agent, rangeSq, self.agentTree[node].right)
            else:
                if distSqRight < rangeSq:
                    self.queryAgentTreeRecursive(agent, rangeSq, self.agentTree[node].right)

                    if distSqLeft < rangeSq:
                        self.queryAgentTreeRecursive(agent, rangeSq, self.agentTree[node].left)
        # return rangeSq

    def buildObstacleTree(self):
        """
        Builds an obstacle k-D tree.
        """
        if len(self.obstacles) != 0:
            self.buildObstacleTreeRecursive(0, len(self.obstacles), 0)

    def computeObstacleNeighbors(self, agent, rangeSq):
        """
        Computes the agent neighbors of the specified agent.

        Args:
            agent (Agent): The agent for which agent neighbors are to be computed.
            rangeSq (float): The squared range around the agent.
        """
        self.queryObstacleTreeRecursive(agent, rangeSq, 0)

    def buildObstacleTreeRecursive(self, begin, end, node):
        """
        Recursive method for building an obstacle k-D tree.

        Args:
            begin (int): The beginning obstacle k-D tree node node index.
            end (int): The ending obstacle k-D tree node index.
            node (int): The current obstacle k-D tree node index.
        """
        self.obstacleTree[node].begin = begin
        self.obstacleTree[node].end = end
        self.obstacleTree[node].minX = self.obstacleTree[node].maxX = self.obstacles[begin].pos_global_frame[0]
        self.obstacleTree[node].minY = self.obstacleTree[node].maxY = self.obstacles[begin].pos_global_frame[1]

        for i in range(begin + 1, end):
            self.obstacleTree[node].maxX = max(self.obstacleTree[node].maxX, self.obstacles[i].pos_global_frame[0])
            self.obstacleTree[node].minX = min(self.obstacleTree[node].minX, self.obstacles[i].pos_global_frame[0])
            self.obstacleTree[node].maxY = max(self.obstacleTree[node].maxY, self.obstacles[i].pos_global_frame[1])
            self.obstacleTree[node].minY = min(self.obstacleTree[node].minY, self.obstacles[i].pos_global_frame[1])

        if end - begin > self.max_leaf_size:
            # No leaf node.
            isVertical = self.obstacleTree[node].maxX - self.obstacleTree[node].minX > self.obstacleTree[node].maxY - \
                         self.obstacleTree[node].minY
            splitValue = 0.5 * (
                self.obstacleTree[node].maxX + self.obstacleTree[node].minX if isVertical else self.obstacleTree[
                                                                                                   node].maxY +
                                                                                               self.obstacleTree[
                                                                                                   node].minY)
            left = begin
            right = end

            while left < right:
                while left < right and (
                        self.obstacles[left].pos_global_frame[0] if isVertical else
                        self.obstacles[left].pos_global_frame[1]) < splitValue:
                    left += 1

                while right > left and (self.obstacles[right - 1].pos_global_frame[0] if isVertical else self.obstacles[
                    right - 1].pos_global_frame[1]) >= splitValue:
                    right -= 1

                if left < right:
                    tempobj = self.obstacles[left]
                    self.obstacles[left] = self.obstacles[right - 1]
                    self.obstacles[right - 1] = tempobj
                    left += 1
                    right -= 1

            leftSize = left - begin

            if leftSize == 0:
                leftSize += 1
                left += 1
                right += 1

            self.obstacleTree[node].left = node + 1
            self.obstacleTree[node].right = node + 2 * leftSize

            self.buildObstacleTreeRecursive(begin, left, self.obstacleTree[node].left)
            self.buildObstacleTreeRecursive(left, end, self.obstacleTree[node].right)

    def queryObstacleTreeRecursive(self, agent, rangeSq, node):
        """
        Recursive method for computing the agent neighbors of the specified obstacle.

        Args:
            agent (Agent): The agent for which agent neighbors are to be computed.
            rangeSq (float): The squared range around the obstacle.
            node (int): The current obstacle k-D tree node index.
        """
        if self.obstacleTree[node].end - self.obstacleTree[node].begin <= self.max_leaf_size:
            for i in range(self.obstacleTree[node].begin, self.obstacleTree[node].end):
                agent.insertObstacleNeighbor(self.obstacles[i], rangeSq)
        else:
            distSqLeft = sqr(
                max(0.0, self.obstacleTree[self.obstacleTree[node].left].minX - agent.pos_global_frame[0])) + \
                         sqr(max(0.0,
                                 agent.pos_global_frame[0] - self.obstacleTree[self.obstacleTree[node].left].maxX)) + \
                         sqr(max(0.0,
                                 self.obstacleTree[self.obstacleTree[node].left].minY - agent.pos_global_frame[1])) + \
                         sqr(max(0.0, agent.pos_global_frame[1] - self.obstacleTree[self.obstacleTree[node].left].maxY))
            distSqRight = sqr(
                max(0.0, self.obstacleTree[self.obstacleTree[node].right].minX - agent.pos_global_frame[0])) + \
                          sqr(max(0.0,
                                  agent.pos_global_frame[0] - self.obstacleTree[self.obstacleTree[node].right].maxX)) + \
                          sqr(max(0.0,
                                  self.obstacleTree[self.obstacleTree[node].right].minY - agent.pos_global_frame[1])) + \
                          sqr(max(0.0,
                                  agent.pos_global_frame[1] - self.obstacleTree[self.obstacleTree[node].right].maxY))

            if distSqLeft < distSqRight:
                if distSqLeft < rangeSq:
                    self.queryObstacleTreeRecursive(agent, rangeSq, self.obstacleTree[node].left)

                    if distSqRight < rangeSq:
                        self.queryObstacleTreeRecursive(agent, rangeSq, self.obstacleTree[node].right)
            else:
                if distSqRight < rangeSq:
                    self.queryObstacleTreeRecursive(agent, rangeSq, self.obstacleTree[node].right)

                    if distSqLeft < rangeSq:
                        self.queryObstacleTreeRecursive(agent, rangeSq, self.obstacleTree[node].left)
        # return rangeSq

    def buildTargetTree(self):
        """
        Builds an obstacle k-D tree.
        """
        if len(self.targets) != 0:
            self.buildTargetTreeRecursive(0, len(self.targets), 0)

    def computeTargetNeighbors(self, agent, rangeSq):
        """
        Computes the agent neighbors of the specified agent.

        Args:
            agent (Agent): The agent for which agent neighbors are to be computed.
            rangeSq (float): The squared range around the agent.
        """
        self.queryTargetTreeRecursive(agent, rangeSq, 0)

    def buildTargetTreeRecursive(self, begin, end, node):
        """
        Recursive method for building an obstacle k-D tree.

        Args:
            begin (int): The beginning obstacle k-D tree node node index.
            end (int): The ending obstacle k-D tree node index.
            node (int): The current obstacle k-D tree node index.
        """
        self.targetTree[node].begin = begin
        self.targetTree[node].end = end
        self.targetTree[node].minX = self.targetTree[node].maxX = self.targets[begin].pos_global_frame[0]
        self.targetTree[node].minY = self.targetTree[node].maxY = self.targets[begin].pos_global_frame[1]

        for i in range(begin + 1, end):
            self.targetTree[node].maxX = max(self.targetTree[node].maxX, self.targets[i].pos_global_frame[0])
            self.targetTree[node].minX = min(self.targetTree[node].minX, self.targets[i].pos_global_frame[0])
            self.targetTree[node].maxY = max(self.targetTree[node].maxY, self.targets[i].pos_global_frame[1])
            self.targetTree[node].minY = min(self.targetTree[node].minY, self.targets[i].pos_global_frame[1])

        if end - begin > self.max_leaf_size:
            # No leaf node.
            isVertical = self.targetTree[node].maxX - self.targetTree[node].minX > self.targetTree[node].maxY - \
                         self.targetTree[node].minY
            splitValue = 0.5 * (
                self.targetTree[node].maxX + self.targetTree[node].minX if isVertical else self.targetTree[
                                                                                               node].maxY +
                                                                                           self.targetTree[
                                                                                               node].minY)
            left = begin
            right = end

            while left < right:
                while left < right and (
                        self.targets[left].pos_global_frame[0] if isVertical else self.targets[left].pos_global_frame[
                            1]) < splitValue:
                    left += 1

                while right > left and (self.targets[right - 1].pos_global_frame[0] if isVertical else self.targets[
                    right - 1].pos_global_frame[1]) >= splitValue:
                    right -= 1

                if left < right:
                    tempobj = self.targets[left]
                    self.targets[left] = self.targets[right - 1]
                    self.targets[right - 1] = tempobj
                    left += 1
                    right -= 1

            leftSize = left - begin

            if leftSize == 0:
                leftSize += 1
                left += 1
                right += 1

            self.targetTree[node].left = node + 1
            self.targetTree[node].right = node + 2 * leftSize

            self.buildTargetTreeRecursive(begin, left, self.targetTree[node].left)
            self.buildTargetTreeRecursive(left, end, self.targetTree[node].right)

    def queryTargetTreeRecursive(self, agent, rangeSq, node):
        """
        Recursive method for computing the agent neighbors of the specified target.

        Args:
            agent (Agent): The agent for which agent neighbors are to be computed.
            rangeSq (float): The squared range around the target.
            node (int): The current target k-D tree node index.
        """
        if self.targetTree[node].end - self.targetTree[node].begin <= self.max_leaf_size:
            for i in range(self.targetTree[node].begin, self.targetTree[node].end):
                agent.insertTargetNeighbor(self.targets[i], rangeSq)
        else:
            distSqLeft = sqr(max(0.0, self.targetTree[self.targetTree[node].left].minX - agent.pos_global_frame[0])) + \
                         sqr(max(0.0, agent.pos_global_frame[0] - self.targetTree[self.targetTree[node].left].maxX)) + \
                         sqr(max(0.0, self.targetTree[self.targetTree[node].left].minY - agent.pos_global_frame[1])) + \
                         sqr(max(0.0, agent.pos_global_frame[1] - self.targetTree[self.targetTree[node].left].maxY))
            distSqRight = sqr(max(0.0, self.targetTree[self.targetTree[node].right].minX - agent.pos_global_frame[0])) + \
                          sqr(max(0.0, agent.pos_global_frame[0] - self.targetTree[self.targetTree[node].right].maxX)) + \
                          sqr(max(0.0, self.targetTree[self.targetTree[node].right].minY - agent.pos_global_frame[1])) + \
                          sqr(max(0.0, agent.pos_global_frame[1] - self.targetTree[self.targetTree[node].right].maxY))

            if distSqLeft < distSqRight:
                if distSqLeft < rangeSq:
                    self.queryTargetTreeRecursive(agent, rangeSq, self.targetTree[node].left)

                    if distSqRight < rangeSq:
                        self.queryTargetTreeRecursive(agent, rangeSq, self.targetTree[node].right)
            else:
                if distSqRight < rangeSq:
                    self.queryTargetTreeRecursive(agent, rangeSq, self.targetTree[node].right)

                    if distSqLeft < rangeSq:
                        self.queryTargetTreeRecursive(agent, rangeSq, self.targetTree[node].left)
        # return rangeSq

    def buildPolyObstacleTree(self):
        """
        Builds an polygonal obstacle k-D tree.
        """
        poly_obstacles = self.polyobs_vertices
        self.polyObstacleTree = self.buildPolyObstacleTreeRecursive(poly_obstacles)

    def computePolyObstacleNeighbors(self, agent, rangeSq):
        """
        Computes the obstacle neighbors of the specified agent.

        Args:
            agent (Agent): The agent for which obstacle neighbors are to be computed.
            rangeSq (float): The squared range around the agent.
        """
        self.queryPolyObstacleTreeRecursive(agent, rangeSq, self.polyObstacleTree)

    def buildPolyObstacleTreeRecursive(self, poly_obstacles):
        """
        Recursive method for building an obstacle k-D tree.

        Args:
            poly_obstacles (list): A list of poly_obstacles.

        Returns:
            PolyObstacleTreeNode: An polygonal obstacle k-D tree node.
        """
        if len(poly_obstacles) == 0:
            return None

        node = PolyObstacleTreeNode()

        optimalSplit = 0
        minLeft = len(poly_obstacles)
        minRight = len(poly_obstacles)

        for i in range(len(poly_obstacles)):
            leftSize = 0
            rightSize = 0

            obstacleI1 = poly_obstacles[i]
            obstacleI2 = obstacleI1.next_

            # Compute optimal split node.
            for j in range(len(poly_obstacles)):
                if i == j:
                    continue

                obstacleJ1 = poly_obstacles[j]
                obstacleJ2 = obstacleJ1.next_

                j1LeftOfI = left_of(obstacleI1.point_, obstacleI2.point_, obstacleJ1.point_)
                j2LeftOfI = left_of(obstacleI1.point_, obstacleI2.point_, obstacleJ2.point_)

                if j1LeftOfI >= -self.epsilon and j2LeftOfI >= -self.epsilon:
                    leftSize += 1
                elif j1LeftOfI <= self.epsilon and j2LeftOfI <= self.epsilon:
                    rightSize += 1
                else:
                    leftSize += 1
                    rightSize += 1

                if FloatPair(max(leftSize, rightSize), min(leftSize, rightSize)) >= FloatPair(max(minLeft, minRight), min(minLeft, minRight)):
                    break

            if FloatPair(max(leftSize, rightSize), min(leftSize, rightSize)) < FloatPair(max(minLeft, minRight), min(minLeft, minRight)):
                minLeft = leftSize
                minRight = rightSize
                optimalSplit = i

        # Build split node.
        leftObstacles = [None for _ in range(minLeft)]
        rightObstacles = [None for _ in range(minRight)]

        leftCounter = 0
        rightCounter = 0
        i = optimalSplit

        obstacleI1 = poly_obstacles[i]
        obstacleI2 = obstacleI1.next_

        for j in range(len(poly_obstacles)):
            if i == j:
                continue

            obstacleJ1 = poly_obstacles[j]
            obstacleJ2 = obstacleJ1.next_

            j1LeftOfI = left_of(obstacleI1.point_, obstacleI2.point_, obstacleJ1.point_)
            j2LeftOfI = left_of(obstacleI1.point_, obstacleI2.point_, obstacleJ2.point_)

            if j1LeftOfI >= -self.epsilon and j2LeftOfI >= -self.epsilon:
                leftObstacles[leftCounter] = poly_obstacles[j]
                leftCounter += 1
            elif j1LeftOfI <= self.epsilon and j2LeftOfI <= self.epsilon:
                rightObstacles[rightCounter] = poly_obstacles[j]
                rightCounter += 1
            else:
                # Split obstacle j.
                t = det(obstacleI2.point_ - obstacleI1.point_, obstacleJ1.point_ - obstacleI1.point_) / det(obstacleI2.point_ - obstacleI1.point_, obstacleJ1.point_ - obstacleJ2.point_)

                splitPoint = obstacleJ1.point_ + t * (obstacleJ2.point_ - obstacleJ1.point_)

                newObstacle = Vertice()
                newObstacle.point_ = splitPoint
                newObstacle.previous_ = obstacleJ1
                newObstacle.next_ = obstacleJ2
                newObstacle.convex_ = True
                newObstacle.direction_ = obstacleJ1.direction_

                newObstacle.id_ = len(self.polyobs_vertices)

                self.polyobs_vertices.append(newObstacle)

                obstacleJ1.next_ = newObstacle
                obstacleJ2.previous_ = newObstacle

                if j1LeftOfI > 0.0:
                    leftObstacles[leftCounter] = obstacleJ1
                    leftCounter += 1
                    rightObstacles[rightCounter] = newObstacle
                    rightCounter += 1
                else:
                    rightObstacles[rightCounter] = obstacleJ1
                    rightCounter += 1
                    leftObstacles[leftCounter] = newObstacle
                    leftCounter += 1

        node.obstacle_ = obstacleI1
        node.left_ = self.buildPolyObstacleTreeRecursive(leftObstacles)
        node.right_ = self.buildPolyObstacleTreeRecursive(rightObstacles)

        return node

    def queryPolyObstacleTreeRecursive(self, agent, rangeSq, node):
        """
        Recursive method for computing the obstacle neighbors of the specified agent.

        Args:
            agent (Agent): The agent for which polygonal obstacle neighbors are to be computed.
            rangeSq (float): The squared range around the agent.
            node (PolyObstacleTreeNode): The current polygonal obstacle k-D node.
        """
        if node is not None:
            obstacle1 = node.obstacle_
            obstacle2 = obstacle1.next_

            agentLeftOfLine = left_of(obstacle1.point_, obstacle2.point_, agent.pos_global_frame)

            self.queryPolyObstacleTreeRecursive(agent, rangeSq, node.left_ if agentLeftOfLine >= 0.0 else node.right_)

            distSqLine = sqr(agentLeftOfLine) / absSq(obstacle2.point_ - obstacle1.point_)

            if distSqLine < rangeSq:
                if agentLeftOfLine < 0.0:
                    # Try obstacle at this node only if agent is on right side of obstacle (and can see obstacle).
                    agent.insertPolyObstacleNeighbor(node.obstacle_, rangeSq)

                # Try other side of line.
                self.queryPolyObstacleTreeRecursive(agent, rangeSq, node.right_ if agentLeftOfLine >= 0.0 else node.left_)
