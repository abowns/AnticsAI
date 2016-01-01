
import random
from Player import *
from Constants import *
from Construction import *
from Ant import *
from Move import Move
from GameState import *
from AIPlayerUtils import *

# constants
DEPTH_LIMIT = 2

##
# AIPlayer
# Description: The responsbility of this class is to interact with the game by
# deciding a valid move based on a given game state. This class has methods that
# will be implemented by students in Dr. Nuxoll's AI course.
#
# Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):
    # __init__
    # Description: Creates a new Player
    #
    # Parameters:
    #   inputPlayerId - The id to give the new player (int)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer, self).__init__(inputPlayerId, "Very Very Smart")

    ##
    # getPlacement
    #
    # Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    # Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    # Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        # implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:  # stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    # Choose any x location
                    x = random.randint(0, 9)
                    # Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    # Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        # Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:  # stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    # Choose any x location
                    x = random.randint(0, 9)
                    # Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    # Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        # Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]

    ##
    #getMove
    #Description: Gets the next move from the Player.
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    #Return: The Move to be made
    ##
    def getMove(self, currentState):
        bestMove = self.findBestMove(currentState, self.playerId, 0)
        return bestMove

    ##
    # getAttack
    # Description: Gets the attack to be made from the Player
    #
    # Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        # Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    # getFutureState
    # Description: Simulates and returns the future state that would exist after
    #   a move is applied to current state
    #
    # Parameters:
    #   currentState - A clone of the current state (GameState)
    #   nextMove - The Move that is to be simulated
    #
    # Return: The simulated future state (GameState)
    ##
    def getFutureState(self, currentState, nextMove):
        """
        :type currentState: GameState
        :type nextMove: Move
        :rtype: GameState
        """

        # for ending turns, there is no change to the board
        if nextMove.moveType == END:
            return currentState

        futureState = currentState.fastclone()
        """:type : GameState"""

        playerId = currentState.whoseTurn
        enemyId = playerId * (-1) + 1

        for ant in futureState.inventories[playerId].ants:
            ant.hasMoved = False

        if nextMove.moveType == BUILD:
            if nextMove.buildType == TUNNEL:
                futureState.inventories[playerId].foodCount -= CONSTR_STATS[TUNNEL][BUILD_COST]
                futureState.inventories[playerId].constrs.append(
                    Construction(nextMove.coordList[0], nextMove.buildType))
            else:
                # build an ant
                futureState.inventories[playerId].foodCount -= UNIT_STATS[nextMove.buildType][COST]
                futureState.inventories[playerId].ants.append(
                    Ant(nextMove.coordList[0], nextMove.buildType, playerId))

        else:
            # the move is a MOVE_ANT move, so
            # find the ant and move it to the target
            ant = getAntAt(futureState, nextMove.coordList[0])
            """:type : Ant"""
            ant.coords = nextMove.coordList[-1]

            # see if it moved onto a structure
            constr = getConstrAt(futureState, nextMove.coordList[-1])

            if constr and ant.type == WORKER:
                # pickup/dropoff food
                if ant.carrying and (constr.type == ANTHILL or constr.type == TUNNEL):
                    ant.carrying = False
                    futureState.inventories[playerId].foodCount += 1
                elif not ant.carrying and constr.type == FOOD:
                    ant.carrying = True

            # search for possible attacks and execute one
            for coords in listAdjacent(ant.coords):

                targetAnt = getAntAt(futureState, coords)
                """:type : Ant"""

                # verify that the targetAnt is an enemy
                if targetAnt and targetAnt.player == playerId:
                    targetAnt = None

                if targetAnt:
                    targetAnt.health -= UNIT_STATS[ant.type][ATTACK]
                    # remove from board if enemy is killed
                    if targetAnt.health < 1:
                        futureState.inventories[enemyId].ants.remove(targetAnt)
                    continue

        return futureState

    ##
    # evaluateState
    # Description: Evaluates how "good" a state is based on whose turn it is and
    #   what is on the board
    #
    # Parameters:
    #   state - The GameState to evaluate
    #
    # Return: a double between 0.0 (loss) and 1.0 (victory)
    ##
    def evaluateState(self, state):
        """
        :type state: GameState
        """

        # local constants for adjusting weights of evaluation
        # (these should add up to 1.0 or less)

        EVAL_NUM_ANTS = 0.2
        EVAL_HEALTH = 0.3
        EVAL_FOOD = 0.1
        EVAL_CARRYING = 0.01
        EVAL_POSITION = 0.1
        EVAL_QUEEN_ON_BUILDING = 0.1
        EVAL_PARK_ENEMY_ANTHILL = 0.1

        # do some lookups for variables we will use more than once
        # (to save on time)

        playerId = state.whoseTurn
        enemyId = playerId * (-1) + 1

        playerQueen = state.inventories[playerId].getQueen()
        enemyQueen = state.inventories[enemyId].getQueen()

        playerAnthill = state.inventories[playerId].getAnthill()
        enemyAnthill = state.inventories[enemyId].getAnthill()

        playerFoodCount = state.inventories[playerId].foodCount
        enemyFoodCount = state.inventories[enemyId].foodCount

        # search for win/loss
        if (not enemyQueen or enemyAnthill.captureHealth <= 0
                    or playerFoodCount >= FOOD_GOAL):
            return 1.0
        if (not playerQueen or playerAnthill.captureHealth <= 0
                    or enemyFoodCount >= FOOD_GOAL):
            return 0.0

        # start with a "neutral" rating of 0.5
        eval = 0.5

        # PART 1: number of ants
        # (also sets vars for other parts)

        numWorkers = 0
        numDrones = 0
        playerHealthSum = 0
        antHasFood = False

        for ant in state.inventories[playerId].ants:
            if ant.type == WORKER:
                if ant.carrying:
                    antHasFood = True
                numWorkers += 1
            elif numDrones < 3 and ant.type == DRONE:
                # encourage building 3 drones
                eval += .1 * EVAL_NUM_ANTS
                numDrones += 1
            playerHealthSum += ant.health

        # encourage building 2 workers
        if numWorkers == 1:
            eval += .075 * EVAL_NUM_ANTS
        elif numWorkers > 1:
            eval += .1

        if numDrones > 0:
            eval += .005 * EVAL_NUM_ANTS
        if numDrones <= 8:
            eval += numDrones * .02 * EVAL_NUM_ANTS
        else:
            eval += .4 * EVAL_NUM_ANTS

        # prevent overbuilding
        if numWorkers > 2 or numDrones > 0:
            return .1

        # encourage killing enemies
        numEnemies = len(state.inventories[enemyId].ants)
        if numEnemies > 8:
            numEnemies = 8
        eval -= numEnemies * .0625 * EVAL_NUM_ANTS

        # PART 2: health of ants

        # find sum of ants' HP for each player
        enemyHealthSum = 0
        for ant in state.inventories[enemyId].ants:
            enemyHealthSum += ant.health
        if playerHealthSum > 20:
            playerHealthSum = 20
        if enemyHealthSum > 20:
            enemyHealthSum = 20
        eval += playerHealthSum * .025 * EVAL_HEALTH
        eval -= enemyHealthSum * .025 * EVAL_HEALTH

        # PART 3: food delivery

        # encourage workers to drop off food
        if playerFoodCount > 1:
            eval += .5 * EVAL_FOOD
        elif playerFoodCount == 1:
            eval += .25 * EVAL_FOOD
        eval -= (enemyFoodCount / 10.0) * .5 * EVAL_FOOD

        # PART 4: carrying food

        # encourage workers to pick up food
        if antHasFood:
            eval += EVAL_CARRYING / 2
        else:
            eval -= EVAL_CARRYING / 2

        # PART 5: positions

        for ant in state.inventories[playerId].ants:
            # generally keep worker ants in friendly territory
            # if ant.type == WORKER and ant.coords[1] <= 3:
            #     eval += .05 * EVAL_POSITION
            # encourage drones to move toward enemy anthill
            if ant.type == WORKER:
                stepsToHill = abs(ant.coords[0] - enemyAnthill.coords[0]) \
                        + abs(ant.coords[1] - enemyAnthill.coords[1])
                eval += (1 - stepsToHill/20) * .05 * EVAL_POSITION
            # keep queen at edge of map
            elif ant.type == QUEEN:
                if ant.coords[1] > 0:
                    eval -= .1 * EVAL_POSITION

        # PART 6: keep queen off buildings and food

        for constr in state.inventories[playerId].constrs:
            if ((playerQueen.coords == constr.coords) and (constr.type == ANTHILL
                        or constr.type == TUNNEL or constr.type == FOOD)):
                eval -= EVAL_QUEEN_ON_BUILDING
                continue

        # PART 7: encourage sitting on enemy anthill

        for ant in state.inventories[playerId].ants:
            if ant.coords == enemyAnthill.coords:
                eval += EVAL_PARK_ENEMY_ANTHILL
                continue

        return round(eval, 8)

    ##
    #evaluateNodeList
    #
    #Description: finds the evaluation score of a list of nodes
    #
    #Parameters:
    #       nodeList - a list of nodes to be evaluated
    #
    #Return:
    #       a float representing the score of the list of nodes
    ##
    def evaluateNodeList(self, nodeList):
        bestScore = 0.0
        for node in nodeList:
            if bestScore < node.evalu:
                bestScore = node.evalu
        return bestScore

    ##
    #findBestMove
    #
    #Description: from a the current state, analyze all moves and return the best possible
    #       move given the current state, looking ahead a pre-defined (DEPTH_LIMIT) number of
    #       moves to analyze possible future moves.
    #
    #Parameters:
    #       currentState - the current state of the board
    #       playerId - the id of the player analyzing the move
    #       currentDepth - the depth of the
    #
    #Return:
    #       the best possible Move found
    #
    ##
    def findBestMove(self, currentState, playerId, currentDepth):
        moves = listAllLegalMoves(currentState)
        currentNode = self.Node(None, currentState, None, None)
        moveNodes = []

        # create a node for every possible move from current node
        for move in moves:
            hypotheticalState = self.getFutureState(currentState, move) #store the hypothetical node
            evalu = self.evaluateState(hypotheticalState)
            moveNodes.append(self.Node(move, hypotheticalState, currentNode, evalu))

        if len(moveNodes) > 3:
            moveNodes.sort(key=lambda x: x.evalu, reverse=True)
            topNodes = moveNodes[:3]
        else:
            topNodes = moveNodes


        # if we aren't at the depth limit, traverse subnodes
        if currentDepth < DEPTH_LIMIT:
            # recurse through subnode trees and update subnode evaluations
            for node in topNodes:
                if node.evalu < 1.0:
                    futureState = self.getFutureState(currentState, node.move)
                    node.evalu = self.findBestMove(futureState, playerId, currentDepth + 1)


        # Base Case: using initial scores provided by the hueristic eval function...

        # if we are returning to a recursive call, simply return the best subnode score
        if currentDepth > 0:
            # return the best score at the current depth
            return self.evaluateNodeList(topNodes)

        # otherwise, pick a move to return

        bestNodes = []

        # find best moves (resolve ties by picking randomly)
        bestRating = 0.0
        for node in topNodes:
            if node.evalu > bestRating:
                bestRating = node.evalu
        for node in topNodes:
            if node.evalu == bestRating:
                bestNodes.append(node)
        # return a random selection from winners to resolve ties
        returnNode = bestNodes[random.randint(0, len(bestNodes) - 1)]

        return returnNode.move

    ##
    #Node
    #Description: an inner class used for a representation for a 'node' in our search tree.
    ##
    class Node(object):
        ##
        #__init__
        #Description: Creates a new Construction. Only ever called by subclasses.
        #
        #Parameters:
        #   inputMove - The move that the node would take from its parent
        #   inputFutureState - the state that would be reached by the node's move
        #   inputParent - a reference to the parent node
        ##
        def __init__(self, inputMove, inputFutureState, inputParent, evalState):
            self.move = inputMove
            self.futureState = inputFutureState
            self.evalu = evalState
            self.parent = inputParent


############################
### UNIT TESTING SECTION ###
############################

aiPlayer = AIPlayer(PLAYER_ONE)

### Test 1 ###
# Build a Drone

# build a test board
antCarrying = Ant((3, 3), WORKER, PLAYER_ONE)
antCarrying.carrying = True
inventory1 = Inventory(PLAYER_ONE,
                       [
                           Ant((4, 2), WORKER, PLAYER_ONE),
                           antCarrying,
                           Ant((7, 0), SOLDIER, PLAYER_ONE),
                           Ant((0, 0), QUEEN, PLAYER_ONE)
                       ], [
                           Building((2, 2), ANTHILL, PLAYER_ONE),
                           Building((3, 4), TUNNEL, PLAYER_ONE)
                       ], 1)
inventory2 = Inventory(PLAYER_TWO,
                       [
                           Ant((0, 9), SOLDIER, PLAYER_TWO),
                           Ant((1, 9), WORKER, PLAYER_TWO),
                           Ant((7, 2), WORKER, PLAYER_TWO),
                           Ant((9, 9), QUEEN, PLAYER_TWO)
                       ], [
                           Building((8, 8), ANTHILL, PLAYER_TWO),
                           Building((2, 8), TUNNEL, PLAYER_TWO)
                       ], 1)
inventory3 = Inventory(NEUTRAL, [],
                       [
                           Construction((9, 6), GRASS),
                           Construction((4, 0), FOOD)
                       ], 0)
test1InitState = GameState(None, (inventory1, inventory2, inventory3), PLAY_PHASE, PLAYER_ONE)

# evaluate the initial board
eval1 = aiPlayer.evaluateState(test1InitState)

testMoveBuildDrone = Move(BUILD, [(8, 3)], DRONE)
futureState = aiPlayer.getFutureState(test1InitState, testMoveBuildDrone)

# verify that the future state is correct
droneExists = False
for ant in futureState.inventories[PLAYER_ONE].ants:
    if ant.type == DRONE and ant.coords == (8, 3):
        droneExists = True
        continue

if droneExists:
    eval2 = aiPlayer.evaluateState(futureState)
    if eval2 == .605:
        print "Unit Test #1 Passed"
    else:
        print "Unit Test #1 FAILED"
else:
    print "Unit Test #1 FAILED"

### Test 2 ###
# Delivering Food to Anthill

testMoveDeliverFood = Move(MOVE_ANT, [(3, 3), (2, 3), (2, 2)], None)
futureState = aiPlayer.getFutureState(test1InitState, testMoveDeliverFood)

# verify that the future state is correct
antDelivered = False
for ant in futureState.inventories[PLAYER_ONE].ants:
    if ant.carrying == False and ant.coords == (2, 2):
        antDelivered = True
        continue

if antDelivered and futureState.inventories[PLAYER_ONE].foodCount == 2:
    eval2 = aiPlayer.evaluateState(futureState)
    if eval2 == .6:
        print "Unit Test #2 Passed"
    else:
        print "Unit Test #2 FAILED"
else:
    print "Unit Test #2 FAILED"

### Test 3 ###
# Moving and Delivering a Killing Attack

testMoveKill = Move(MOVE_ANT, [(7, 0), (7, 1)], None)
futureState = aiPlayer.getFutureState(test1InitState, testMoveKill)

# verify that the future state is correct
antKilled = True
for ant in futureState.inventories[PLAYER_TWO].ants:
    if ant.coords == (7, 2):
        antKilled = False
    continue

if antKilled:
    eval2 = aiPlayer.evaluateState(futureState)
    if eval2 == .6125:
        print "Unit Test #3 Passed"
    else:
        print "Unit Test #3 FAILED"
else:
    print "Unit Test #3 FAILED"

### Test 4 ###
# Moving and Delivering a Non-Fatal Attack

testMoveAttack = Move(MOVE_ANT, [(4, 2), (5, 2), (6, 2)], None)
futureState = aiPlayer.getFutureState(test1InitState, testMoveAttack)

# verify that the future state is correct
attackWorked = False
for ant in futureState.inventories[PLAYER_TWO].ants:
    if ant.coords == (7, 2) and ant.health == 1:
        attackWorked = True
        continue

if attackWorked:
    eval2 = aiPlayer.evaluateState(futureState)
    if eval2 == .5925:
        print "Unit Test #4 Passed"
    else:
        print "Unit Test #4 FAILED"
else:
    print "Unit Test #4 FAILED"

### Test 5 ###
# Collect Food

for ant in test1InitState.inventories[PLAYER_ONE].ants:
    if ant.carrying:
        ant.carrying = False
eval1 = aiPlayer.evaluateState(test1InitState)

testMoveCollectFood = Move(MOVE_ANT, [(4, 2), (4, 1), (4, 0)], None)
futureState = aiPlayer.getFutureState(test1InitState, testMoveCollectFood)

# verify that the future state is correct
foodCollected = False
for ant in futureState.inventories[PLAYER_ONE].ants:
    if ant.coords == (4, 0) and ant.carrying:
        foodCollected = True
        continue

if foodCollected:
    eval2 = aiPlayer.evaluateState(futureState)
    if eval2 == .585:
        print "Unit Test #5 Passed"
    else:
        print "Unit Test #5 FAILED"
else:
    print "Unit Test #5 FAILED"