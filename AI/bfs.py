import random
import math

from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from Ant import *
from AIPlayerUtils import *

##
# weight
#
# Description: Takes a raw value and determines where it lies along the weight list.
# It returns the value at that position in the weight list. If it lies between two
# values, then it returns the weighed average of the two weights. If it lies beyond the
# end of the list, it returns the last item in the list.
#
# Parameters:
#    raw        - The raw score which will be weighted (float or int)
#    weightList - The list to weigh the raw value with (list[float or int])
#
# Return:
#    The weighted value (float)
##
def weight(raw, weightList):
    if raw > len(weightList) - 1:
        return float(weightList[-1])
    elif int(raw) == raw:
        return float(weightList[int(raw)])
    else:
        bottom = weightList[int(math.floor(raw))] * (raw - math.floor(raw))
        top = weightList[int(math.ceil(raw))] * (raw - math.ceil(raw))
        return float(bottom + top) / 2.


# Preference for which ant to attack first if there is a choice
ATTACK_PREFERENCE = [QUEEN, SOLDIER, R_SOLDIER, DRONE, WORKER]

# Grading weight for different factors
# Each of these is a function which gets passed the raw score for that category and
# weights it.

# How much to weight our food count
FOODSCOREWEIGHT = lambda x: weight(x, [0, 1000, 1800, 2500, 2900, 3200, 3400, 3500, 33500, 63500, 93500])

# How much to weight the number of workers who reach their destination
REACHDESTWEIGHT = lambda x: 9 * x

# How much to weight the distance a worker is from a goal
WORKERLOCATIONWEIGHT = lambda x: -3 * x #Distance from goal is bad

# How much to weight soldiers being in the correct area
SOLDIERLOCATIONWEIGHT = lambda x: 6 * x

# How much to weight drones being in the correct area
DRONELOCATIONWEIGHT = lambda x: -5 * x #Distance from goal is bad

# How much to weight how many ants have moved this turn
MOVEDANTSSCOREWIEGHT = lambda x: x


# Depth limit for node searches
DEPTH_LIMIT = 0


#Grading weight for ant types count
#Queen, worker, drone, soldier, ranged soldier
antTypeGradingWeight = [
    lambda x: 0,                                       #QUEEN (never build a queen)
    lambda x: weight(x, [-100000, 100000, 120000, 0]), #WORKER
    lambda x: weight(x, [0, 900]),                     #DRONE
    lambda x: weight(x, [0, 1300, 2600]),              #SOLDIER
    lambda x: 0,                                       #RANGE SOLDIER (never build a range soldier)
    ]

##
#AIPlayer
#Description: The responsbility of this class is to interact with the game by
#deciding a valid move based on a given game state. This class has methods that
#will be implemented by students in Dr. Nuxoll's AI course.
#
#Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):

    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer,self).__init__(inputPlayerId, "INSERT COOL NAME HERE AI")

    ##
    #getPlacement
    #
    #Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    #Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    #Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        numToPlace = 0
        #implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:    #stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:   #stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
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
    #getAttack
    #Description: Gets the attack to be made from the Player
    #
    #Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        target = None
        for coords in enemyLocations:
            ant = getAntAt(currentState, coords)
            if (not target or
                ATTACK_PREFERENCE.index(ant.type) < ATTACK_PREFERENCE.index(target.type)
                ):
                target = ant

        return target.coords


    ##
    #hypotheticalMove
    #
    #Description: Determine what the agent's state would look like after a given move.
    #             We Will assume that all Move objects passed are valid.
    #
    #Parameters:
    #   state - A clone of the theoretical state given (GameState)
    #   move - a list of all move objects passed (Move)
    #
    #Returns:
    #   What the agent's state would be like after a given move.
    ##
    def hypotheticalMove(self, state, move):
        newState = state.fastclone()
        if move.moveType == END:
            return newState
        elif move.moveType == MOVE_ANT:
            ant = getAntAt(newState, move.coordList[0])
            ant.coords = move.coordList[-1]

            #check if ant is depositing food
            if ant.carrying:
                targets = getConstrList(newState, self.playerId, (ANTHILL, TUNNEL))
                if tuple(ant.coords) in (tuple(t.coords) for t in targets):
                    ant.carrying = False
                    newState.inventories[self.playerId].foodCount += 1

            #check if ant can attack
            targets = [] #coordinates of attackable ants
            range = UNIT_STATS[ant.type][RANGE]

            for ant in newState.inventories[1 - self.playerId].ants:
                dist = math.sqrt((ant.coords[0] - ant.coords[0]) ** 2 +
                                 (ant.coords[1] - ant.coords[1]) ** 2)
                if dist <= range:
                    #target is in range and may be attacked
                    targets.append(ant.coords)

            if targets:
                #Attack the ant chosen by the AI
                target = self.getAttack(newState, ant, targets)
                targetAnt = getAntAt(newState, target)
                targetAnt.health -= UNIT_STATS[ant.type][ATTACK]

                if targetAnt.health <= 0:
                    #Remove the dead ant
                    newState.inventories[1 - self.playerId].ants.remove(targetAnt)

        else: #Move type BUILD
            if move.buildType in (WORKER, DRONE, SOLDIER, R_SOLDIER):
                #Build ant on hill
                ant = Ant(move.coordList[0], move.buildType, self.playerId)
                newState.inventories[self.playerId].ants.append(ant)

                newState.inventories[self.playerId].foodCount -= UNIT_STATS[move.buildType][COST]
            else:
                #build new building
                building = Building(move.coordList[0], move.buildType, self.playerId)
                newState.inventories[self.playerId].constrs.append(building)

                newState.inventories[self.playerId].foodCount -= CONSTR_STATS[move.buildType][BUILD_COST]

        return newState


    ## scoreAnts - Create a score for the list of ants given
    def scoreAnts(self, ants, type):
        count = 0.

        for ant in ants:
            if ant.type == type:
                count += float(ant.health) / float(UNIT_STATS[ant.type][HEALTH])

        return antTypeGradingWeight[type](count)


    ##
    # getPlayerScore
    # Description: takes a state and player number and returns a number estimating that player's
    # score. Note, this score may be negative and have a very large magnitude (> 100000)
    # Parameters:
    #    hypotheticalState - The state to score
    #    playerNo          - The player number to determine the score for
    # Returns:
    #    A float representing that player's score.
    #
    def getPlayerScore(self, hypotheticalState, playerNo):

        #################################################################################
        #Score the ants we have based on number, type and health

        #get the number of ants on the board, and for certain types of ants
        antScore = 0
        for type in (WORKER, DRONE, SOLDIER, R_SOLDIER):
            score = self.scoreAnts(hypotheticalState.inventories[playerNo].ants, type)
            antScore += score


        #################################################################################
        #Score the food we have

        #get the food count from the move
        foodScore = hypotheticalState.inventories[playerNo].foodCount
        foodScore = FOODSCOREWEIGHT(foodScore)


        #################################################################################
        #Score the workers for getting to their goals

        ourBuildings = getConstrList(hypotheticalState, playerNo, (ANTHILL, TUNNEL))
        ourBuildingCoords = [tuple(b.coords) for b in ourBuildings]

        foods = getConstrList(hypotheticalState, None, (FOOD,))
        foodCoords = [tuple(f.coords) for f in foods]

        #get the total food which will be being carried at the end of this turn
        workerDestReached = 0
        for worker in getAntList(hypotheticalState, playerNo, (WORKER,)):
            if worker.carrying:
                goals = ourBuildingCoords
            else:
                goals = foodCoords

            if tuple(worker.coords) in goals:
                workerDestReached += 1

        workerDestScore = REACHDESTWEIGHT(workerDestReached)


        #################################################################################
        #Score the progress of workers towards their destinations

        #workers get bonus points for being closer to a goal (the distance will be weighted negatively)
        workerLocationScore = 0
        for worker in getAntList(hypotheticalState, playerNo, (WORKER,)):
            if worker.carrying:
                goals = ourBuildingCoords
            else:
                goals = foodCoords

            wc = worker.coords
            dist = min(abs(wc[0]-gc[0]) + abs(wc[1]-gc[1]) for gc in goals)

            workerLocationScore += dist

        # average this score
        if workerLocationScore:
            workerLocationScore /= len(getAntList(hypotheticalState, playerNo, (WORKER,)))

        workerLocationScore = WORKERLOCATIONWEIGHT(workerLocationScore)


        #################################################################################
        #Score the location of soldier ants

        #soldier ants get bonus points for being on the other side of the field
        soldierLocationScore = 0
        for soldier in getAntList(hypotheticalState, playerNo, (SOLDIER, )):
            if soldier.coords[1] > 6:
                soldierLocationScore += 1
            else:
                soldierLocationScore = soldier.coords[1] - 6

        # average this score
        if soldierLocationScore:
            soldierLocationScore /= len(getAntList(hypotheticalState, playerNo, (SOLDIER,)))

        soldierLocationScore = SOLDIERLOCATIONWEIGHT(soldierLocationScore)


        #################################################################################
        #Score the location of drone ants

        #drone ants are always to go towards the enemy hill
        droneLocationScore = 0
        enemyHill = getConstrList(hypotheticalState, 1 - playerNo, (ANTHILL,))[0]
        for drone in getAntList(hypotheticalState, playerNo, (DRONE,)):
            dist = (abs(drone.coords[0]-enemyHill.coords[0]) +
                    abs(drone.coords[1]-enemyHill.coords[1]))
            droneLocationScore += dist

        # average this score
        if droneLocationScore:
            droneLocationScore /= len(getAntList(hypotheticalState, playerNo, (DRONE,)))

        droneLocationScore = DRONELOCATIONWEIGHT(droneLocationScore)


        #################################################################################
        #Score every ant having moved

        #It is to our advantage to have every ant move every turn
        movedAnts = 0
        for ant in hypotheticalState.inventories[playerNo].ants:
            if ant.hasMoved:
                movedAnts += 1

        movedAntsScore = MOVEDANTSSCOREWIEGHT(movedAnts)

        return (antScore +
                foodScore +
                workerDestScore +
                workerLocationScore +
                soldierLocationScore +
                droneLocationScore +
                movedAntsScore)

    ##
    # hasWon
    # Description: Takes a GameState and a player number and returns if that player has won
    # Parameters:
    #    hypotheticalState - The state to test for victory
    #    playerNo          - What player to test victory for
    # Returns:
    #    True if the player has won else False.
    ##
    def hasWon(self, hypotheticalState, playerNo):

        #Check if enemy anthill has been captured
        for constr in hypotheticalState.inventories[1 - playerNo].constrs:
            if constr.type == ANTHILL and constr.captureHealth == 1:
                #This anthill will be destroyed if there is an opposing ant sitting on it
                for ant in hypotheticalState.inventories[playerNo].ants:
                    if tuple(ant.coords) == tuple(constr.coords):
                        return True
                break

        #Check if enemy queen is dead
        for ant in hypotheticalState.inventories[1 - playerNo].ants:
            if ant.type == QUEEN and ant.health == 0:
                return True

        #Check if we have 11 food
        if hypotheticalState.inventories[playerNo].foodCount >= 11:
            return True

        return False


    ##
    #evaluateMove
    #
    #Description: Examines a GameState and ranks how "good" that state is for the agent whose turn it is.
    #              A rating is given on the players state. 1.0 is if the agent has won, 0.0 if the enemy has won,
    #              any value > 0.5 means the agent is winning.
    #
    #Parameters:
    #   hypotheticalState - The state being considered by the AI for ranking.
    #
    #Return:
    #   The move rated as the "best"
    ##
    def evaluateMove(self, hypotheticalState):

        #Check if the game is over
        if self.hasWon(hypotheticalState, self.playerId):
            return 1.0
        elif self.hasWon(hypotheticalState, 1 - self.playerId):
            return 0.0

        playerScore = self.getPlayerScore(hypotheticalState, self.playerId)
        enemyScore = self.getPlayerScore(hypotheticalState, 1 - self.playerId)

        #Normalize the score to be between 0.0 and 1.0
        return (math.atan(playerScore - enemyScore) + math.pi/2) / math.pi

    ##
    #evaluateNodeList
    #
    #Description: finds the average evaluation score of a list of nodes
    #
    #Parameters:
    #       nodeList - a list of nodes to be evaluated
    #
    #Return:
    #       a float representing the average score of the nodes
    ##
    def evaluateNodeList(self, nodeList):
        sumScore = 0.0
        for node in nodeList:
            sumScore += node.evalu
        return sumScore/len(nodeList)

    ##
    #findBestMove
    #
    #Description: from a the current state node, analyze all nodes that expand from that move,
    # if not at the depth limit, then recurse and do the same thing for one of the expanded nodes.
    #
    #Parameters:
    #       currentState - the current state of the board
    #       playerId - the id of the player analyzing the move
    #       currentDepth - the depth of the
    #
    #Return:
    #
    ##
    def findBestMove(self, currentState, playerId, currentDepth):
        moves = listAllLegalMoves(currentState)
        currentNode = self.Node(None, currentState, None, None)
        moveNodes = []

        # create a node for every possible move from current node
        for move in moves:
            hypotheticalState = self.hypotheticalMove(currentState, move) #store the hypothetical node
            evalu = self.evaluateMove(hypotheticalState)
            moveNodes.append(self.Node(move, hypotheticalState, currentNode, evalu))

        # if we aren't at the depth limit, traverse subnodes
        if currentDepth < DEPTH_LIMIT:
            # recurse through subnode trees and update subnode evaluations
            for node in moveNodes:
                futureState = self.hypotheticalMove(currentState, node.move) #store the futureState node
                bestNode = self.findBestMove(futureState, playerId, currentDepth + 1)
                #evalu = self.evaluateMove(node.futureState)
                node.evalu = bestNode.evalu
                #node.evalu = bestNode


        #Base Case: use initial scores provided by the hueristic eval function, record the best node at the current depth
        bestNodeEvalu = self.evaluateNodeList(moveNodes)

        bestScore = 0.0
        bestNode = None
        for node in moveNodes:
            if node.evalu > bestScore:
                bestScore = node.evalu
                bestNode = node

        if currentDepth == 0:
            return bestNode.move
        else:
            return bestNode

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
            #self.evalu = AIPlayer.evaluateMove(self.futureState) #contain an evaluation of the state caused by the node
            self.parent = inputParent

##
# unitTest1
# Description: Tests the AIPlayer.hypotheticalMove method
# Returns:
#    False if anything is wrong else True
##
def unitTest1():
    board = [[Location((col, row)) for row in xrange(0,BOARD_LENGTH)] for col in xrange(0,BOARD_LENGTH)]
    p1Inventory = Inventory(PLAYER_ONE, [], [], 10)
    p2Inventory = Inventory(PLAYER_TWO, [], [], 0)
    neutralInventory = Inventory(NEUTRAL, [], [], 0)

    state = GameState(board, [p1Inventory, p2Inventory, neutralInventory], MENU_PHASE, PLAYER_ONE)

    #Add an ant to move
    ant = Ant((0,0), WORKER, 0)
    board[0][0].ant = ant
    p1Inventory.ants.append(ant)

    player = AIPlayer(0)
    newState = player.hypotheticalMove(state, Move(MOVE_ANT, ((0,0), (0,1), (0,2)), None))
    if tuple(newState.inventories[0].ants[0].coords) != (0, 2):
        print "didn't move ant"
        return False

    #test adding a building
    newState = player.hypotheticalMove(state, Move(BUILD, ((3,3),), TUNNEL))

    if len(newState.inventories[0].constrs) == 0:
        print "didn't create construction"
        return False

    if newState.inventories[0].constrs[0].type != TUNNEL:
        print "created wrong type of construction"
        return False

    if tuple(newState.inventories[0].constrs[0].coords) != (3, 3):
        print "created construction at wrong place"
        return False

    if newState.inventories[0].foodCount != 7:
        print "didn't subtract food cost"
        return False

    return True

if unitTest1():
    print "Unit Test 1 passed!"