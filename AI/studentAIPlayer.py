# -*- coding: latin-1 -*-

##
#
# Homework 5
#
# Author(s): Alex Bowns, Joel Simard
#
import random, time
import sys
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Construction import Construction
from Ant import UNIT_STATS
from Ant import Ant
from Move import Move
from GameState import addCoords
from AIPlayerUtils import *
from decimal import *

# representation of inf
INFINITY = 9999
                
# establishing weights for the weighted linear equation
queenSafetyWeight = 0.3

# "max" values for determining how good a state is
maxNumAnts = 98.0 # 100 square minus 2 queens
maxDist = 18.0

# a representation of a 'node' in the search tree
treeNode = {
    # the Move that would be taken in the given state from the parent node
    "move"              : None,
    # the state that would be reached by taking the above move
    "potential_state"   : None,
    # an evaluation of the potential_state
    "state_value"       : 0.0,
    # a reference to the parent node
    "parent_node"       : None
}

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
        # a depth limit for the search algorithm
        self.maxDepth = 1
        # Sleepy_Gary - Beth's husband for 15 years and Gerry's best friend
        super(AIPlayer,self).__init__(inputPlayerId, "Sleepy Gary")
        #a list of lists to store our current population of genes
        self.genePopulationList = []
        #a list to store the fitness of each gene in the current population
        self.fitnessList = []
        #a globalized index to which gene/fitness we're evaluating
        self.index = 0
        #a variable used to decide whether the population is being evaluated the first time or not
        self.firstTime = 0
        #determine the population size
        self.populationSize = 10
        #reference to the best during the genetic process
        self.bestGene = None
        #reference to the fitness score of the best gene
        self.bestFitness = 0
        #reference for asciiPrintState '0' means no moves have happened
        self.noMoves = 0

        
    ##
    # vectorDistance
    # Description: Given two cartesian coordinates, determines the 
    #   manhattan distance between them (assuming all moves cost 1)
    #
    # Parameters:
    #   self - The object pointer
    #   pos1 - The first position
    #   pos2 - The second position
    #
    # Return: The manhattan distance
    #
    def vectorDistance(self, pos1, pos2):
        return (abs(pos1[0] - pos2[0]) +
                    abs(pos1[1] - pos2[1]))
                    
    
    ##
    # distClosestAnt
    # Description: Determines the distance between a cartesian coordinate
    #   and the coordinates of the enemy ant closest to it.
    #
    # Parameters:
    #   self - The object pointer
    #   currentState - The state to analyze
    #   initialCoords - The positition to check enemy ant distances from
    #
    # Return: The minimum distance between initialCoords and the closest
    #           enemy ant.
    #
    def distClosestAnt(self, currentState, initialCoords):
        # get a list of the enemy player's ants
        closestAntDist = 999
        for ant in currentState.inventories[(currentState.whoseTurn+1)%2].ants:
            tempAntDist = self.vectorDistance(ant.coords, initialCoords)
            if tempAntDist < closestAntDist:
                closestAntDist = tempAntDist
        return closestAntDist
    
    
    ##
    # evaluateNodes
    # Description: The evaluateNodes method evaluates a list of nodes
    # and determines their overall evaluation score.
    #
    # Parameters:
    #   self - The object pointer
    #   nodes - The list of nodes to evaluate
    #
    # Return: An overall evaluation score of the list of nodes
    #
    def evaluateNodes(self, nodes):
        # holds the greatest state_value in the list of nodes
        bestValue = 0.0
        # look through the nodes and find the greatest state_value
        for node in nodes:
            if node["state_value"] > bestValue:
                bestValue = node["state_value"]
        # return the greatest state_value
        return bestValue

        
    ##
    # alpha_beta_search
    # Description: use minimax with alpha beta pruning to determine what move to make
    #
    # Parameters:
    #   self - the object pointer
    #   node - the initial node, before any moves are explored
    #
    # Returns: the move which benefits the opposing player the least.
    #
    ##
    def alpha_beta_search(self, node):
        bestNode = self.max_value(node, -INFINITY, INFINITY, 0)
        while bestNode["parent_node"]["parent_node"] is not None:
            bestNode = bestNode["parent_node"]
        return bestNode["move"]

    
    ##
    # createNode
    # Description: Creates a node with values set based on parameters
    #
    # Parameters:
    #   self - The object pointer
    #   move - The move that leads to the resultingState
    #   resultingState - The state that results from making the move
    #   parent - The parent node of the node being created
    #
    # Returns: A new node with the values initialized using the parameters
    #
    def createNode(self, move, resultingState, parent):
        # Create a new node using treeNode as a model
        newNode = treeNode.copy()
        # set the move
        newNode["move"] = move
        # set the state that results from making the move
        newNode["potential_state"] = resultingState
        # set the value of the resulting state
        newNode["state_value"] = self.evaluateState(resultingState)
        # store a reference to the parent of this node
        newNode["parent_node"] = parent
        return newNode


    ##
    # max_value
    # Description: returns the best move our player can make from the current state
    #
    # Parameters:
    #   self - the object pointer
    #   node - the current node, before any moves are explored
    #   alpha - the alpha value, the value of our best move
    #   beta - the value of the opponent's best move
    #   currentDepth - the current depth of the node from the initial node
    #
    # Returns: the move which benefits the opposing player the least (alpha).
    #
    def max_value(self, node, alpha, beta, currentDepth):
        # base case, maxDepth reached, return the value of the currentState
        if currentDepth == self.maxDepth:
            return node
        state = node["potential_state"]
        v = -INFINITY

        # holds a list of nodes reachable from the currentState
        nodeList = []
        # loop through all legal moves for the currentState
        for move in listAllLegalMoves(state):
            # don't bother doing any move evaluations for the queen
            # unless we need to build a worker (she is in the way!)
            if move.moveType == MOVE_ANT:
                initialCoords = move.coordList[0]
                if ((getAntAt(state, initialCoords).type == QUEEN) and 
                    len(state.inventories[state.whoseTurn].ants) >= 2):
                        continue
            if move.moveType == BUILD:
                # hacky way to speed up the code by forcing building workers
                # over any other ant
                if move.buildType != WORKER:
                    continue
            # get the state that would result if the move is made
            resultingState = self.processMove(state, move)
            #create a newNode for the resulting state
            newNode = self.createNode(move, resultingState, node)
            # if a goal state has been found, stop evaluating other branches
            if newNode["state_value"] == 1.0:
                #we have a goal state, no alpha_beta evaluation is needed
                return newNode
            nodeList.append(newNode)

        #sort nodes from greatest to least
        sortedNodeList = sorted(nodeList, key=lambda k: k['state_value'], reverse=True)
        
        # throw away the last half of the list to minimize the number of nodes
        sortedNodeList = sortedNodeList[:(len(sortedNodeList)+1)/2]
        
        #holds a reference to the current best node to move to
        bestValNode = None
                
        #if it is our players turn
        if (self.playerId == state.whoseTurn):
            for tempNode in sortedNodeList:
                maxValNode = self.max_value(tempNode, alpha, beta, currentDepth+1)
                #if it's our turn and we're in max_value, stay in max_value
                if v < maxValNode["state_value"]:
                        bestValNode = maxValNode
                        v = maxValNode["state_value"]
                if v >= beta:
                    return maxValNode
                alpha = max(alpha, v)
        #else it is the opponents player turn
        else:
            sortedNodeList = sorted(nodeList, key=lambda k: k['state_value'])
            for tempNode in sortedNodeList:
                maxValNode = self.min_value(tempNode, alpha, beta, currentDepth+1)
                #if it's opponent's turn and they're in max_value, to toggle to min_value
                if v < maxValNode["state_value"]:
                       bestValNode = maxValNode
                       v = maxValNode["state_value"]
                if v >= beta:
                    return maxValNode
                alpha = max(alpha, v)

        return bestValNode


    ##
    # min_value
    # Description: returns the best move our opponent can make from the current state
    #
    # Parameters:
    #   self - the object pointer
    #   node - the current node, before any moves are explored
    #   alpha - the alpha value, the value of our best move
    #   beta - the value of the opponent's best move
    #   currentDepth - the current depth of the node from the initial node
    #
    # Returns: the move which benefits the opposing player the least (alpha).
    #
    def min_value(self, node, alpha, beta, currentDepth):
        # base case, maxDepth reached, return the value of the currentState
        if currentDepth == self.maxDepth:
            return node
        state = node["potential_state"]
        v = INFINITY

        # holds a list of nodes reachable from the currentState
        nodeList = []
        # loop through all legal moves for the currentState
        for move in listAllLegalMoves(state):
            # don't bother doing any move evaluations for the queen
            # unless we need to build a worker (she is in the way!)
            if move.moveType == MOVE_ANT:
                initialCoords = move.coordList[0]
                if ((getAntAt(state, initialCoords).type == QUEEN) and 
                    len(state.inventories[state.whoseTurn].ants) >= 2):
                        continue
            if move.moveType == BUILD:
                # hacky way to speed up the code by forcing building workers
                # over any other ant
                if move.buildType != WORKER:
                    continue
            # get the state that would result if the move is made
            resultingState = self.processMove(state, move)
            #create a newNode for the resulting state
            newNode = self.createNode(move, resultingState, node)
            # if a goal state has been found, stop evaluating other branches
            if newNode["state_value"] == 0.0:
                #we have a goal state, no alpha_beta evaluation is needed
                return newNode
            nodeList.append(newNode)
            
        #sort nodes from least to greatest
        sortedNodeList = sorted(nodeList, key=lambda k: k['state_value'])
        
        # throw away the last half of the list to minimize the number of nodes
        sortedNodeList = sortedNodeList[:(len(sortedNodeList)+1)/2]
        
        #holds a reference to the current best node to move to
        bestValNode = None
        
        #if it is our players turn
        if (self.playerId == state.whoseTurn):
            for tempNode in sortedNodeList:
                minValNode = self.max_value(tempNode, alpha, beta, currentDepth+1)
                #if it's our turn and we're in max_value, stay in max_value
                if v > minValNode["state_value"]:
                        bestValNode = minValNode
                        v = minValNode["state_value"]
                if v <= alpha:
                    return minValNode
                beta = min(beta, v)
        #else it is the opponents player turn
        else:
            for tempNode in sortedNodeList:
                minValNode = self.min_value(tempNode, alpha, beta, currentDepth+1)
                #if it's opponent's turn and they're in max_value, to toggle to min_value
                if v > minValNode["state_value"]:
                       bestValNode = minValNode
                       v = minValNode["state_value"]
                if v <= alpha:
                    return minValNode
                beta = min(beta, v)

        return bestValNode
    
    
    ##
    # processMove
    # Description: The processMove method looks at the current state
    # of the game and returns a copy of the state that results from
    # making the move
    #
    # Parameters:
    #   self - The object pointer
    #   currentState - The current state of the game
    #   move - The move which alters the state
    #
    # Return: The resulting state after move is made
    #
    def processMove(self, currentState, move):
        # create a copy of the state (this will be returned
        # after being modified to reflect the move)
        copyOfState = currentState.fastclone()
        
        # get a reference to the player's inventory
        playerInv = copyOfState.inventories[copyOfState.whoseTurn]
        # get a reference to the enemy player's inventory
        enemyInv = copyOfState.inventories[(copyOfState.whoseTurn+1) % 2]
        
        # player is building a constr or ant
        if move.moveType == BUILD:
            # building a constr
            if move.buildType < 0:  
                playerInv.foodCount -= CONSTR_STATS[move.buildType][BUILD_COST]
                playerInv.constrs.append(Construction(move.coordList[0], move.buildType))
            # building an ant
            else: 
                playerInv.foodCount -= UNIT_STATS[move.buildType][COST]
                playerInv.ants.append(Ant(move.coordList[0], move.buildType, copyOfState.whoseTurn))                
        # player is moving an ant
        elif move.moveType == MOVE_ANT:
            # get a reference to the ant
            ant = getAntAt(copyOfState, move.coordList[0])
            # update the ant's location after the move
            ant.coords = move.coordList[-1]
            
            # get a reference to a potential constr at the destination coords
            constr = getConstrAt(copyOfState, move.coordList[-1])
            # check to see if the ant is on a food or tunnel or hill and act accordingly
            if constr:
                # we only care about workers
                if ant.type == WORKER:
                    # if dest is food and can carry, pick up food
                    if constr.type == FOOD:
                        if not ant.carrying:
                            ant.carrying = True
                    # if dest is tunnel or hill and ant is carrying food, ditch it
                    elif constr.type == TUNNEL or constr.type == ANTHILL:
                        if ant.carrying:
                            ant.carrying = False
                            playerInv.foodCount += 1
            # get a list of the coordinates of the enemy's ants                 
            enemyAntCoords = [enemyAnt.coords for enemyAnt in enemyInv.ants]
            # contains the coordinates of ants that the 'moving' ant can attack
            validAttacks = []
            # go through the list of enemy ant locations and check if 
            # we can attack that spot and if so add it to a list of
            # valid attacks (one of which will be chosen at random)
            for coord in enemyAntCoords:
                #pythagoras would be proud
                if UNIT_STATS[ant.type][RANGE] ** 2 >= abs(ant.coords[0] - coord[0]) ** 2 + abs(ant.coords[1] - coord[1]) ** 2:
                    validAttacks.append(coord)
            # if we can attack, pick a random attack and do it
            if validAttacks:
                enemyAnt = getAntAt(copyOfState, random.choice(validAttacks))
                attackStrength = UNIT_STATS[ant.type][ATTACK]
                if enemyAnt.health <= attackStrength:
                    # just to be safe, set the health to 0
                    enemyAnt.health = 0
                    # remove the enemy ant from their inventory (He's dead Jim!)
                    enemyInv.ants.remove(enemyAnt)
                else:
                    # lower the enemy ant's health because they were attacked
                    enemyAnt.health -= attackStrength
        # move ends the player's turn
        elif move.moveType == END:
            # toggle between PLAYER_ONE (0) and PLAYER_TWO (1)
            copyOfState.whoseTurn += 1
            copyOfState.whoseTurn %= 2
        
        # return a copy of the original state, but reflects the move
        return copyOfState


    ##
    # evaluateState
    # Description: The evaluateState method looks at a state and
    # assigns a value to the state based on how well the game is
    # going for the current player
    #
    # Parameters:
    #   self - The object pointer
    #   currentState - The state to evaluate
    #
    # Return: The value of the state on a scale of 0.0 to 1.0
    # where 0.0 is a loss and 1.0 is a victory and 0.5 is neutral
    # (neither winning nor losing)
    #
    # Direct win/losses are either a technical victory or regicide
    #
    def evaluateState(self, currentState):        
        # get a reference to the player's inventory
        playerInv = currentState.inventories[currentState.whoseTurn]
        # get a reference to the enemy player's inventory
        enemyInv = currentState.inventories[(currentState.whoseTurn+1) % 2]
        # get a reference to the enemy's queen
        enemyQueen = enemyInv.getQueen()
        
        # game over (lost) if player does not have a queen
        #               or if enemy player has 11 or more food
        if playerInv.getQueen() is None or enemyInv.foodCount >= 11:
            return 0.0
        # game over (win) if enemy player does not have a queen
        #              or if player has 11 or more food
        if enemyQueen is None or playerInv.foodCount >= 11:
            return 1.0
        
        # initial state value is neutral ( no player is winning or losing )
        valueOfState = 0.5        
            
        # hurting the enemy queen is a very good state to be in
        valueOfState += 0.025 * (UNIT_STATS[QUEEN][HEALTH] - enemyQueen.health)
        
        # keeps track of the number of ants the player has besides the queen
        numNonQueenAnts = 0   
        enemyDistFromQueen = maxDist         
        
        # loop through the player's ants and handle rewards or punishments
        # based on whether they are workers or attackers
        for ant in playerInv.ants:
            if ant.type == QUEEN:
                enemyDistFromQueen = self.distClosestAnt(currentState, ant.coords)
                queenSafety = enemyDistFromQueen / maxDist
                valueOfState += queenSafety * queenSafetyWeight
            else:
                valueOfState += 0.01
                numNonQueenAnts += 1
                # Punish the AI less and less as its ants approach the enemy's queen
                valueOfState -= 0.005 * self.vectorDistance(ant.coords, enemyQueen.coords)
                            
        # ensure that 0.0 is a loss and 1.0 is a win ONLY
        if valueOfState < 0.0:
            valueOfState = 0.001 + (valueOfState * 0.0001)
        if valueOfState > 1.0:
            valueOfState =  0.999
            
        # return the value of the currentState
        # Value if our turn, otherwise 1-value if opponents turn
        # Doing 1-value is the equivalent of looking at the min value
        # since it is the best move for the opponent, and therefore the worst move
        # for our AI
        if currentState.whoseTurn == self.playerId:
            return valueOfState
        return 1-valueOfState


    ##
    #isGeneInvalid
    #Description: make sure a gene has legal locations for each individual game state
    #
    #Parameters:
    #   currentState - a reference to the current game state
    #   geneChunk - the chunk of gene we are checking (to make sure it has valid food placement coords)
    #
    #Return: two valid coordinates for food placement.
    #
    def isGeneInvalid(self, currentState, geneChunk):
        # get a reference to the enemy inventory
        constructions = getConstrList(currentState, None, (GRASS,ANTHILL,TUNNEL,FOOD))
        #make sure the enemy food placements are legit for first food coord
        myCoord = None
        x = geneChunk[0][0]
        y = geneChunk[0][1]
        while myCoord == None:
            for z in range(0, len(constructions)):
                #if we have an invalid coord, replace it
                if (x, y) == constructions[z].coords:
                     x = random.randint(0, 9)
                     y = random.randint(6, 9)
                     myCoord = None
                     break
            myCoord = (x, y)
        #same for the second food coord
        myCoord1 = None
        x = geneChunk[1][0]
        y = geneChunk[1][1]
        while myCoord1 == None:
            for z in range(0, len(constructions)):
                #if we have an invalid coord, replace it
                if (x, y) == constructions[z].coords:
                     x = random.randint(0, 9)
                     y = random.randint(6, 9)
                     myCoord1 = None
                     break
            myCoord1 = (x, y)
        #now return two valid food coordinates
        return [myCoord, myCoord1]


    ##
    #initPopulation
    #Description: generates a population of genes with random values, reset the fitness list to default values,
    # which are based off of a criteria of fitness scores.
    #
    # Parameters:
    #   currentState - the state of the board
    #   childrenList - a list of the child genes to be analyzed
    #   childFitList - a list of the fitness associated to each child gene
    #
    # Return: nothing, we have updated our global variables within this function
    #
    def initPopulation(self, currentState, childrenList, childFitList):
        #reset the genePopulationList
        self.genePopulationList = []
        #reset the fitnessList
        self.fitnessList = []

        #index to know how many genes to randomly produce
        idx = 0

        #if the population has children genes to consider, add those child genes to the start of the population
        if childrenList is not None:
            self.genePopulationList = childrenList
            self.fitnessList = childFitList
            idx = len(self.genePopulationList)

        #initialize as many new genes as the popSize wants, then append each one to the genePopulation list
        for i in range(idx, self.populationSize):
            #a 'gene' is a list of coordinates, containing...
            # [(0)anthill, (1)tunnel, (2)grass...,(10)grass, (11)food, (12)food]
            gene = []
            #for each coordinate in a gene, pick a new unused coordinate
            for j in range(0, 11):
                  myCoord = None
                  yMin = 0
                  yMax = 3
                  #select a random coordinate, see if the coordinate is occupied, if it is select a new random coordinate
                  while myCoord == None:
                      #Choose any x location
                      x = random.randrange(0, 9)
                      #Choose any y location on your side of the board
                      y = random.randrange(yMin, yMax)
                      #Set the move if this space is empty
                      if currentState.board[x][y].constr == None:
                         myCoord = (x, y)
                         #make the space non-empty, so we don't create another gene with the same coord
                         currentState.board[x][y].constr = True
                      else:
                         #try a new random coordinate
                         myCoord = None
                  #append the coordinate to the gene list
                  gene.append(myCoord)
            #coordinates for the enemy food
            for j in range(0, 2):
                myCoord = None
                yMin = 6
                yMax = 9
                #select a random coordinate, see if the coordinate is occupied, if it is select a new random coordinate
                while myCoord == None:
                  #Choose any x location
                  x = random.randrange(0, 9)
                  #Choose any y location on your side of the board
                  y = random.randrange(yMin, yMax)
                  #Set the move if this space is empty
                  if currentState.board[x][y].constr == None:
                     myCoord = (x, y)
                     #make the space non-empty, so we don't create another gene with the same coord
                     currentState.board[x][y].constr = True
                  else:
                     #try a new random coordinate
                     myCoord = None
                #append the coordinate to the gene list
                gene.append(myCoord)

            #this gene is now finished, give a fitness value to the gene
            geneFitness = self.fitnessFunction(gene, currentState)
            #now append the gene to genePopulation and its fitness to the fitnessList
            self.genePopulationList.append(gene)
            self.fitnessList.append(geneFitness)
            #reset all board coordinates to be empty before moving to the next gene
            for cnt in range(0, 13):
                coord = gene[cnt]
                y2 = coord[1]
                x2 = coord[0]
                currentState.board[x2][y2].constr = None
        #don't need to return anything anymore
        return None


    ##
    #fitnessFunction
    #Description: a helper method to judge the default fitness of a specific gene.
    #
    # Parameters:
    #   gene - the gene to be analyzed on fitness
    #   currentState - the state of the game
    #
    # Return: the fitness score for the gene (a higher value means the gene is more 'fit')
    #
    def fitnessFunction(self, gene, currentState):
        #the fitnessScore will be returned, begins at 0 for starting score
        fitnessScore = 0

        #a reference variable for the anthill y coord
        referenceY = 0

        #part of fitness score is  based off of the anthill y coords
        anthillYCoord = gene[0][1]
        if anthillYCoord == 0:
           fitnessScore += 5
        elif anthillYCoord == 1:
            fitnessScore += 3
        elif anthillYCoord == 2:
            fitnessScore += 0
        else:
            referenceY = 1
            anthillYCoord -= 1

        #part of fitness score is based off of the enemy food distance to the enemy tunnel/anthill
        #find the two farthest locations
        bestFoodSpots = self.twoFarthestFoods(currentState)
        enemyFoodSet = gene[11:]
        goodFoodPlacement = set(bestFoodSpots).intersection((set(enemyFoodSet)))
        fitnessScore += len(goodFoodPlacement)*5

        #part of the fitness score considers if there are grass blockades in front of our own anthill
        #if anthill is in
        if referenceY == 1:
            #no need to look for grass
            pass
        else:
            #store all adjacent coodinates from the anthill
            adjCoords = listAdjacent(gene[0])
            grassCoords = gene[2:11]
            blockadeGrass = set(adjCoords).intersection(set(grassCoords))
            fitnessScore += len(blockadeGrass)

        #part of the fitness score considers the distance from our own anthill and tunnel (farther is better)
        constrDist = stepsToReach(currentState, gene[1], gene[0])
        if constrDist >= 5:
            fitnessScore += 2
        elif constrDist < 2:
            fitnessScore -= 1
        else:
            fitnessScore += 0

        #now return the default score
        return fitnessScore


    ##
    #twoFarthestFoods
    #Description: a helper method to see what foods are farthest away from the enemy anthill and tunnel
    #
    # Parameters:
    #   currentState - the state of the game
    #
    # Return: a tuple of the two farthest food coordinates
    #
    def twoFarthestFoods(self, currentState):
        buildings = getConstrList(currentState, PLAYER_ONE,(ANTHILL,TUNNEL))  #stuff on foe's side
        constructions = getConstrList(currentState, None, (GRASS,ANTHILL,TUNNEL,FOOD))
        buildingLocations = []
        constrLocations = []
        foodBestLocations = []
        for i in buildings: ## add ANTHILL and TUNNEL coordinates into a list
            buildingLocations.append(i.coords)
        for i in constructions: ## add GRASS, ANTHILL, TUNNEL, FOOD coordinates into a list
            constrLocations.append(i.coords)
            cost1 = 0 #movement cost to food1
            cost2 = 0 #movement cost to food2
            foodBestLocation = None
            secondBest = None
        for x in range(0,10):
            for y in range(6,10):
                average = None
                if (x, y) in constrLocations:
                    pass # if spot is not empty do nothing
                else: # measure the average distance from ANTHILL and TUNNEL and choose the farthest
                    distanceFromAnthill = stepsToReach(currentState, (x,y), buildingLocations[0])
                    distanceFromTunnel = stepsToReach(currentState, (x,y), buildingLocations[1])
                    average = (distanceFromAnthill + distanceFromTunnel)/2.0
                if(average > cost1):
                    cost1 = average
                    secondBest = foodBestLocation
                    foodBestLocation = (x,y)
                elif(average > cost2):
                    cost2 = average
                    secondBest = (x,y)
        return (foodBestLocation, secondBest)


    ##
    #randomSelection
    #Description: this helper method will randomly select a gene from a list of genes in the population,
    # a the probability for a gene being chosen is directly proportional to the fitness score a gene has
    #
    # Parameters: None
    #
    # Return: a gene from the list of genes
    #
    def randomSelection(self):
        #a list of 100, each index stores a reference to a gene from the population
        referenceList = []
        reference = None

        #to determine a percent probability of picking a specific gene:
        #divide the fitness of a gene by the sum of all the fitnesses
        geneSum = sum(self.fitnessList)
        for i in range(0, self.populationSize):
            getcontext().prec = 2
            geneProb = self.fitnessList[i]/Decimal(geneSum)
           # geneProb = Decimal(self.fitnessList[i]/geneSum)
            geneProb = geneProb * 100
            #now add the reference to the gene to the list
            for p in range(0, int(geneProb)):
                referenceList.append(i)

        #randomly select a gene reference from the reference list
        while reference == None:
            x = random.randint(0, 99)
            if len(referenceList) > x:
                reference = referenceList[x]
            else:
                continue

        #return the selected gene
        return self.genePopulationList[reference]


    ##
    #reproduce
    #Description: a helper method for reproducing, it takes two parent genes, slices them at the same index,
    #  and swaps slices from each parent to form two new children.
    #
    # Parameters:
    #   parent1 - the first parent gene
    #   parent2 - the second parent gene
    #
    # Return: a tuple (child1, child2)
    #
    def reproduce(self, parent1, parent2):
        n = len(parent1)
        c = random.randint(0, n-1)
        child1 = parent1[0:c] + parent2[c:n]
        child2 = parent2[0:c] + parent1[c:n]

        #randomly mutate child1
        for i in range(0,12):
            randInt = random.randrange(1, 20)
            #give a 5% chance for an element of the gene to mutate
            if randInt == 10:
                randIdx = random.randrange(0,12)
                #make sure the element is swapping with another element
                while randIdx == i:
                    randIdx = random.randrange(0,12)
                #swap the selected element and the random element in the gene
                tempTuple = child1[randIdx]
                child1[randIdx] = child1[i]
                child1[i] = tempTuple

        #randomly mutate child2
        for i in range(0,12):
            randInt = random.randrange(1, 20)
            #give a 5% chance for an element of the gene to mutate
            if randInt == 10:
                randIdx = random.randrange(0,12)
                #make sure the element is swapping with another element
                while randIdx == i:
                    randIdx = random.randrange(0,12)
                #swap the selected element and the random element in the gene
                tempTuple = child1[randIdx]
                child2[randIdx] = child1[i]
                child2[i] = tempTuple

        return (child1, child2)


    ##
    #getPlacement
    #Description: The getPlacement method corresponds to the
    #action taken on setup phase 1 and setup phase 2 of the game.
    #In setup phase 1, the AI player will be passed a copy of the
    #state as currentState which contains the board, accessed via
    #currentState.board. The player will then return a list of 10 tuple
    #coordinates (from their side of the board) that represent Locations
    #to place the anthill and 9 grass pieces. In setup phase 2, the player
    #will again be passed the state and needs to return a list of 2 tuple
    #coordinates (on their opponent?s side of the board) which represent
    #Locations to place the food sources. This is all that is necessary to
    #complete the setup phases.
    #
    #Parameters:
    #   currentState - The current state of the game at the time the Game is
    #       requesting a placement from the player.(GameState)
    #
    #Return: If setup phase 1: list of ten 2-tuples of ints -> [(x1,y1), (x2,y2),?,(x10,y10)]
    #       If setup phase 2: list of two 2-tuples of ints -> [(x1,y1), (x2,y2)]
    #
    def getPlacement(self, currentState):
        #for only the first phase of setup, check if it is needed to reinitialize a population
        if currentState.phase == SETUP_PHASE_1:
            #if the global index is 0, we need to reinitialize the population
            if self.index == 0:
                if self.firstTime == 0:
                    #initialize a new randomized genePopulation list and a fitness eval for each gene,
                    # no child genes to start with
                    self.initPopulation(currentState, None, None)
                    #initPopulationTuple = self.initPopulation(currentState, None, None)

                else: #child genes now included to the new population
                    #first select two parents to reproduce
                    parent1 = self.randomSelection()
                    parent2 = self.randomSelection()
                    #reproduce two children genes from the parent genes
                    kids = self.reproduce(parent1, parent2)
                    #create the extra parameters needed to initialize the new population
                    kidsPopList = list(kids)
                    kidsFitList = []
                    kidsFitList.append(self.fitnessFunction(kidsPopList[0], currentState))
                    kidsFitList.append(self.fitnessFunction(kidsPopList[1], currentState))
                    #initialize a new population, with the child genes as well
                    self.initPopulation(currentState, kidsPopList, kidsFitList)

            #play a game with the selected gene in the population
            gene = self.genePopulationList[self.index]
            #return the constructs on my side
            return gene[0:11]
        elif currentState.phase == SETUP_PHASE_2:   #stuff on opponent side
            gene = self.genePopulationList[self.index]
            #print "food placement: " + str(gene[11:13])
            enemyFoodGene = self.isGeneInvalid(currentState, gene[11:13])
            return enemyFoodGene
        else:
            return [(0, 0)]
            
    
    ##
    #getMove
    #Description: The getMove method corresponds to the play phase of the game 
    #and requests from the player a Move object. All types are symbolic 
    #constants which can be referred to in Constants.py. The move object has a 
    #field for type (moveType) as well as field for relevant coordinate 
    #information (coordList). If for instance the player wishes to move an ant, 
    #they simply return a Move object where the type field is the MOVE_ANT constant 
    #and the coordList contains a listing of valid locations starting with an Ant 
    #and containing only unoccupied spaces thereafter. A build is similar to a move 
    #except the type is set as BUILD, a buildType is given, and a single coordinate 
    #is in the list representing the build location. For an end turn, no coordinates 
    #are necessary, just set the type as END and return.
    #
    #Parameters:
    #   currentState - The current state of the game at the time the Game is 
    #       requesting a move from the player.(GameState)   
    #
    #Return: Move(moveType [int], coordList [list of 2-tuples of ints], buildType [int]
    #
    def getMove(self, currentState):
        #if no moves have been made
        if self.noMoves == 0:
            self.noMoves = 1
            asciiPrintState(currentState)

        # save our id
        self.playerId = currentState.whoseTurn
        #create the initial node to analyze
        initNode = self.createNode(None, currentState, None)
        return self.alpha_beta_search(initNode)

    
    ##
    #getAttack
    #Description: The getAttack method is called on the player whenever an ant completes 
    #a move and has a valid attack. It is assumed that an attack will always be made 
    #because there is no strategic advantage from withholding an attack. The AIPlayer 
    #is passed a copy of the state which again contains the board and also a clone of 
    #the attacking ant. The player is also passed a list of coordinate tuples which 
    #represent valid locations for attack. Hint: a random AI can simply return one of 
    #these coordinates for a valid attack. 
    #
    #Parameters:
    #   currentState - The current state of the game at the time the Game is requesting 
    #       a move from the player. (GameState)
    #   attackingAnt - A clone of the ant currently making the attack. (Ant)
    #   enemyLocation - A list of coordinate locations for valid attacks (i.e. 
    #       enemies within range) ([list of 2-tuples of ints])
    #
    #Return: A coordinate that matches one of the entries of enemyLocations. ((int,int))
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        #Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]
        
    ##
    #registerWin
    #Description: The last method, registerWin, is called when the game ends and simply 
    #indicates to the AI whether it has won or lost the game. This is to help with 
    #learning algorithms to develop more successful strategies.
    #
    #Parameters:
    #   hasWon - True if the player has won the game, False if the player lost. (Boolean)
    #
    def registerWin(self, hasWon):
        self.noMoves = 0
        self.firstTime = 1
        #if the ai gene won, give extra fitness to it
        if hasWon == True:
            self.fitnessList[self.index] += 5
        #keep a reference to the highest scoring gene
        if self.fitnessList[self.index] > self.bestFitness:
            self.bestFitness = self.fitnessList[self.index]
            self.bestGene = self.genePopulationList[self.index]
        #point towards the next gene in the population
        self.index += 1
        if self.index == self.populationSize:
            self.index = 0
            print "-----------Next Generation---------"
