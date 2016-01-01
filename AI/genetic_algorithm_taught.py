
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
from decimal import *

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
        # Sleepy_Gary - Beth's husband for 15 years and Gerry's best friend
        super(AIPlayer,self).__init__(inputPlayerId, "Silly Sleepy Gary")
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
    #Description: Gets the next move from the Player.
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    #Return: The Move to be made
    ##
    def getMove(self, currentState):
        #if no moves have been made
        if self.noMoves == 0:
            self.noMoves = 1
            asciiPrintState(currentState)
        moves = {}

        for move in listAllLegalMoves(currentState):
            hypotheticalState = self.hypotheticalMove(currentState, move)
            rating = self.evaluateMove(hypotheticalState)

            if not rating in moves:
                moves[rating] = [move]
            else:
                moves[rating].append(move)

        # randomly select from the best moves
        bestMoves = moves[max(moves.keys())]
        move = bestMoves[random.randint(0, len(bestMoves) - 1)]
        hypotheticalState = self.hypotheticalMove(currentState, move)
        self.getPlayerScore(hypotheticalState, self.playerId)

        return move

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
