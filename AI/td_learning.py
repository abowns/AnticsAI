# -*- coding: latin-1 -*-

##
# 
# Homework 8
#
# Author(s): Alex Bowns
#
import pickle
from Player import *
import os
from Constants import *
from Construction import CONSTR_STATS
from Construction import Construction
from Ant import UNIT_STATS
from Ant import Ant
from Move import Move
from GameState import addCoords
from AIPlayerUtils import *

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

# a representation of a consolidated game state
stateDictionary = {
    "loss": False,  # record 'true' if the player AI has lost
    "win": False,   # record 'true' if the player AI has won the game
    "FoodCount": None,     # record the player AI food count
    "PQueenHealth": None,   # the player queen health
    "EQueenHealth": None,   # the enemy queen health
    "AntToQueen": None,    # the distance of the closest player ant to the enemy queen
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
        # a tracker for if the match has just begun
        self.justBegun = True
        # a depth limit for the search algorithm
        self.maxDepth = 3
        # representation of inf
        self.INFINITY = 9999.0
        # a count of the game
        self.gamecount = 0
        # establishing weights for the weighted linear equation
        self.queenSafetyWeight = 0.3
        # the max distance an ant can be from the queen
        self.maxDist = 18.0
        # the discount factor
        self.gamma = .95
        # the learning rate
        self.alpha = .99
        # the rate at which the learning rate decreases per game
        self.decreaseRate = .99
        # the name of the file to print and load
        self.fileName = 'bowns16_evidence.pkl'
        # hold a dictionary of all consolidated states examined
        self.utilityList = {}
        # load the utility list if it already exists
        if os.path.isfile(self.fileName):
            self.utilityList = self.loadUtilityList()

        # Smitty Werbenjagermanjensen - He was number 1!!!!!!
        super(AIPlayer,self).__init__(inputPlayerId, "Smitty Werbenjagermanjensen")


    ##
    # consolidate
    # Description: This method will analyze essential components of the current game state:
    #               It judges the health of both players queen ants, win conditions per team,
    #               both player food counts, and the distances between a players queen and opponent ants.
    #               Then it will put all of that info into a dictionary representation of the state.
    #               There are 7,128 different consolidated states to assess.
    #
    # Parameters:
    #   currentState - the current state of the game
    #
    # Return:
    #   consolidatedState - a dictionary of the essentials of the
    #
    def consolidate(self, currentState):
        # basic info to help
        AIPlayer = currentState.whoseTurn
        enemyPlayer = 1-currentState.whoseTurn
        playerInv = currentState.inventories[AIPlayer]
        enemyInv = currentState.inventories[enemyPlayer]
        playerQueen = playerInv.getQueen()
        enemyQueen = enemyInv.getQueen()
        playerFoodCount = playerInv.foodCount
        enemyFoodCount = enemyInv.foodCount

        # copy a stateDictionary skeleton to represent the state
        consolidatedState = stateDictionary.copy()

        # win conditions for each team
        if playerQueen is None or enemyFoodCount >= 11:
            consolidatedState["loss"] = True
        elif enemyQueen is None or playerFoodCount >= 11:
            consolidatedState["win"] = True

        # get the food counts per team
        consolidatedState["FoodCount"] = playerFoodCount

        # judge the health of both player queen ants
        if playerInv.getQueen() is not None:
            consolidatedState["PQueenHealth"] = playerQueen.health
        if enemyInv.getQueen() is not None:
            consolidatedState["EQueenHealth"] = enemyQueen.health
            # record the distance of the closest player ant to the enemy queen
            bestDist = self.INFINITY
            for ant in playerInv.ants:
                if ant.type is not QUEEN:
                    antToQueen = self.vectorDistance(ant.coords, enemyQueen.coords)
                    if antToQueen < bestDist:
                        bestDist = antToQueen
            if bestDist is not None:
                consolidatedState["AntToQueen"] = bestDist

        return consolidatedState

    ##
    # toFrozenset
    # Description: adjustUtility relies on inserting new or previously discovered
    #               consolidated states. However a dictionary cannot be a key to another
    #               dictionary. So I must convert the consolidated state to something
    #               hashable. the set type 'frozenset' is immutable and hashable.
    #
    # Parameter:
    #   myDictionary - the consolidated dictionary of a state that I need to make hashable
    #
    # Return:
    #   hashableState - a frozenset representation of the dictionary
    #
    def toFrozenset(self, myDictionary):
        hashableState = frozenset(myDictionary.items())
        return hashableState

    ##
    # saveUtilityList
    # Description: save the current utility list to a file. Pythons Pickle is effective at saving/loading
    #               lists.
    #     reference from: http://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file-in-python
    #
    def saveUtilityList(self):
        with open(self.fileName, 'wb') as f:
            pickle.dump(self.utilityList, f, 0)
        return


    ##
    # loadUtilityList
    # Description: load the utility list from a file, using pickle file loading
    #
    def loadUtilityList(self):
        with open(self.fileName, 'rb') as f:
            return pickle.load(f)


    ##
    # reward
    # Description: Give a rewarding score for each new board state
    #
    # Parameters:
    #   consolidatedState - the (shrunken down) dictionary description of the state
    #
    # Return:
    #   stateReward - 1 for a winning state, -1 for a losing state, -.04 for anything else
    #
    def reward(self, consolidatedState):
        # init the reward to return
        stateReward = -.04

        # if the state has the player AI winning
        if consolidatedState["win"] == True:
            return 1.0

        # if the state has the AI player losing
        elif consolidatedState["loss"] == True:
            return -1.0

        # all other states are a hard coded -.04 reward
        else:
            return  -.04

        return stateReward


    ##
    # actionToTake
    # Description: From a current state, decide what the highest scoring utility of a state coming
    #               from the list of moves to make,then with a 95% probability take that move,
    #               5% to select a random move to try different things.
    #
    # Parameter:
    #   currentState - the current state of the game
    #
    # Return:
    #   actionMove - 95% the move that results in the highest scoring utility, 5% something random
    #
    def actionToTake(self, currentState):
        # init the move to make, highest scoring utility, all available moves, and a random value
        highestUtility = -self.INFINITY
        actionMove = None
        randomVal = 20.0*random.random()
        allMoves = listAllLegalMoves(currentState)

        # 95% chance to pick the highest scoring utility move
        if randomVal > 1:
            for move in allMoves:
                # get the future (state)
                futureState = self.processMove(currentState, move)
                # get the current consolidated state utility
                currentUtility = self.adjustUtility(currentState, futureState)
                # hold the highest scoring utility
                if currentUtility > highestUtility:
                    highestUtility = currentUtility
                    actionMove = move

        # 5% chance to pick a random move
        else:
            randomIdx = random.randint(0, len(allMoves)-1)
            actionMove = allMoves[randomIdx]

        return actionMove


    ##
    # adjustUtility
    # Description: use TD-learning to adjust the utility of the past state in the list of utilities.
    #             NOTE: At the moment this method is called, the past state I am
    #                   referring to is the current state of the game, thus it is called currentState.
    #
    # Parameter:
    #   currentState - the current state of the game
    #   futureState - the next state of the game
    #
    # Return:
    #   currentUtility - the utility score of the current state before the move occurs
    #
    def adjustUtility(self, currentState, futureState):
        # create a dictionary of the currentState, next state
        currentDict = self.consolidate(currentState)
        currentFreeze = self.toFrozenset(currentDict)
        # init the current state utility
        currentUtility = 0.0

        # special case for the first move of the game...there's no known future state
        if futureState is None:
            if currentFreeze not in self.utilityList:
                self.utilityList[currentFreeze] = 0.0
        else:
            nextDict = self.consolidate(futureState)
            nextFreeze = self.toFrozenset(nextDict)
            # if this state is new to utilityList, give it a default value
            if nextFreeze not in self.utilityList:
                self.utilityList[nextFreeze] = 0.0
            if currentFreeze not in self.utilityList:
                self.utilityList[currentFreeze] = 0.0
            else:
              self.utilityList[currentFreeze] += self.alpha*(self.reward(currentDict) + self.gamma*self.utilityList[nextFreeze] - self.utilityList[currentFreeze])

        # set the current state utility
        currentUtility = self.utilityList[currentFreeze]
        return currentUtility


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
        #self.unitTest()
        # for the first move of the game
        if self.justBegun:
            self.justBegun = False
            # add this state to the list
            self.adjustUtility(currentState, None)

        return self.actionToTake(currentState)


        # return the best move, found by recursively searching potential moves
        #return self.exploreTree(currentState, currentState.whoseTurn, 0)
    
    
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
        self.justBegun = True
        self.gamecount += 1
        self.alpha = self.alpha * self.decreaseRate
        if self.gamecount % 10 == 0:
            self.saveUtilityList()



