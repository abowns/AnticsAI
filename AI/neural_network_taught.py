# -*- coding: latin-1 -*-

##
# 
# Homework 7 - Neural Networking
#
# Author(s): Alex Bowns, Lysa Pinto
#
from Player import *
from Construction import Construction
from Ant import Ant
from AIPlayerUtils import *
import random
import math

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
        self.maxDepth = 3
        self.gameCount = 1
        #the alpha value for how quick we want our neural net to change its weights
        self.alphaQ = 2.0
        self.hardCodedInputWeightList =  [[-0.19900395667672993, 1.0107123512356682, -0.1298513227537575, -0.5363628898331391, 0.9491806697895203, -0.5390457738796782, 0.6581287850217622, 0.9806648887448136, -0.6415356619034415, -0.9521839236248851, -0.1651694437169518, -0.9141547769894994],
                                          [-1.000804615257194, -0.3314825118026529, 0.7717712901385938, -0.003779220054069834, 0.47248568315013034, 0.364087140953163, -0.3336789759073255, 0.9523936863969136, -0.7448709366823567, 0.38670585823368336, -0.35665170454039846, -0.21580881583597852],
                                          [215.56998847257935, 216.840792289589, 216.13580301114175, 216.61036316666394, 215.893962258438, 216.81322363468317, 216.3238029071687, 215.63701129491668, 216.90709024069272, 216.98870189667562, 215.37189464218244, 215.9778942978929],
                                          [-291.4596372475496, -292.45111508892586, -292.53568917281353, -292.2057214105054, -291.7522643459505, -291.9121852493282, -292.61170103791386, -292.21945263396594, -291.3265199960183, -292.72546832935944, -292.2205882482672, -291.4529095513118],
                                          [-0.9499935655455531, 0.9689328791763545, 0.22096958160240088, -0.35757467846305, 0.1908454853206849, -0.6740250430109169, -0.8059500866346327, 0.13457582474339258, 0.29666309650576617, -0.5921148887682253, 0.11260784698254467, -0.7399385951923101],
                                          [5.912097824174489, 5.4671037343946205, 6.1610653054492985, 5.3650754240659655, 5.47728759076573, 5.818008756294898, 5.922044312612913, 6.722173140026038, 6.471913074861952, 5.139005251058886, 5.424991848898831, 5.900625983642664]]
        self.hardCodedLayerWeightList =   [-0.5454092010800999, 0.13967306660718454, 0.6939884701926687, 0.365651975635759, -0.24165962145021033, 0.23459032598955262, -0.3519660930798465, 0.9790079880896103, 0.2846030782449516, 0.37926444911770063, -0.3960423900436394, 0.4188261606951956]
        self.inputWeightList = []
        self.layerWeightList = []

        #self.inputWeightList = []
        # Juanito - He is juan week old
        super(AIPlayer,self).__init__(inputPlayerId, "Juanito")

    ##
    #mapInputs
    #Description: Take the list of all possible moves from a current game state.
    #   Then create an array (of floating point numbers) where each element in the array
    #   is an evaluation score from the evaluationState function of the theoretical
    #   state that would emerge from the specific move.
    #
    #Parameters:
    #   currentState - The current state of the board
    #
    #Return:
    #   inputArray[] - a mapped array of inputs which are now floating pt numbers between 0-1
    def mapInputs(self, currentState):
        inputArray = []
        copiedState = None
        for move in listAllLegalMoves(currentState):
            copiedState = self.processMove(currentState, move)
            inputArray.append(self.evaluateState(copiedState))
        return inputArray

    ##
    #initRandomWeights
    #Description: create a list of random weights paired with a specific input from the inputArray[]
    #
    #Parameters:
    #   inputList - a mapped array of inputs which are now floating pt numbers between 0-1
    #
    #Return:
    #   allInputsWeightList - a list of weights for every single input of the inputList[]
    #
    def initRandomWeight(self, inputList):
        allInputsWeightList = []
        weightsForAnInput = []

        for i in inputList:
            for p in range(0, 2*len(inputList)):
                #initialize all the weights for a single input
                weightsForAnInput.append(random.uniform(-1, 1))
            #append the list of weights from a single input to the allInputsWeightList
            allInputsWeightList.append(weightsForAnInput)
            weightsForAnInput = []

        return allInputsWeightList

    ##
    #layerRandomWeight
    #Description: create a list of random weights for every single output of the layer 'nodes'
    #
    #Parameters:
    #   layerOutputList - a mapped array of outputs which are now floating pt numbers between 0-1
    #
    #Return:
    #   layerWeightList - a list of weights for every single output in the layer section
    #
    def layerRandomWeight(self, layerOutputList):
        layerWeightList = []
        #initialize a random weight for each layer 'node'
        for i in range(0, len(layerOutputList)):
            layerWeightList.append(random.uniform(-1, 1))

        return layerWeightList

    ##
    #gFunction
    #Description: For every input, i,  given, apply a function g(i) which returns an output
    #value for the layer node.
    #
    #Parameters:
    #   inputList - the list of inputs to be run on
    #
    #Returns:
    #   outputList - a list of output values for each input that was run
    def gFunction(self, inputList):
        value = 0.0
        outputList = []
        #for each input value, i, in the input list, apply g(i)
        for i in inputList:
            negI = -i
            value = 1.0/(1.0 + math.exp(negI))
            outputList.append(value)

        return outputList

    ##
    #mapOutput
    #Description: Find the  index of the best move that the neural network has decided.
    #
    #Parameters:
    #   output - the final output of the neural network, recorded as a float between 0-1
    #   inputList - the list of all original input values
    #
    #Return:
    #   closeIdx the index of the element in inputList with the closest value to the output value
    #
    def mapOutput(self, output, inputList):
        closest = INFINITY
        idx = 0
        closeIdx = 0
        #find the closest value from inputList to the output of the neural network,
        # then record the index of the closest value
        for i in inputList:
           if abs(output - i) < closest:
               closest = i
               closeIdx = idx
           idx += idx

        return closeIdx

    ##
    #runHardCodedNetwork
    #Description: run the hard coded weights for the neural network
    #
    #Parameters:
    #   inputList - the 6 evaluations made by the player
    #   targetNode - the targeted node to return
    #
    #Return
    #   the move
    #

    def runHardCodedNetwork(self, inputList, targetNode):
        finalInput = [] #the final input
        layerInputList = [] #a list of all layer inputs created from the input nodes and weights

        #create the input value for every single 'node' in the hidden layer
        input = 0.0
        for w in range(0, 2*len(inputList)):
            for i in range(0, len(inputList)):
                input += (inputList[i]*self.hardCodedInputWeightList[i][w])
            layerInputList.append(input)
            input = 0.0

        #get the list of output values for each 'node' in the hidden layer
        layerOutputList = self.gFunction(layerInputList)

        #determine single input from the sum of all of the layer 'nodes'
        myValue = 0.0
        for z in range(0, len(self.layerWeightList)):
            myValue += (layerOutputList[z]*self.hardCodedLayerWeightList[z])
        finalInput.append(myValue)
        #get the score for this node
        finalOutput = self.gFunction(finalInput)

        #create the move and return it
        move = targetNode["move"]
        return targetNode["move"]

    ##
    #neuralNetOutput
    #Description: Use the neural networking algorithm to determine which state to go to.
    #
    #Parameters:
    #   currentState - the currentState of the board
    #   inputList - the list of values that represent an evaluation of each available move from the current state
    #   targetNode - The target node that we want our neural network to follow
    #
    #Return:
    #   the outputted move that we would select
    #
    def neuralNetOutput(self, currentState, inputList, targetNode):
        if len(inputList) != 6:
            print "hi"
        loopCount = 0
        targetValue = targetNode["state_value"]
        #a helper boolean to tell our network to keep adjusting all of its weights when 'true'
        keepLooping = True
        if self.inputWeightList == []:
            self.inputWeightList = self.initRandomWeight(inputList)
        finalInput = [] #the final input
        layerInputList = [] #a list of all layer inputs created from the input nodes and weights

        #create the input value for every single 'node' in the hidden layer
        input = 0.0
        for w in range(0, 2*len(inputList)):
            for i in range(0, len(inputList)):
                input += (inputList[i]*self.inputWeightList[i][w])
            layerInputList.append(input)
            input = 0.0

        #get the list of output values for each 'node' in the hidden layer
        layerOutputList = self.gFunction(layerInputList)
        #create a list of random weights, each weight is paired with a layer 'node'
        if self.layerWeightList == []:
            self.layerWeightList = self.layerRandomWeight(layerOutputList)

        #keep readjusting until final output is close enough
        while keepLooping:
            #determine single input from the sum of all of the middle 'nodes' output*weight
            myValue = 0.0
            for z in range(0, len(self.layerWeightList)):
                myValue += (layerOutputList[z]*self.layerWeightList[z])
            finalInput.append(myValue)
            #get the score for this node
            finalOutput = self.gFunction(finalInput)
            #if we've looped long enough, just return the value
            if loopCount > 50:
                return (finalOutput[0], targetNode["move"])
            # if the final output is close enough to the target value, parse the output to a move
            if abs(targetValue - finalOutput[0]) < .03:
                keepLooping = False
                #index = self.mapOutput(finalOutput[0], inputList)
                #move = listAllLegalMoves(currentState)[index]
                move = targetNode["move"]
                return (finalOutput[0], targetNode["move"])
                #return (inputWeightList, layerWeightList, finalOutput[0], move)

            #final output not close enough to target, adjust all weights and try again.
            else:
                finalOutputError = targetValue - finalOutput[0]
                #create new weights for the layer and input weight lists
                tuple = self.backPropogation(layerOutputList, finalOutputError, finalOutput, inputList)
                self.layerWeightList = tuple[0]
                self.inputWeightList = tuple[1]

                print "#####################################"
                print "Target: ", targetValue
                print "Output: ", finalOutput[0]
                print "Error: ", targetValue-finalOutput[0]
                print "Continuing backpropogation..."
                print "\n "

                layerInputList = []
                layerOutputList = []
                #rescore layer input list
                for w in range(0, 2*len(self.inputWeightList)):
                    for i in range(0, len(inputList)):
                        input += (inputList[i]*self.inputWeightList[i][w])
                    layerInputList.append(input)
                    input = 0.0
                #rescore layer output list
                layerOutputList = self.gFunction(layerInputList)

                ##TODO reused code, make helper
                #determine single input from the sum of all of the hidden 'nodes' output*weight
                myValue = 0.0
                finalInput = []
                for z in range(0, len(self.layerWeightList)):
                    myValue += (layerOutputList[z]*self.layerWeightList[z])
                #apply the determined final input value to a list, (so we can use an already made function)
                finalInput.append(myValue)
                #get the score for this node
                finalOutput = self.gFunction(finalInput)

                loopCount = loopCount + 1


    ##
    #backPropogation
    #Description: Calculate the new weights for both the input weights and layer weights.
    #Begin by calculating the new weights in the layer section, then use the error and
    #error term values as references to calculate the new weights in the input section (with a separate method).
    #
    #Parameters:
    #   weightList - the list of weights
    #   outputError - the error of the outputNode
    #   finalOutput - the single output evaluation score given
    #
    #Return:
    #   the new list of weights
    def backPropogation(self, layerOutputList, finalOutputError, finalOutput, inputList):
        #calculate the error term for the final output node of the network
        a = finalOutput[0]
        deltaOut = a*(1.0-a)*finalOutputError
        #instance variables to hold the error, error term, and new weights for all of the layer nodes
        errorLayerList = []
        deltaLayerList = []
        newLayerWeightList = []

        #calculate error for all of the layer 'nodes'
        for i in range(0, len(self.layerWeightList)):
           err = self.layerWeightList[i]*deltaOut
           errorLayerList.append(err)
           err = 0

        #calculate error term for all of the layer 'nodes'
        for i in range(0, len(self.layerWeightList)):
            errTerm = layerOutputList[i]*(1.0-layerOutputList[i])*finalOutputError
            deltaLayerList.append(errTerm)
            errTerm = 0

        #calculate new weights for the layer 'nodes'
        for i in range(0, len(self.layerWeightList)):
            wNew = self.layerWeightList[i] + (float(self.alphaQ) * deltaLayerList[i] * layerOutputList[i])
            newLayerWeightList.append(wNew)
            wNew = 0

        propogatedInputList = self.backPropogationInputWeights(inputList, deltaLayerList)
        #return a tuple of the new weights from the input and layer nodes
        return (newLayerWeightList, propogatedInputList)

    ##
    #backPropogationInputWeights
    #Description: Apply propogation to create new weights for every single
    #
    #Parameters:
    #   inputWeightList - the list of weights attached to all of the input nodes
    #   inputList - the list of inputs
    #   deltaList - the list of error terms
    #
    def backPropogationInputWeights(self, inputList, deltaList):
        inputErrorList = []
        deltaInputList = []
        error = 0.0
        #calculate error for all of the input 'nodes'
        for i in range(0, len(inputList)):
            for z in range(0, 2*len(inputList)):
                error += self.inputWeightList[i][z]*deltaList[z]

            inputErrorList.append(error)
            error = 0.0

        #calculate error term (delta) for every input 'node'
        for i in range(0, len(inputList)):
             errTerm = inputList[i]*(1-inputList[i])*inputErrorList[i]
             deltaInputList.append(errTerm)
             errTerm = 0.0

        singleNodeWeightList = []
        allInputWeightList = []

        #calculate new weights for the layer 'nodes'
        for i in range(0, len(self.inputWeightList)):
            for z in range(0, 2*len(self.inputWeightList)):
                wL = self.inputWeightList[i][z]
                mult = (self.alphaQ * deltaInputList[i] * inputList[i])
                wNew = self.inputWeightList[i][z] + (self.alphaQ * deltaInputList[i] * inputList[i])
                singleNodeWeightList.append(wNew)
                wNew = 0.0
            allInputWeightList.append(singleNodeWeightList)
            singleNodeWeightList = []

        return allInputWeightList

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
        # save our id
        self.playerId = currentState.whoseTurn
        #create the target node for our neural network to analyze
        initNode = self.createNode(None, currentState, None)
        targetNode = self.alpha_beta_search2(initNode)
        #apply a neural network to learn the same result as what the original minimax would've learned
        inputList = self.mapCurrentState(currentState)
        neuralResult = self.runHardCodedNetwork(inputList, targetNode)
        return neuralResult

    ##
    #mapCurrentState
    #Description: Map the state of the game to create the neural networks input list.
    #
    #Parameters:
    #   currentState - the currentState of the game
    #
    #Return:
    #   inputList - the list of 6 different input values
    #
    def mapCurrentState(self, currentState):
        #this will store all a value between 0-1 for factors our evaluation function considers as well
        inputList = []
        # get a reference to the player's inventory
        playerInv = currentState.inventories[currentState.whoseTurn]
        # get a reference to the enemy player's inventory
        enemyInv = currentState.inventories[(currentState.whoseTurn+1) % 2]
        # get a reference to the enemy's queen
        enemyQueen = enemyInv.getQueen()
        #player ants reference
        playerAnts = playerInv.ants

        #player food count
        inputList.append(float(playerInv.foodCount/11.0))

        #enemy player food count
        inputList.append(float(enemyInv.foodCount/11.0))

        #enemy queen is dead
        if enemyInv.getQueen() is None:
            inputList.append(float(0)) #enemy queen health is not a concern
            inputList.append(float(0)) #player ants closer to queen

        else:
            #enemy queen health
            inputList.append(1.0 - float(enemyQueen.health/ float(UNIT_STATS[QUEEN][HEALTH])))
            #if down to no extra ants, make sure to add a 0.0 to the inputList
            if len(playerAnts) == 1 and playerAnts[0].type == QUEEN:
                 inputList.append(float(0))
            # player ants moving closer to the queen
            for ant in playerAnts:
                if ant.type == QUEEN:
                    continue
                else:
                    inputList.append(float(1.0/float(self.vectorDistance(ant.coords, enemyQueen.coords))))

        #player queen is dead
        if playerInv.getQueen() is None:
            inputList.append(0.0) #player queen health is not a concern
            inputList.append(float(0)) #queen distance from enemies
        else:
            #player queen health
            inputList.append(float(playerInv.getQueen().health) / float(UNIT_STATS[QUEEN][HEALTH]))
            #queen distance from enemies
            for ant in playerInv.ants:
                if ant.type == QUEEN:
                    enemyDistFromQueen = float(self.distClosestAnt(currentState, ant.coords))
                    queenSafety = float(enemyDistFromQueen / maxDist)
                    inputList.append(queenSafety)

        return inputList

    # alpha_beta_search2
    # Description: use minimax with alpha beta pruning to determine the best evaluation score
    #
    # Parameters:
    #   self - the object pointer
    #   node - the initial node, before any moves are explored
    #
    # Returns: the highest evaluating node
    #
    def alpha_beta_search2(self, node):
        bestNode = self.max_value(node, -INFINITY, INFINITY, 0)
        while bestNode["parent_node"]["parent_node"] is not None:
            bestNode = bestNode["parent_node"]
        return bestNode

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
           # print "Our Node Value: ", newNode["state_value"]
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
            # print "Their Node Value: ", newNode["state_value"]
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
        print "###################################################\nGame Finished" \
              "\nGame Count: ", self.gameCount,"\n###################################################\n"
        print "Input Weight List: ", self.inputWeightList
        print "Hidden Layer Weight List", self.layerWeightList
        print "...Beginning Next Game\n\n"
        self.gameCount = self.gameCount + 1
