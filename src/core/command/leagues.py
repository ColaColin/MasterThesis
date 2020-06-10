
import uuid
import abc
import random
import numpy as np
import math
import json

import datetime
from utils.prints import logMsg

import time

class ServerLeague(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def getPlayers(self, pool, runId):
        """
        return a list of players, sorted by ranking, a player is a tuple:
        (player-id, player-rating, player-parameters, player-stats, player-generation)
        player-parameters is a dict of string-key to single number or a list of numbers.
        player-stats is a dict:
        {wins, losses, draws}
        """

    @abc.abstractmethod
    def reportResultBatch(self, rList, pool, runId):
        """
        give two player ids, and either player1 id, player2id or None for a draw.
        Update ratings and possibly mutate players as a response.

        rList is a list of tuples (p1, p2, winner, policyUUID, runId)
        """

    @abc.abstractmethod
    def getMatchHistory(self, pool, runId):
        """
        returns the match history as a list of tuples:
        (player1Id, player2Id, result (1 is p1 wins, 0 is p2 wins, 0.5 is draw), ratingChange, timestamp)
        """

def mkQMarkListsFor(qLsts):
    result = []
    for qLst in qLsts:
        foo = "(" + ",".join(["%s"] * len(qLst)) + ")"
        result.append(foo)
    return ",".join(result)

flatten = lambda l: [item for sublist in l for item in sublist]

def limitByRollover(val, minVal, maxVal):
    while val > maxVal:
        overSize = val - maxVal
        val = minVal + overSize
    while val < minVal:
        underSize = minVal - val
        val = maxVal - underSize
    return val

def limitByCut(val, minVal, maxVal):
    pass

class EloGaussServerLeague(ServerLeague, metaclass=abc.ABCMeta):
    """
    Use simple elo rating for players, use gaussian mutation for generations.
    """
    def __init__(self, parameters, generationGames, populationSize, mutateTopN, mutateCount, initialRating, n, K, fixedParameters=None, restrictMutations=True):
        self.parameters = parameters
        self.fixedParameters = fixedParameters
        self.generationGames = generationGames
        self.populationSize = populationSize
        self.initialRating = initialRating
        self.n = n
        self.K = K
        self.mutateTopN = mutateTopN
        self.mutateCount = mutateCount
        self.restrictMutations = restrictMutations

        self.loadedPlayers = False
        # shape: [id, rating, params, params-stddev, generation]
        self.players = []

        # caching sets to reduce number of database queries
        self.existingPlayers = set()
        self.networkKnowledge = dict()

        self.loadedMatchHistory = False
        # (player1, player2, result, ratingChange, timestamp in unix ms)
        self.matchHistory = []

        self.playerStatsDict = dict()

        self.currentGeneration = 1

    def getMatchHistory(self, pool, runId):
        if not self.loadedMatchHistory:
            self.loadedMatchHistory = True

            try:
                con = pool.getconn()
                cursor = con.cursor()

                cursor.execute("SELECT player1, player2, result, ratingChange, creation from league_matches where run = %s", (runId, ))
                rows = cursor.fetchall()

                for row in rows:
                    self.matchHistory.append((row[0], row[1], row[2], row[3], int(row[4].timestamp() * 1000)))

            finally:
                if cursor:
                    cursor.close()
                pool.putconn(con)

            logMsg("Loaded a history of %i matches for run %s" % (len(self.matchHistory), runId))

            self.matchHistory.sort(key=lambda x: x[4])

            self.recalcPlayerStats()

        return self.matchHistory

    def loadPlayers(self, pool, runId):
        if not self.loadedPlayers:
            self.loadedPlayers = True

            try:
                con = pool.getconn()
                cursor = con.cursor()

                cursor.execute("SELECT id, run, rating, parameter_vals, parameter_stddevs, generation from league_players where run = %s", (runId, ))
                rows = cursor.fetchall()

                for row in rows:
                    self.players.append([row[0], row[2], json.loads(row[3]), json.loads(row[4]), row[5]])
                    if row[5] > self.currentGeneration:
                        self.currentGeneration = row[5]

            finally:
                if cursor:
                    cursor.close()
                pool.putconn(con)

            logMsg("Loaded existing %i players for run %s" % (len(self.players), runId))

            if len(self.players) == 0:
                # no players in the db -> init first generation of players
                self.initPlayers()
                self.persistPlayers(pool, runId)

                logMsg("Initialized %i players for run %s" % (len(self.players), runId))

        self.sortPlayers()

    def updatePlayers(self, pool, runId, players):
        """
        assumes the players already exist!
        """
        try:
            con = pool.getconn()
            cursor = con.cursor()

            for player in players:
                cursor.execute("update league_players set rating = %s where id = %s", (player[1], player[0]))
                if cursor:
                    cursor.close()
                cursor = con.cursor()

            con.commit()
        finally:
            if cursor:
                cursor.close()
            pool.putconn(con)

    def persistPlayer(self, pool, player, runId):
        try:
            con = pool.getconn()
            cursor = con.cursor()

            if not (player[0] in self.existingPlayers):
                cursor.execute("SELECT id from league_players where id = %s", (player[0],))
                alreadyExists = len(cursor.fetchall()) > 0
                cursor.close()
                cursor = con.cursor()
                if alreadyExists:
                    self.existingPlayers.add(player[0])
            else:
                alreadyExists = True

            if not alreadyExists:
                cursor.execute("insert into league_players (id, run, rating, parameter_vals, parameter_stddevs, generation) VALUES (%s, %s, %s, %s, %s, %s)",\
                    (player[0], runId, player[1], json.dumps(player[2]), json.dumps(player[3]), player[4]))
            else:
                cursor.execute("update league_players set rating = %s where id = %s", (player[1], player[0]))
            con.commit()
            self.existingPlayers.add(player[0])
        finally:
            if cursor:
                cursor.close()
            pool.putconn(con)

    def batchAddMatches(self, pool, runId, matches):
        self.matchHistory += matches

        for m in matches:
            self.addMatchToPlayerStats(m)

        knetworks = set()
        for m in matches:
            if not m[5] in self.networkKnowledge:
                knetworks.add(m[5])
        
        for kn in knetworks:
            try:
                con = pool.getconn()
                cursor = con.cursor()
                cursor.execute("SELECT id from networks where id = %s", (kn,))
                knowsNetwork = len(cursor.fetchall()) > 0
                self.networkKnowledge[kn] = knowsNetwork
            finally:
                if cursor:
                    cursor.close()
                pool.putconn(con)

        valLists = [];

        for match in matches:
            tstamp = datetime.datetime.fromtimestamp(match[4] / 1000).astimezone().isoformat()
            valLists.append([runId, match[5] if self.networkKnowledge[match[5]] else None, match[0], match[1], match[2], match[3], tstamp])
        
        qList = mkQMarkListsFor(valLists)
        valFlat = flatten(valLists)

        iSql = "insert into league_matches (run, network, player1, player2, result, ratingChange, creation) VALUES " + qList

        try:
            con = pool.getconn()
            cursor = con.cursor()
            cursor.execute(iSql, valFlat)
            con.commit()
        finally:
            if cursor:
                cursor.close()
            pool.putconn(con)

    def persistPlayers(self, pool, runId):
        for p in self.players:
            self.persistPlayer(pool, p, runId)

    def addDefaultsToParamsDict(self, pDict):
        newDict = dict(pDict)
        for k in self.fixedParameters:
            if k != "name":
                newDict[k] = self.fixedParameters[k]
        return newDict

    def newPlayer(self):
        playerId = str(uuid.uuid4())
        playerRating = self.initialRating

        playerParameters = dict()
        playerStddevs = dict()
        for k in self.parameters:
            if isinstance(self.parameters[k], list):
                minimum = self.parameters[k][0]
                maximum = self.parameters[k][1]
                diff = maximum - minimum
                dr = random.random() * diff
                playerParameters[k] = minimum + dr

                halfDiff = diff / 2
                stddev = random.random() * halfDiff
                playerStddevs[k] = stddev
            
            elif isinstance(self.parameters[k], dict):
                length = self.parameters[k]["length"]
                rangx = self.parameters[k]["range"]
                minimum = rangx[0]
                maximum = rangx[1]
                diff = maximum - minimum
                vector = [0] * length
                stddevs = [0] * length
                for vi in range(length):
                    vector[vi] = minimum + random.random() * diff
                    stddevs[vi] = random.random() * diff * 0.5
                playerParameters[k] = vector
                playerStddevs[k] = stddevs

        return [playerId, playerRating, self.addDefaultsToParamsDict(playerParameters), playerStddevs, 1]

    def mutatePlayer(self, player):
        playerId = str(uuid.uuid4())
        playerRating = player[1]

        currentParameters = player[2]
        currentStddevs = player[3]

        playerParameters = dict()
        playerStddevs = dict()

        parameterNum = 0
        for k in self.parameters:
            if isinstance(self.parameters[k], list):
                parameterNum += 1
            elif isinstance(self.parameters[k], dict):
                parameterNum += self.parameters[k]["length"]

        r = 1 / ((2 * (parameterNum ** 0.5)) ** 0.5)
        r1 = 1 / ((2 * parameterNum) ** 0.5)

        globalFactor = np.random.normal()

        for k in self.parameters:
            if isinstance(self.parameters[k], list):
                minimum = self.parameters[k][0]
                maximum = self.parameters[k][1]

                curVal = currentParameters[k]
                curStddev = currentStddevs[k]

                nextStddev = curStddev * math.exp(r1 * globalFactor + r * np.random.normal())
                nextVal = curVal + nextStddev * np.random.normal()
                
                if self.restrictMutations:
                    if nextVal > maximum:
                        nextVal = maximum
                    if nextVal < minimum:
                        nextVal = minimum
                
                playerParameters[k] = nextVal
                playerStddevs[k] = nextStddev
            
            elif isinstance(self.parameters[k], dict):
                length = self.parameters[k]["length"]
                rangx = self.parameters[k]["range"]
                minimum = rangx[0]
                maximum = rangx[1]

                for vi in range(length):
                    curVal = currentParameters[k][vi]
                    curStddev = currentStddevs[k][vi]

                    nextStddev = curStddev * math.exp(r1 * globalFactor + r * np.random.normal())
                    nextVal = curVal + nextStddev * np.random.normal()

                    if self.restrictMutations:
                        nextVal = limitByRollover(nextVal, minimum, maximum)

                    playerParameters[k][vi] = nextVal
                    playerStddevs[k][vi] = nextStddev

        return [playerId, playerRating, self.addDefaultsToParamsDict(playerParameters), playerStddevs, self.currentGeneration]

    def initPlayers(self):
        for _ in range(self.populationSize):
            self.players.append(self.newPlayer())
        self.sortPlayers()
        
    def sortPlayers(self):
        self.players.sort(key=lambda x: x[1], reverse=True)

    def addMatchToPlayerStats(self, x):
        if not x[0] in self.playerStatsDict:
            self.playerStatsDict[x[0]] = [0,0,0]
        if not x[1] in self.playerStatsDict:
            self.playerStatsDict[x[1]] = [0,0,0]

        if x[2] == 1:
            self.playerStatsDict[x[0]][0] += 1
            self.playerStatsDict[x[1]][1] += 1
        elif x[2] == 0:
            self.playerStatsDict[x[0]][1] += 1
            self.playerStatsDict[x[1]][0] += 1
        else:
            self.playerStatsDict[x[0]][2] += 1
            self.playerStatsDict[x[1]][2] += 1

    def recalcPlayerStats(self):
        self.playerStatsDict = dict()

        for x in self.matchHistory:
            self.addMatchToPlayerStats(x)

    def getPlayers(self, pool, runId):
        """
        return a list of players, sorted by ranking, a player is a tuple:
        (player-id, player-rating, player-parameters, player-stats)
        """
        self.loadPlayers(pool, runId)
        self.getMatchHistory(pool, runId)

        pstats = self.playerStatsDict

        result = list(map(lambda x: (x[0], x[1], x[2], pstats[x[0]] if x[0] in pstats else [0,0,0], x[4]), self.players[:self.populationSize * 2]))

        # print("==== players:")
        # for idx, p in enumerate(result):
        #     if idx > 42:
        #         break
        #     foo = str(idx + 1)
        #     if len(foo) < 2:
        #         foo = "0" + foo
        #     print("#"+foo, p[0].split("-")[1], p[4], int(p[1]), p[3], p[2])
        # print("... %i more players not shown" % (len(result) - 42))

        return result

    def hasLatestGenerationEnoughGames(self):
        gamesOfLatestGen = 0
        activeNewPlayers = 0
        for pidx, player in enumerate(self.players):
            if pidx >= self.populationSize:
                break
            if player[4] == self.currentGeneration:
                activeNewPlayers += 1
                if player[0] in self.playerStatsDict:
                    gamesOfLatestGen += np.sum(self.playerStatsDict[player[0]])
        # games show up twice in the stats, once as a win and once as a loss, or twice as a draw, so divide by 2.
        gamesOfLatestGen /= 2
        # only consider new generation players that have not dropped out of the active population
        # e.g. that have performed so bad, they got thrown down the ladder far enough to not start games anymore.
        if self.currentGeneration == 1:
            # the first generation has populationSize players
            neededGames = (activeNewPlayers / (self.populationSize)) * self.generationGames
        else:
            # all following generations have mutateCount * mutateTopN players
            neededGames = (activeNewPlayers / (self.mutateCount * self.mutateTopN)) * self.generationGames
        print("Current generation is %i, it has played %i/%i games. Still active: %i" % (self.currentGeneration, gamesOfLatestGen, neededGames, activeNewPlayers))
        return gamesOfLatestGen >= neededGames

    def storeGeneration(self, pool, runId):
        # store the current top 50 for the latest generation to be used later for evaluation purposes
        # as in "these were the best players of this generation"
        try:
            con = pool.getconn()

            # first figure out the current network in use
            cursor = con.cursor()
            cursor.execute("SELECT id from networks where run = %s order by creation desc limit 1", (runId, ))
            networkRows = cursor.fetchall()
            if len(networkRows) > 0:
                currentNetwork = networkRows[0][0]
            else:
                currentNetwork = None
            cursor.close()

            # no need to store generations for the random initial network
            # it never gets evaluated anyway.
            if currentNetwork is not None:
                cursor = con.cursor()
                valLists = []
                for player in self.players[:self.populationSize]:
                    valLists.append([player[0], self.currentGeneration, currentNetwork, player[1]])

                qList = mkQMarkListsFor(valLists)
                valFlat = flatten(valLists)

                iSql = "insert into league_players_snapshot (id, generation, network, rating) VALUES " + qList

                cursor.execute(iSql, valFlat)
                con.commit()                
        finally:
            if cursor:
                cursor.close()
            pool.putconn(con)

    def handleGenerations(self, pool, runId):
        shouldMutate = self.hasLatestGenerationEnoughGames()
        if shouldMutate:
            self.storeGeneration(pool, runId)

            self.currentGeneration += 1
            newPlayers = []
            for mi in range(self.mutateTopN):
                for _ in range(self.mutateCount):
                    nplayer = self.mutatePlayer(self.players[mi])
                    newPlayers.append(nplayer)
                    self.persistPlayer(pool, nplayer, runId)

            self.players += newPlayers
            self.sortPlayers()
            logMsg("Next generation has begun, now there are %i players!" % len(self.players))

    def getNewRatings(self, p1R, p2R, sa):
        ea = 1.0 / (1 + 10 ** ((p2R - p1R) / self.n))
        ratingDiff = abs(self.K * (sa - ea))
        if sa == 1:
            p1R += ratingDiff
            p2R -= ratingDiff
        if sa == 0:
            p1R -= ratingDiff
            p2R += ratingDiff
        if sa == 0.5:
            if p1R > p2R:
                p1R -= ratingDiff
                p2R += ratingDiff
            else:
                p1R += ratingDiff
                p2R -= ratingDiff
        
        return (p1R, p2R)

    def reportResultBatch(self, rList, pool, runId):
        self.loadPlayers(pool, runId)
        self.getMatchHistory(pool, runId)
        pDict = dict()
        for p in self.players:
            pDict[p[0]] = p
        
        persistsPlayers = set()
        newMatches = []

        for r in rList:
            p1, p2, winner, policyUUID, runId = r

            p1Object = pDict[p1]
            p2Object = pDict[p2]

            sa = 1
            if winner is None:
                sa = 0.5
            if winner == p2:
                sa = 0

            p1R, p2R = self.getNewRatings(p1Object[1], p2Object[1], sa)

            r1Change = p1R - p1Object[1]
            r2Change = p2R - p2Object[1]

            p1Object[1] = p1R
            p2Object[1] = p2R

            persistsPlayers.add(p1)
            persistsPlayers.add(p2)

            newMatches.append((p1, p2, sa, abs(r1Change), int(1000.0 * datetime.datetime.utcnow().timestamp()), policyUUID))

        self.updatePlayers(pool, runId, list(map(lambda x: pDict[x], persistsPlayers)))
       
        self.batchAddMatches(pool, runId, newMatches)

        self.handleGenerations(pool, runId)

        self.sortPlayers()