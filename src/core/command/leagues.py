
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
        (player-id, player-rating, player-parameters, player-stats)
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

class EloGaussServerLeague(ServerLeague, metaclass=abc.ABCMeta):
    """
    Use simple elo rating for players, use gaussian mutation for generations.
    """
    def __init__(self, parameters, generationGames, populationSize, mutateTopN, mutateCount, initialRating, n, K):
        self.parameters = parameters
        self.generationGames = generationGames
        self.populationSize = populationSize
        self.initialRating = initialRating
        self.n = n
        self.K = K
        self.mutateTopN = mutateTopN
        self.mutateCount = mutateCount

        self.loadedPlayers = False
        # players in the EloGaussServerLeague have a 4th entry in their tuple: a dict of
        # the same shape as the 3rd entry, which represents the std-devs of the gaussian mutation steps.
        self.players = []

        # caching sets to reduce number of database queries
        self.existingPlayers = set()
        self.networkKnowledge = dict()

        self.loadedMatchHistory = False
        self.matchHistory = []

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
        return self.matchHistory

    def loadPlayers(self, pool, runId):
        if not self.loadedPlayers:
            self.loadedPlayers = True

            try:
                con = pool.getconn()
                cursor = con.cursor()

                cursor.execute("SELECT id, run, rating, parameter_vals, parameter_stddevs from league_players where run = %s", (runId, ))
                rows = cursor.fetchall()

                for row in rows:
                    self.players.append([row[0], row[2], json.loads(row[3]), json.loads(row[4])])

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
                cursor.execute("insert into league_players (id, run, rating, parameter_vals, parameter_stddevs) VALUES (%s, %s, %s, %s, %s)",\
                    (player[0], runId, player[1], json.dumps(player[2]), json.dumps(player[3])))
            else:
                cursor.execute("update league_players set rating = %s, parameter_vals = %s, parameter_stddevs = %s where id = %s",\
                    (player[1], json.dumps(player[2]), json.dumps(player[3]), player[0]))
            con.commit()
            self.existingPlayers.add(player[0])
        finally:
            if cursor:
                cursor.close()
            pool.putconn(con)

    def batchAddMatches(self, pool, runId, matches):
        self.matchHistory += matches

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
        
        def mkQMarkListsFor(qLsts):
            result = ""
            for qLst in qLsts:
                result += "("
                result += ",".join(["%s"] * len(qLst))
                result += ")"
            return result

        flatten = lambda l: [item for sublist in l for item in sublist]

        valLists = [];

        for match in matches:
            tstamp = datetime.datetime.fromtimestamp(match[4] / 1000).astimezone().isoformat()
            valLists.append([runId, match[5] if self.networkKnowledge[match[5]] else None, match[0], match[1], match[2], match[3], tstamp])
        
        qLists = mkQMarkListsFor(valLists)

        qList = ",".join(qLists)
        valFlat = flatten(valLists)

        iSql = "insert into league_matches (run, network, player1, player2, result, ratingChange, creation) VALUES " + qList
        print(iSql, valFlat)

        try:
            con = pool.getconn()
            cursor = con.cursor()
            cursor.execute(iSql, valFlat)
            con.commit()
        finally:
            if cursor:
                cursor.close()
            pool.putconn(con)

    def addNewMatch(self, pool, runId, match):
        self.matchHistory.append(match)

        try:
            con = pool.getconn()
            cursor = con.cursor()

            cursor.execute("SELECT id from networks where id = %s", (match[5],))
            knowsNetwork = len(cursor.fetchall()) > 0

            cursor.close()
            cursor = con.cursor()

            tstamp = datetime.datetime.fromtimestamp(match[4] / 1000).astimezone().isoformat()
            cursor.execute("insert into league_matches (run, network, player1, player2, result, ratingChange, creation) VALUES (%s,%s,%s,%s,%s,%s,%s)",\
                (runId, match[5] if knowsNetwork else None, match[0], match[1], match[2], match[3], tstamp))

            con.commit()
        finally:
            if cursor:
                cursor.close()
            pool.putconn(con)

    def persistPlayers(self, pool, runId):
        for p in self.players:
            self.persistPlayer(pool, p, runId)

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

        return [playerId, playerRating, playerParameters, playerStddevs]

    def mutatePlayer(self, player):
        playerId = str(uuid.uuid4())
        # TODO configuration of the "mutated players probably suck and need to prove themselves"-value
        playerRating = player[1] - 100

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

                    if nextVal > maximum:
                        nextVal = maximum
                    if nextVal < minimum:
                        nextVal = minimum

                    playerParameters[k][vi] = nextVal
                    playerStddevs[k][vi] = nextStddev

        return [playerId, playerRating, playerParameters, playerStddevs]

    def initPlayers(self):
        for _ in range(self.populationSize):
            self.players.append(self.newPlayer())
        self.sortPlayers()
        
    def sortPlayers(self):
        self.players.sort(key=lambda x: x[1], reverse=True)

    def getAllPlayerStats(self):
        statsDict = dict()

        for x in self.matchHistory:
            if not x[0] in statsDict:
                statsDict[x[0]] = [0,0,0]
            if not x[1] in statsDict:
                statsDict[x[1]] = [0,0,0]

            if x[2] == 1:
                statsDict[x[0]][0] += 1
                statsDict[x[1]][1] += 1
            elif x[2] == 0:
                statsDict[x[0]][1] += 1
                statsDict[x[1]][0] += 1
            else:
                statsDict[x[0]][2] += 1
                statsDict[x[1]][2] += 1

        return statsDict

    def getPlayers(self, pool, runId):
        """
        return a list of players, sorted by ranking, a player is a tuple:
        (player-id, player-rating, player-parameters, player-stats)
        """
        self.loadPlayers(pool, runId)
        self.getMatchHistory(pool, runId)

        pstats = self.getAllPlayerStats()

        result = list(map(lambda x: (x[0], x[1], x[2], pstats[x[0]] if x[0] in pstats else [0,0,0]), self.players))

        # print("==== players:")
        # for idx, p in enumerate(result):
        #     if idx > 42:
        #         break
        #     foo = str(idx + 1)
        #     if len(foo) < 2:
        #         foo = "0" + foo
        #     print("#"+foo, p[0].split("-")[1], int(p[1]), p[3], p[2])
        # print("... %i more players not shown" % (len(result) - 42))

        return result

    def calculateNumberOfPlayers(self):
        generations = len(self.matchHistory) // self.generationGames
        newPlayersPerGeneration = self.mutateTopN * self.mutateCount
        return self.populationSize + newPlayersPerGeneration * generations

    def handleGenerations(self, pool, runId):
        expectedPlayers = self.calculateNumberOfPlayers()
        if expectedPlayers > len(self.players):
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

        for pKey in persistsPlayers:
            self.persistPlayer(pool, pDict[pkey], runId)
        


    def reportResult(self, p1, p2, winner, policyUUID, runId, pool):
        """
        give two player ids, and either player1 id, player2id or None for a draw.
        Update ratings and possibly mutate players as a response.
        """

        assert False, "old code, do not call, delete it soon"

        self.loadPlayers(pool, runId)
        self.getMatchHistory(pool, runId)
        p1s = list(filter(lambda x: x[0] == p1, self.players))
        p2s = list(filter(lambda x: x[0] == p2, self.players))
        assert len(p1s) == 1
        assert len(p2s) == 1

        sa = 1
        if winner is None:
            sa = 0.5
        if winner == p2:
            sa = 0

        p1R, p2R = self.getNewRatings(p1s[0][1], p2s[0][1], sa)

        r1Change = p1R - p1s[0][1]
        r2Change = p2R - p2s[0][1]

        p1s[0][1] = p1R
        p2s[0][1] = p2R

        self.persistPlayer(pool, p1s[0], runId)
        self.persistPlayer(pool, p2s[0], runId)

        self.addNewMatch(pool, runId, (p1, p2, sa, abs(r1Change), int(1000.0 * datetime.datetime.utcnow().timestamp()), policyUUID))

        self.handleGenerations(pool, runId)

        self.sortPlayers()