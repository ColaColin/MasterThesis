# cython: profile=False

from core.base.GameState import GameState

from utils.fields.fields cimport initField, writeField, printField, readField, areFieldsEqual, mallocWithZero

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

import abc

import numpy as np

# this is a copy of the MNK code, modified to play as connectk (with k = 4 you get the standard game...)

cdef struct Connect4_c:
    unsigned char m
    unsigned char n
    unsigned char k # = 4 by default
    unsigned char winningPlayer
    int turn
    # empty field is 0
    # filled fields contain the player number
    signed char* board

cdef Connect4_c* initConnect4(unsigned char m, unsigned char n, unsigned char k):
    cdef Connect4_c* result = <Connect4_c*> malloc(sizeof(Connect4_c))
    
    result.m = m
    result.n = n
    result.k = k
    result.winningPlayer = 255
    result.turn = 0
    result.board = initField(m, n, 0)

    return result;

cdef void copyConnect4(Connect4_c* target, Connect4_c* source):
    
    cdef int bsize = source.m * source.n * sizeof(signed char)

    if (target.m != source.m or target.n != source.n):
        free(target.board)
        target.board = <signed char*> malloc(bsize)
        
    memcpy(target.board, source.board, bsize)

    target.m = source.m
    target.n = source.n
    target.k = source.k

    target.turn = source.turn
    target.winningPlayer = source.winningPlayer

cdef inline void freeConnect4(Connect4_c* mnk):
    free(mnk.board)
    free(mnk)

cdef inline int getPlayerOnTurnNumberMNK(Connect4_c* mnk):
    return (mnk.turn % 2) + 1

cdef void searchWinner(Connect4_c* mnk, int lx, int ly):
    if mnk.winningPlayer != 255:
        return

    cdef signed char p = readField(mnk.board, mnk.m, lx, ly);
    cdef int[4][2] dirs = [[1, 0], [0, 1], [1, -1], [1, 1]]
    cdef int[2] invs = [1, -1]
    cdef int l, x, y, xdir, ydir, d, di

    if p != 0:
        for d in range(4):
            l = 0;
            
            for di in range(2):
                x = lx;
                y = ly;
                
                xdir = invs[di] * dirs[d][0];
                ydir = invs[di] * dirs[d][1];
                
                while x > -1 and y > -1 and x < mnk.m and y < mnk.n and readField(mnk.board, mnk.m, x, y) == p:
                    l += 1
                    x += xdir;
                    y += ydir;
                    
                    if l - 1 >= mnk.k:
                        mnk.winningPlayer = p
                        return

    if mnk.turn == mnk.m * mnk.n:
        mnk.winningPlayer = 0
        return

cdef void placeStone(Connect4_c* mnk, int x, int y):
    writeField(mnk.board, mnk.m, x, y, getPlayerOnTurnNumberMNK(mnk))
    mnk.turn += 1
    searchWinner(mnk, x, y)

cdef inline unsigned int updateMNKHash(unsigned int old, int x, int y, int playerNumber):
    cdef unsigned int turnCnt = old & 63
    cdef unsigned int sumPart = old >> 6
    cdef unsigned int acc = (x * 29 + y * 11027) * playerNumber
    if acc % 2 == 0:
        acc *= acc
    
    if x == 0 and y == 0 and playerNumber == 1:
        acc *= acc

    acc = acc % 25000

    return ((sumPart + acc) << 6) | ((turnCnt + 1) & 63)

cdef inline int mapPlayerNumberTurnRel(Connect4_c* mnk, int playerNumber):
    cdef int tidx
    tidx = getPlayerOnTurnNumberMNK(mnk)

    if tidx == playerNumber:
        return 0
    else:
        return 1

cdef void fillNetworkInput0(Connect4GameData state, float[:, :, :, :] tensor, int batchIndex):
    cdef int x, y, b
    cdef Connect4_c* mnk = state._mnk

    # do not use the zero, as pytorch uses zero padding at the end of the input (=board) and
    # thus 0 is an input that encodes "end of the board"
    # 1 -> current player
    # 2 -> next player
    # 3 -> empty field
    cdef int[3] playerMappings = [3, mapPlayerNumberTurnRel(mnk, 1) + 1, mapPlayerNumberTurnRel(mnk, 2) + 1]
    
    for y in range(mnk.n):
        for x in range(mnk.m):
            b = readField(mnk.board, mnk.m, x, y)
            b = playerMappings[b]
            tensor[batchIndex, 0, y, x] = b

cdef class Connect4GameData():
    cdef Connect4_c* _mnk
    cdef unsigned int _hashVal
    cdef list _legalMovesList
    cdef int lastMove
    cdef char* placeHeights

    def __init__(self, unsigned char m, unsigned char n, unsigned char k, char doInit = 1):
        # the hash value of the empty game state is 0
        self._hashVal = 0
        self.lastMove = -1
        self._mnk = initConnect4(m, n, k)
        self.placeHeights = <char*> mallocWithZero(m * sizeof(char))

        if doInit:
            self._legalMovesList = list(range(m))

    def isEqual(self, Connect4GameData other):
        if self._mnk.turn != other._mnk.turn or self._mnk.m != other._mnk.m or self._mnk.n != other._mnk.n \
            or self._mnk.k != other._mnk.k or self._mnk.winningPlayer != other._mnk.winningPlayer:
            return False
        return areFieldsEqual(self._mnk.m, self._mnk.n, self._mnk.board, other._mnk.board)

    def mapPlayerNumberToTurnRelative(self, int number):
        return mapPlayerNumberTurnRel(self._mnk, number)

    def toString(self, networkMoves = None, networkWins = None, iteratedMoves = None, observedWins = None, replyMoves = None):
        mm = ['.', '░', '█']
        
        if self.lastMove != -1:
            lastMoveX = self.lastMove
            lastMoveY = 0
            while lastMoveY < self._mnk.n and readField(self._mnk.board, self._mnk.m, lastMoveX, lastMoveY) == 0:
                lastMoveY += 1
        else:
            lastMoveX = -500
            lastMoveY = -500
        
        m = self._mnk.m
        n = self._mnk.n
        k = self._mnk.k
        mnk = self._mnk
        s = "Connect4(%i,%i,%i), " %  (m, n, k)

        if not self.hasEnded():
            s += "Turn %i: %s\n" % (mnk.turn+1, mm[getPlayerOnTurnNumberMNK(mnk)])
        elif mnk.winningPlayer > 0:
            s += "Winner: %s\n" % mm[mnk.winningPlayer]
        else:
            s += "Draw\n"
        
        def getPDisplay(x, y):
            stone = readField(self._mnk.board, self._mnk.m, x, y)
            if x == lastMoveX and y == lastMoveY:
                return "│" + mm[stone] + "│ "
            else:
                return " " + mm[stone] + "  "
        
        for x in range(m):
            if x < 9:
                s += " %i" % (x+1)
            else:
                s += "%i" % (x+1)
            if x != m -1:
                s += " |"
        
        s += "\n";
        
        for y in range(n):
            for x in range(m+1):
                if x == lastMoveX and y == lastMoveY:
                    s += "┌─┐ "
                elif x == lastMoveX and y == lastMoveY+1:
                    s += "└─┘ "
                else:
                    s += "    "
            s += "\n";
            for x in range(m):
                s += getPDisplay(x, y);
            s += "\n";
        for _ in range(m+1):
            s += "    ";

        if networkMoves is not None or iteratedMoves is not None:
            s = "\n".join(map(lambda x: "        |" + x, s.split("\n")))

        s += "\n"

        def pFmt(p):
            mp = str(int(round(p * 100.0)))
            if len(mp) < 2:
                mp = " " + mp
            mp += "%"
            return mp

        if networkMoves is not None:
            nline = "Network |"
            for mi, m in enumerate(networkMoves):
                nline += pFmt(m)
                if len(networkMoves) - 1 > mi:
                    nline += "|"
            s += nline + "\n"

        if iteratedMoves is not None:
            iline = "Iterated|"
            for mi, m in enumerate(iteratedMoves):
                iline += pFmt(m)
                if len(iteratedMoves) - 1 > mi:
                    iline += "|"
            s += iline + "\n"

        if replyMoves is not None:
            iline = "Reply   |"
            for mi, m in enumerate(replyMoves):
                iline += pFmt(m)
                if len(replyMoves) - 1 > mi:
                    iline += "|"
            s += iline + "\n"

        if networkWins is not None:
            s += "Network win predictions: "
            for pi in range(3):
                s += mm[pi] + ":" + pFmt(networkWins[pi]) + "   "
            s += "\n"

        if observedWins is not None:
            s += "Observed results: "
            for o in observedWins:
                s += mm[o]
            s += "\n"

        return s;

    def getM(self):
        return self._mnk.m

    def getN(self):
        return self._mnk.n

    def getK(self):
        return self._mnk.k

    def getPlayerOnTurnNumber(self):
        return getPlayerOnTurnNumberMNK(self._mnk)

    def hasEnded(self):
        return self._mnk.winningPlayer != 255

    def _setWinnerNumber(self, num):
        self._mnk.winningPlayer = num

    def getWinnerNumber(self):
        return self._mnk.winningPlayer

    def getLegalMoves(self):
        return self._legalMovesList.copy()

    def getLastMove(self):
        return self.lastMove

    def _setLastMove(self, int lm):
        self.lastMove = lm

    def _setTurn(self, t):
        self._mnk.turn = t

    def getTurn(self):
        return self._mnk.turn

    def _setHash(self, h):
        self._hashVal = h

    def getHash(self):
        return self._hashVal

    def playMove(self, int legalMoveIndex):
        cdef int x, y

        x = legalMoveIndex

        assert self.placeHeights[x] < self._mnk.n

        y = self._mnk.n - self.placeHeights[x] - 1

        cdef Connect4GameData result = Connect4GameData(self._mnk.m, self._mnk.n, self._mnk.k, 0)
        copyConnect4(result._mnk, self._mnk)
        memcpy(result.placeHeights, self.placeHeights, self._mnk.m * sizeof(char))
        result._hashVal = updateMNKHash(self._hashVal, x, y, getPlayerOnTurnNumberMNK(self._mnk))

        placeStone(result._mnk, x, y)
        result.placeHeights[x] += 1

        result.lastMove = legalMoveIndex

        if y == 0:
            result._legalMovesList = list(filter(lambda cm: cm != legalMoveIndex, self._legalMovesList))
        else:
            result._legalMovesList = self._legalMovesList
        
        return result

    def _getBoardByte(self, int idx):
        return self._mnk.board[idx]

    def _setBoardByte(self, int idx, b):
        self._mnk.board[idx] = b

    def __dealloc__(self):
        freeConnect4(self._mnk)
        free(self.placeHeights)

class Connect4GameState(GameState, metaclass=abc.ABCMeta):
    """
    The game of Connect4. Typical size to play is m = 7 and n = 6
    """

    def __init__(self, m = 7, n = 6, k = 4, prepData = None):
        if prepData is None:
            self._data = Connect4GameData(m, n, k)
        else:
            self._data = prepData

    def getGameConstructorName(self):
        return "impls.games.connect4.connect4.Connect4GameState"

    def getGameConstructorParams(self):
        return {"m": self._data.getM(), "n": self._data.getN(), "k": self._data.getK()}

    def getM(self):
        return self._data.getM()
    
    def getN(self):
        return self._data.getN()

    def getK(self):
        return self._data.getK()

    def getGameName(self):
        assert(sizeof(unsigned int) == 4)
        return "Connect4(" + str(self._data.getM()) + "," + str(self._data.getN()) + "," + str(self._data.getK()) + ")"
    
    def getPlayerOnTurnNumber(self):
        return self._data.getPlayerOnTurnNumber()
    
    def hasEnded(self):
        return self._data.hasEnded()

    def getWinnerNumber(self):
        return self._data.getWinnerNumber()

    def getLegalMoves(self):
        return self._data.getLegalMoves()
    
    def getPlayerCount(self):
        return 2

    def getMoveCount(self):
        return self._data.getM()

    def playMove(self, int legalMoveIndex):
        return Connect4GameState(self._data.getM(), self._data.getN(), self._data.getK(), self._data.playMove(legalMoveIndex))

    def getTurn(self):
        return self._data.getTurn()

    def getDataShape(self):
        return (1, self._data.getN(), self._data.getM(), )

    def mapPlayerNumberToTurnRelative(self, int number):
        return self._data.mapPlayerNumberToTurnRelative(number)

    def encodeIntoTensor(self, object tensor, int batchIndex, augment):
        # augment is not implemented so far
        cdef float[:, :, :, :] tx = tensor
        fillNetworkInput0(self._data, tensor, batchIndex)

    def store(self):
        """
        encodes the state into a byte array with this layout:
        m,n,k,winningPlayer(1byte),hashval (4bytes),turns (4bytes),m*n field values
        """

        cdef unsigned int hashval = self._data.getHash()
        cdef int turn = self._data.getTurn()
        cdef int lastMove = self._data.getLastMove()
        cdef int idx
        cdef int fsize = self._data.getM() * self._data.getN()
        

        arSize = 16 + fsize
        result = np.zeros(arSize, dtype="uint8")
        
        result[0] = self._data.getM()
        result[1] = self._data.getN()
        result[2] = self._data.getK()
        result[3] = self._data.getWinnerNumber()
        result[4] = (hashval >> 24) & 255
        result[5] = (hashval >> 16) & 255
        result[6] = (hashval >> 8) & 255
        result[7] = hashval & 255
        result[8] = (turn >> 24) & 255
        result[9] = (turn >> 16) & 255
        result[10] = (turn >> 8) & 255
        result[11] = turn & 255
        result[12] = (lastMove >> 24) & 255
        result[13] = (lastMove >> 16) & 255
        result[14] = (lastMove >> 8) & 255
        result[15] = (lastMove >> 0) & 255
        for idx in range(fsize):
            result[16 + idx] = self._data._getBoardByte(idx)
        
        return result

    def load(self, encoded):
        cdef unsigned char m = encoded[0]
        cdef unsigned char n = encoded[1]
        cdef unsigned char k = encoded[2]
        cdef unsigned char winner = encoded[3]
        cdef unsigned char hash24 = encoded[4]
        cdef unsigned char hash16 = encoded[5]
        cdef unsigned char hash8 = encoded[6]
        cdef unsigned char hash0 = encoded[7]
        cdef unsigned char turn24 = encoded[8]
        cdef unsigned char turn16 = encoded[9]
        cdef unsigned char turn8 = encoded[10]
        cdef unsigned char turn0 = encoded[11]
        cdef unsigned char lastMove24 = encoded[12]
        cdef unsigned char lastMove16 = encoded[13]
        cdef unsigned char lastMove8 = encoded[14]
        cdef unsigned char lastMove0 = encoded[15]

        cdef unsigned int hv = (hash24 << 24) | (hash16 << 16) | (hash8 << 8) | hash0
        cdef int t = (turn24 << 24) | (turn16 << 16) | (turn8 << 8) | turn0
        cdef int lastMove = (lastMove24 << 24) | (lastMove16 << 16) | (lastMove8 << 8) | (lastMove0 << 0)

        result = Connect4GameState(m, n, k)
        result._data._setWinnerNumber(winner)
        result._data._setHash(hv)
        result._data._setTurn(t)
        result._data._setLastMove(lastMove)

        cdef int fsize = m * n
        for idx in range(fsize):
            result._data._setBoardByte(idx, encoded[16 + idx])
        
        return result

    def __eq__(self, other):
        return self._data.isEqual(other._data)

    def __hash__(self):
        return self._data.getHash()

    def prettyString(self, networkMoves, networkWins, iteratedMoves, observedWins, replyMoves=None):
        return self._data.toString(networkMoves=networkMoves, networkWins=networkWins, iteratedMoves=iteratedMoves, observedWins=observedWins, replyMoves=replyMoves)

    def __str__(self):
        return self._data.toString()
