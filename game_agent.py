"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    max_player = player
    min_player = game.get_opponent(player)
    return next_move_heuristics(game, max_player, min_player)
    # blocking_heuristics(game, max_player, min_player)
    # mirroring_heuristics(game, max_player, min_player)

def mirroring_heuristics(game, max_player, min_player):
    if can_mirror(game, max_player, min_player):
        return 10.0  # More Than Possible Moves (8)
    else:
        return improved_heuristics(game, max_player, min_player)


def can_mirror(game, max_player, min_player):
    max_player_location = game.get_player_location(max_player)
    min_player_location = game.get_player_location(min_player)
    x = max_player_location[0] + min_player_location[0] - 6
    y = max_player_location[1] + min_player_location[1] - 6
    if x == 0 and y == 0:
        return True
    else:
        return False


def improved_heuristics(game, max_player, min_player):
    max_player_num_moves = len(game.get_legal_moves(max_player))
    min_player_num_moves = len(game.get_legal_moves(min_player))
    return max_player_num_moves - min_player_num_moves


def calculate_possible_moves(player_location):
    sign_array = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    shape_array = [(1, 2), (2, 1)]
    moves = []
    for signs in sign_array:
        for shapes in shape_array:
            multiple = [signs[i]*shapes[i] for i in range(2)]
            multiple = [multiple[i] + player_location[i] for i in range(2)]
            moves.append(tuple(multiple))
    return moves


def blocking_heuristics(game, max_player, min_player):
    player_location = game.get_player_location(max_player)
    min_player_location = game.get_player_location(min_player)
    possible_moves_for_min_player = set(calculate_possible_moves(min_player_location))
    if player_location in possible_moves_for_min_player:
        return 1.0
    return 0.0


def next_move_heuristics(game, max_player, min_player):
    possible_moves = set([])
    blank_spaces = set(game.get_blank_spaces())
    for move in game.get_legal_moves(max_player):
        current_possible_moves = set(calculate_possible_moves(move))
        possible_moves = possible_moves.union(current_possible_moves)
    possible_moves = possible_moves.intersection(blank_spaces)
    possible_moves = possible_moves.difference(game.get_legal_moves(min_player))
    return float(len(possible_moves))


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left
        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        result = (-1, -1)
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative:
                depth = 1
                while True:
                    if self.method is "alphabeta":
                        result = self.alphabeta(game, depth)[1]
                    if self.method is "minimax":
                        result = self.minimax(game, depth)[1]
                    depth += 1
            else:
                if self.method is "alphabeta":
                    result = self.alphabeta(game, self.search_depth)[1]
                if self.method is "minimax":
                    result = self.minimax(game, self.search_depth)[1]
            return result
        except Timeout:
            # Handle any actions required at timeout, if necessary
            return result
        # Return the best move from the last completed search iteration

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        # Base Case: Depth is Zero get Number of Legal Moves (Scoring is Based on This Number)
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        max_player = game.__active_player__ if maximizing_player else game.get_opponent(game.__active_player__)
        legal_moves_list = []
        if len(game.get_legal_moves()) == 0:
            return game.utility(max_player), (-1, -1)
        if depth <= 0:
            return self.score(game, max_player), (-1, -1)
        for x, y in game.get_legal_moves():
            current_game = game.forecast_move((x, y))
            score = self.minimax(current_game, depth-1, not maximizing_player)
            move = (x, y)
            legal_moves_list.append((score[0], move))
        if maximizing_player:
            return max(legal_moves_list, key=lambda t: t[0])
        else:
            return min(legal_moves_list, key=lambda t: t[0])

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        max_player = game.__active_player__ if maximizing_player else game.get_opponent(game.__active_player__)
        if len(game.get_legal_moves()) == 0: # TODO: Move This Logic into Score Function
            return game.utility(max_player), (-1, -1)
        if depth <= 0:
            return self.score(game, max_player), (-1, -1)
        if maximizing_player:
            min_value = (float("-inf"), (-1, -1))
            for x, y in game.get_legal_moves():
                current_game = game.forecast_move((x, y))
                min_value = max([min_value, (self.alphabeta(current_game, depth - 1, alpha, beta, not maximizing_player)[0], (x, y))], key=lambda t: t[0])
                if min_value[0] >= beta:
                    return min_value
                alpha = max(alpha, min_value[0])
            return min_value
        else:
            max_value = (float("inf"), (-1, -1))
            for x, y in game.get_legal_moves():
                current_game = game.forecast_move((x, y))
                max_value = min([max_value, (self.alphabeta(current_game, depth - 1, alpha, beta, not maximizing_player)[0], (x, y))], key=lambda t: t[0])
                if max_value[0] <= alpha:
                    return max_value
                beta = min(beta, max_value[0])
            return max_value
