# Import required libraries
import numpy as np
from mdptoolbox import mdp

# The number of rows on the game board
BOARD_ROWS = 5
# The number of columns on the game board
BOARD_COLS = 5
# The number of states is the product of board rows and board columns
STATES = BOARD_ROWS * BOARD_COLS
# The number of actions which equals two. Go up or go right.
ACTIONS = 2
ACTION_UP = 0
ACTION_RIGHT = 1
# Probability of making the desired move
Pr = 0.75

def check_action(a):
    """Check that the action is in the valid range."""
    if not (0 <= a < ACTIONS):
        msg = "Invalid action '%s', it should be in {0, 1}." % str(a)
        raise ValueError(msg)

def check_rows(r):
    """Check that the number of rows is in the valid range."""
    if not (0 <= r < BOARD_ROWS):
        msg = "Invalid row number '%s', it should be in {0, 1, …, %d}." \
              % (str(r), BOARD_ROWS - 1)
        raise ValueError(msg)

def check_columns(c):
    """Check that the number of columns is in the valid range."""
    if not (0 <= c < BOARD_COLS):
        msg = "Invalid column number '%s', it should be in {0, 1, …, %d}." % \
              (str(c), BOARD_COLS - 1)
        raise ValueError(msg)

def convert_state_to_index(row, column):
    """Convert state parameters to transition probability matrix index.

    Parameters
    ----------
    row : int
        The row on the game board.
    column : int
        The column on the game board.

    Returns
    -------
    index : int
        The index into the transition probability matrix that corresponds to
        the state parameters.
    """
    check_rows(row)
    check_columns(column)
    return row * BOARD_COLS + column

def convert_index_to_state(index):
    """Convert transition probability matrix index to state parameters.

    Parameters
    ----------
    index : int
        The index into the transition probability matrix that corresponds to
        the state parameters.

    Returns
    -------
    row, column : tuple of int
        ``row``, the row on the game board
        ``column``, the column on the game board
    """
    if not (0 <= index < STATES):
        msg = "Invalid index '%s', it should be in {0, 1, …, %d}." % \
              (str(index), STATES - 1)
        raise ValueError(msg)
    row = index // BOARD_COLS
    column = index % BOARD_COLS
    return (row, column)

def get_transition_probabilities(r, c, a):
    """Calculate the transition probabilities for the given state and action.

    Parameters
    ----------
    r : int
        The row on the game board.
    c : int
        The column on the game board.
    a : int
        The action undertaken.

    Returns
    -------
    prob : array
        The transition probabilities as a vector from state (``r``, ``c``) to
        every other state given that action ``a`` is taken.
    """
    # Check that input is in range
    check_rows(r)
    check_columns(c)
    check_action(a)

    # a vector to store the transition probabilities
    prob = np.zeros(STATES)

    # New column (row) for movement right (up)
    if c == BOARD_COLS - 1:
        right_col = c
    else:
        right_col = c + 1
    if r == BOARD_ROWS - 1:
        up_row = r
    else:
        up_row = r + 1
    # Convert new row & column into proper state
    right_state = convert_state_to_index(r, right_col)
    up_state = convert_state_to_index(up_row, c)
    # Fill in correctly based on value of Action
    if up_state == right_state:
        prob[up_state] = 1
    elif a == 0:
        prob[up_state] = Pr
        prob[right_state] = 1 - Pr
    else:
        prob[right_state] = Pr
        prob[up_state] = 1 - Pr
    # Make sure that the probabilities sum to one
    assert (prob.sum() - 1) < np.spacing(1)
    return prob

def get_transition_and_reward_arrays():
    """Generate the transition and reward matrices.
    """
    # The transition probability array
    transition = np.zeros((ACTIONS, STATES, STATES))
    # The reward vector
    reward = np.zeros(STATES)
    # Loop over all states
    for idx in range(STATES):
        # Convert the state index into rows & columns of transition matrix
        r, c = convert_index_to_state(idx)
        # Set up rewoard matrix.  +2 if in upper right corner of board, else -0.05.
        if r == BOARD_ROWS - 1 and c == BOARD_COLS - 1:
            reward[idx] = 2 # Could be adjusted for other scenarios
        elif r == BOARD_ROWS - 2 and c == BOARD_COLS - 1:
            reward[idx] = -0.05 # Could be adjusted for other scenarios
        else:
            reward[idx] = -0.05 # Could be adjusted for other scenarios
        # Loop over all actions
        for a in range(ACTIONS):
            # Assign the transition probabilities for this state, action pair
            transition[a][idx] = get_transition_probabilities(r, c, a)
    return (transition, reward)

def solve_mdp_value():
    """Solve the problem as a value iteration Markov decision process.
    """
    P, R = get_transition_and_reward_arrays()
    sdp = mdp.ValueIteration(P, R, 0.96, epsilon = 0.01, max_iter = 1000)
    sdp.run()
    return sdp

def solve_mdp_policy():
    """Solve the problem as a policy iteration Markov decision process.
    """
    P, R = get_transition_and_reward_arrays()
    sdp = mdp.PolicyIteration(P, R, 0.96, policy0 = None, max_iter = 1000)
    sdp.run()
    return sdp

def solve_rl_Qlearn():
    """Solve the problem as a reinforcement learning (Q-learning) algorithm.
    """
    np.random.seed(1)
    P, R = get_transition_and_reward_arrays()
    rlq = mdp.QLearning(P, R, 0.96, n_iter = 10000)
    rlq.run()
    return rlq

def print_policy(policy):
    """Print out a policy vector as a table to console

    Let ``S`` = number of states.

    The output is a table that has the board rows as rows, and the board
    columns as the columns. The items in the table are the optimal action
    for that board row and board column placement on the game board.

    Parameters
    ----------
    p : array
        ``p`` is a numpy array of length ``S``.

    """
    p = np.array(policy).reshape(BOARD_ROWS, BOARD_COLS)
    print("    " + " ".join("%2d" % f for f in range(BOARD_COLS)))
    print("    " + "---" * BOARD_COLS)
    for x in range(BOARD_ROWS - 1, -1, -1):
        print(" %2d|" % x + " ".join("%2d" % p[x, f] for f in
                                     range(BOARD_COLS)))
        
sdp_val = solve_mdp_value()
print_policy(sdp_val.policy)
print()
print("Iterations = " + str(sdp_val.iter))
print("Time = " + str(sdp_val.time))

sdp_pol = solve_mdp_policy()
print_policy(sdp_pol.policy)
print()
print("Iterations = " + str(sdp_pol.iter))
print("Time = " + str(sdp_pol.time))

rlq_Qlearn = solve_rl_Qlearn()
print_policy(rlq_Qlearn.policy)
print(rlq_Qlearn.Q)
