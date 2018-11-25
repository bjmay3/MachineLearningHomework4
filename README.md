All coding done in Python.  Code files can be found at the following link:  https://github.com/bjmay3/MachineLearningHomework4

Two (2) separate Python code files exist.  These are as follows:
	a. BoardGame - MDP and RL solutions for the Board Game scenario (small number of states).
	b. ControlledBurn - MDP and RL solutions for the Controlled Burn scenario (large number of states).
	
The Board Game scenario requires the following global variables:
	a. BOARD_ROWS - The number of rows on the game board
	b. BOARD_COLS - The number of columns on the game board
	c. STATE - The number of states on the board. Equals the product of board rows and board columns
	d. ACTIONS - The number of actions that can be used in the game
	e. ACTION_UP - Value that determines that the action is to move up
	f. ACTION_RIGHT - Value that determines the action is to move right
	g. Other action variables are necessary if more than two actions are selected
	h. Pr - Probability of making the desired move. (1 - Pr) is the probability of inadvertently making the opposite move

The Board Game scenario contains the following defined functions:
	a. def check_action(a) - Receives an action value and determines whether or not it exists within its allowed range
	b. def check_rows(r) - Receives a number of rows and determines whether or not it exists within its allowed range
	c. def check_columns(c) - Receives a number of columns and determines whether or not it exists within its allowed range
	e. def convert_state_to_index(row, column) - Receives a row & column value and converts them to a matrix index number
	f. def convert_index_to_state(index) - Receives a matrix index number and converts it to its corresponding row & column values
	g. def get_transition_probabilities(r, c, a) - Receives a row, column, and action and calculates the transition probabilities   associated with the state defined by the row and column and the corresponding action
	h. def get_transition_and_reward_arrays() - Generates the transition and reward matrices
	i. def solve_mdp_value() - Solves the problem as a value iteration Markov decision process
	j. def solve_mdp_policy() - Solves the problem as a policy iteration Markov decision process
	k. def solve_rl_Qlearn() - Solves the problem as a reinforcement learning (Q-learning) algorithm
	l. def print_policy(policy) - Prints out a policy vector as a table of rows vs. columns

The Controlled Burn scenario requires the following global variables:
	a. POPULATION_CLASSES - The number of population abundance classes
	b. FIRE_CLASSES - The number of years since a fire classes
	c. STATES - The number of states in the problem.  Equals the product of Population Classes & Fire Classes
	d. ACTIONS - The number of actions which are allowed in the problem
	e. ACTION_NOTHING - The value that defines the action of doing nothing
	f. ACTION_BURN - The value that defines the action of performing a burn

The Controlled Burn scenario contains the following defined functions:
	a. def check_action(x) - Receives an action value and determines whether or not it exists within its allowed range
	b. def check_population_class(x) - Receives a population abundance class value and checks that it is in the valid range
	c. def check_fire_class(x) - Receives a fire class value and checks that it is in the valid range
	d. def check_probability(x, name="probability") - Receives a probability value and checks that it is between 0 and 1
	e. def get_habitat_suitability(years) - Receives some number of years and calculates the habitat suitability relatve to the time since last fire
	f. def convert_state_to_index(population, fire) - Receives a population and a fire class and converts them to a transition probability matrix index
	g. def convert_index_to_state(index) - Receives a transition probability matrix index and converts it to corresponding population & fire classes
	h. def transition_fire_state(F, a) - Receives a fire class and action and transitions the years since last fire based on the action taken
	i. def get_transition_probabilities(s, x, F, a) - Receives probability of staying in current state, population class, fire class, and action and calculates the transition probabilities for the given state and action
	j. def get_transition_and_reward_arrays(s) - Receives the probability of staying in current state and generates the fire management transition and reward matrices
	k. def solve_mdp_value() - Solves the problem as a value iteration Markov decision process
	l. def solve_mdp_policy() - Solves the problem as a policy iteration Markov decision process
	m. def solve_rl_Qlearn() - Solves the problem as a reinforcement learning (Q-learning) algorithm
	n. def print_policy(policy) - Prints out a policy vector as a table of population classes vs. fire classes

Attribution:  Some code and ideas were "borrowed" from elsewhere.  The following gives credit to the places where various pieces of code & ideas were obtained.
	a. "Application of Stochastic Dynamic Programming to Optimal Fire Management of a Spatially Structured Threatened Species" by Hugh Possingham
	b. "Optimal fire management of a threatened species, part 1 - Python MDP Toolbox worked example" by Steven AW Cordwell

Attribution Websites
	a. Possingham paper - http://www.mssanz.org.au/MODSIM97/Vol%202/Possingham.pdf
	b. Cordwell coding & analysis - http://sawcordwell.github.io/mdp/conservation/2015/01/10/possingham1997-1/

