"""
Sutton and Barto from Scratch 4: Planning and Policy-Gradient Methods

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_maze
def build_maze(walls=None, start=(2, 0), goal=(0, 8), rows=6, cols=9):
    # Default wall locations from Sutton and Barto's Figure 8.2.
    if walls is None:
        walls = [
            (1, 2),
            (2, 2),
            (3, 2),
            (4, 5),
            (0, 7),
            (1, 7),
            (2, 7),
        ]

    # Convert the supplied wall coordinates into a set of tuples.
    walls = {tuple(cell) for cell in walls}

    # Return the maze configuration.
    return {
        'rows': int(rows),
        'cols': int(cols),
        'start': tuple(start),
        'goal': tuple(goal),
        'walls': walls,
        'n_actions': 4,  # 0=up, 1=right, 2=down, 3=left
    }

# Step 2 - maze_step
def maze_step(maze, state, action):
    # Define the movement directions:
    # 0 = up, 1 = right, 2 = down, 3 = left.
    directions = {
        0: (-1, 0),
        1: (0, 1),
        2: (1, 0),
        3: (0, -1),
    }

    # Calculate the target cell for the chosen action.
    row, col = state
    dr, dc = directions[action]
    target = (row + dr, col + dc)

    # Check whether the target is outside the maze boundaries
    # or blocked by a wall. If so, the agent stays in place.
    if (
        target[0] < 0
        or target[0] >= maze['rows']
        or target[1] < 0
        or target[1] >= maze['cols']
        or target in maze['walls']
    ):
        next_state = state
    else:
        next_state = target

    # The agent receives a reward of 1.0 at the goal and 0.0 otherwise.
    done = next_state == maze['goal']
    reward = 1.0 if done else 0.0

    # Return the next state, reward, and terminal status.
    return next_state, reward, done

# Step 3 - epsilon_greedy_tabular
import numpy as np

def epsilon_greedy_tabular(q_table, state, epsilon, rng):
    # Get the action-value estimates for the given state.
    action_values = q_table[state]
    n_actions = len(action_values)

    # Draw a random number to decide whether to explore or exploit.
    u = rng.random()

    if u < epsilon:
        # Exploration: choose an action uniformly at random.
        return int(rng.integers(0, n_actions))

    # Exploitation: identify all actions with the maximum Q-value.
    max_value = np.max(action_values)
    candidates = np.flatnonzero(action_values == max_value)

    # Break ties uniformly at random among the greedy actions.
    return int(rng.choice(candidates))

# Step 4 - q_learning_update
def q_learning_update(q_table, state, action, reward, next_state, alpha, gamma):
    # Create a copy so the original Q-table remains unchanged.
    updated_q_table = q_table.copy()

    # Get the current action-value estimate Q(s, a).
    current_value = q_table[state][action]

    # Compute the Q-learning target:
    # reward + gamma * max_a Q(next_state, a)
    target = reward + gamma * max(q_table[next_state])

    # Apply the tabular Q-learning update.
    updated_q_table[state][action] = (
        current_value + alpha * (target - current_value)
    )

    # Return the updated copy of the Q-table.
    return updated_q_table

# Step 5 - planning_updates (not yet solved)
# TODO: implement

# Step 6 - dyna_q_episode (not yet solved)
# TODO: implement

# Step 7 - greedy_path_length (not yet solved)
# TODO: implement

# Step 8 - update_model_plus (not yet solved)
# TODO: implement

# Step 9 - dyna_q_plus_planning (not yet solved)
# TODO: implement

# Step 10 - run_blocking_maze (not yet solved)
# TODO: implement

# Step 11 - push_priority (not yet solved)
# TODO: implement

# Step 12 - prioritized_sweeping_episode (not yet solved)
# TODO: implement

# Step 13 - corridor_step (not yet solved)
# TODO: implement

# Step 14 - corridor_policy (not yet solved)
# TODO: implement

# Step 15 - corridor_true_value (not yet solved)
# TODO: implement

# Step 16 - reinforce_episode (not yet solved)
# TODO: implement

# Step 17 - reinforce_baseline_episode (not yet solved)
# TODO: implement

# Step 18 - actor_critic_corridor_episode (not yet solved)
# TODO: implement

# Step 19 - access_control_step (not yet solved)
# TODO: implement

# Step 20 - differential_sarsa_update (not yet solved)
# TODO: implement

# Step 21 - run_access_control (not yet solved)
# TODO: implement

# Step 22 - access_control_policy_table (not yet solved)
# TODO: implement

