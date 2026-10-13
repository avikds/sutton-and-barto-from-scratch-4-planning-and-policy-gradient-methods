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

# Step 5 - planning_updates
def planning_updates(q_table, model, n_planning, alpha, gamma, rng):
    # Return an unchanged copy if there are no modelled transitions
    # or no planning updates are requested.
    if not model or n_planning <= 0:
        return q_table.copy()

    # Preserve the model's insertion order when creating the key list.
    keys = list(model.keys())

    # Sample all planning indices in a single random-number-generator call.
    indices = rng.integers(0, len(keys), size=n_planning)

    # Work on a copy so the input Q-table is never mutated.
    updated_q_table = q_table.copy()

    # Perform simulated Q-learning updates using the learned model.
    for index in indices:
        state, action = keys[int(index)]
        next_state, reward = model[(state, action)]

        updated_q_table = q_learning_update(
            updated_q_table,
            state,
            action,
            reward,
            next_state,
            alpha,
            gamma,
        )

    return updated_q_table

# Step 6 - dyna_q_episode
def dyna_q_episode(maze, q_table, model, n_planning, alpha, gamma, epsilon, rng):
    # Copy the model so the caller's dictionary remains unchanged.
    model = model.copy()

    # Begin each episode at the maze's start position.
    state = maze['start']
    done = False
    n_steps = 0

    # Continue until the agent reaches the goal.
    while not done:
        # (a) Select an action using epsilon-greedy exploration.
        action = epsilon_greedy_tabular(q_table, state, epsilon, rng)

        # (b) Execute the action in the real environment.
        next_state, reward, done = maze_step(maze, state, action)

        # (c) Update the Q-table using the real experience.
        q_table = q_learning_update(
            q_table, state, action, reward, next_state, alpha, gamma
        )

        # (d) Learn the deterministic transition model from experience.
        model[(state, action)] = (next_state, reward)

        # (e) Perform simulated Q-learning backups using the learned model.
        q_table = planning_updates(
            q_table, model, n_planning, alpha, gamma, rng
        )

        # Move to the next state and count the real environment step.
        state = next_state
        n_steps += 1

    # Return the updated Q-table, copied model, and real episode length.
    return q_table, model, n_steps

# Step 7 - greedy_path_length
def greedy_path_length(maze, q_table, max_steps=100):
    # Start at the maze's initial position.
    state = maze['start']

    # Follow the greedy policy for at most max_steps moves.
    for step in range(max_steps):
        # Select the action with the highest Q-value.
        # np.argmax chooses the smallest action index when values are tied.
        action = int(np.argmax(q_table[state]))

        # Execute the selected action in the maze.
        next_state, reward, done = maze_step(maze, state, action)

        # Return the number of moves taken if the goal is reached.
        if done:
            return step + 1

        # Update the current state.
        state = next_state

    # Return max_steps if the goal was not reached within the limit.
    return max_steps

# Step 8 - update_model_plus
def update_model_plus(model, state, action, next_state, reward, time, n_actions=4):
    # Copy the model so the input dictionary remains unchanged.
    updated_model = model.copy()

    # Add default transitions for actions not yet tried from this state.
    # Each untried action leads back to the same state with zero reward
    # and an initial last-visit time of 0.
    for a in range(n_actions):
        key = (state, a)
        if key not in updated_model:
            updated_model[key] = (state, 0.0, 0)

    # Record the actual observed transition and its latest visit time.
    updated_model[(state, action)] = (next_state, reward, int(time))

    return updated_model

# Step 9 - dyna_q_plus_planning
def dyna_q_plus_planning(q_table, model, n_planning, time, kappa, alpha, gamma, rng):
    # Return an unchanged copy if the model is empty or no planning is requested.
    if not model or n_planning <= 0:
        return q_table.copy()

    # Preserve the insertion order of the model entries.
    keys = list(model.keys())

    # Sample all planning indices in a single RNG call.
    indices = rng.integers(0, len(keys), size=n_planning)

    # Work on a copy so the input Q-table is never mutated.
    updated_q_table = q_table.copy()

    # Perform simulated Q-learning backups with the Dyna-Q+ exploration bonus.
    for index in indices:
        state, action = keys[int(index)]
        next_state, reward, last_time = model[(state, action)]

        # Encourage exploration of state-action pairs not visited recently.
        bonus = kappa * np.sqrt(time - last_time)
        augmented_reward = reward + bonus

        # Update the Q-value using the augmented reward.
        updated_q_table = q_learning_update(
            updated_q_table,
            state,
            action,
            augmented_reward,
            next_state,
            alpha,
            gamma,
        )

    return updated_q_table

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

