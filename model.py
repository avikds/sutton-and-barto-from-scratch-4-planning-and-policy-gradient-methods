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

# Step 10 - run_blocking_maze
def run_blocking_maze(use_plus, max_steps, change_step, n_planning, kappa, alpha, gamma, epsilon, rng):
    # Build the initial 6x9 maze with a wall that blocks row 3
    # except at column 8.
    maze = build_maze(
        walls=[(3, c) for c in range(8)],
        start=(5, 3),
        goal=(0, 8),
        rows=6,
        cols=9,
    )

    # Initialize the action-value table, learned model, and agent state.
    q_table = np.zeros((6, 9, 4))
    model = {}
    state = maze['start']

    # Store cumulative rewards after every real environment step.
    cumulative = np.zeros(max_steps, dtype=float)
    total_reward = 0.0

    for t in range(1, max_steps + 1):
        # Change the wall after the first change_step steps.
        # Keep the current state, Q-table, and learned model unchanged.
        if t == change_step + 1:
            maze = build_maze(
                walls=[(3, c) for c in range(1, 9)],
                start=(5, 3),
                goal=(0, 8),
                rows=6,
                cols=9,
            )

        # Select an action using epsilon-greedy exploration.
        action = epsilon_greedy_tabular(q_table, state, epsilon, rng)

        # Execute one real transition in the current maze.
        next_state, reward, done = maze_step(maze, state, action)

        # Update the Q-table using the real experience.
        q_table = q_learning_update(
            q_table,
            state,
            action,
            reward,
            next_state,
            alpha,
            gamma,
        )

        if use_plus:
            # Record the transition, initialize untried actions,
            # and timestamp the observed state-action pair.
            model = update_model_plus(
                model,
                state,
                action,
                next_state,
                reward,
                t,
                4,
            )

            # Perform Dyna-Q+ planning with the exploration bonus.
            q_table = dyna_q_plus_planning(
                q_table,
                model,
                n_planning,
                t,
                kappa,
                alpha,
                gamma,
                rng,
            )
        else:
            # Learn the deterministic model from real experience.
            model[(state, action)] = (next_state, reward)

            # Perform ordinary Dyna-Q planning without an exploration bonus.
            q_table = planning_updates(
                q_table,
                model,
                n_planning,
                alpha,
                gamma,
                rng,
            )

        # Accumulate rewards and record the cumulative reward curve.
        total_reward += reward
        cumulative[t - 1] = total_reward

        # Reset to the start after reaching the goal; otherwise continue.
        if done:
            state = maze['start']
        else:
            state = next_state

    return cumulative

# Step 11 - push_priority
import heapq

def push_priority(queue, q_table, state, action, reward, next_state, gamma, theta):
    # Compute the Q-learning target for the observed transition.
    target = reward + gamma * max(q_table[next_state])

    # Calculate the magnitude of the pending Q-value update.
    priority = abs(target - q_table[state][action])

    # Add the state-action pair only when its priority exceeds theta.
    if priority > theta:
        heapq.heappush(
            queue,
            (-float(priority), tuple(state), int(action))
        )

    # Return the same queue object.
    return queue

# Step 12 - prioritized_sweeping_episode
def prioritized_sweeping_episode(maze, q_table, model, n_planning, theta, alpha, gamma, epsilon, rng):
    # Copy the model so the caller's dictionary is not mutated.
    model = model.copy()

    # Initialize the priority queue and episode counters.
    queue = []
    state = maze['start']
    done = False
    n_steps = 0
    n_updates = 0

    # Continue until the agent reaches the goal.
    while not done:
        # Select an action using epsilon-greedy exploration.
        action = epsilon_greedy_tabular(q_table, state, epsilon, rng)

        # Execute the action in the real maze.
        next_state, reward, done = maze_step(maze, state, action)

        # Store the observed transition in the deterministic model.
        model[(state, action)] = (next_state, reward)

        # Queue the real transition according to its update magnitude.
        push_priority(
            queue, q_table, state, action,
            reward, next_state, gamma, theta
        )

        # Perform up to n_planning prioritized backups.
        for _ in range(n_planning):
            # Stop planning if the queue is empty.
            if not queue:
                break

            # Pop the transition with the largest priority.
            _, s, a = heapq.heappop(queue)

            # Retrieve the transition from the learned model.
            s_next, r = model[(s, a)]

            # Apply one Q-learning backup and count the update.
            q_table = q_learning_update(
                q_table, s, a, r, s_next, alpha, gamma
            )
            n_updates += 1

            # Find predecessor transitions leading into state s.
            # Reprioritize them because the value of s may have changed.
            for (ps, pa), (pn, pr) in model.items():
                if pn == s:
                    push_priority(
                        queue, q_table, ps, pa,
                        pr, s, gamma, theta
                    )

        # Advance the real agent and count its environment step.
        state = next_state
        n_steps += 1

    # Return the learned Q-table, copied model, real steps, and planning updates.
    return q_table, model, n_steps, n_updates

# Step 13 - corridor_step
def corridor_step(state, action):
    # Reverse the action in state 1, as specified in Sutton and Barto's Example 13.1.
    if state == 1:
        action = 1 - action

    # Action 0 moves left; action 1 moves right.
    if action == 0:
        # Moving left from state 0 leaves the agent in state 0.
        next_state = max(0, state - 1)
    else:
        # Moving right advances one state, including to terminal state 3.
        next_state = state + 1

    # Every transition incurs a reward of -1.0.
    reward = -1.0
    done = next_state == 3

    return next_state, reward, done

# Step 14 - corridor_policy
def corridor_policy(theta):
    # Subtract the maximum preference for numerical stability.
    shifted_theta = theta - np.max(theta)

    # Compute the exponentials of the shifted preferences.
    exp_theta = np.exp(shifted_theta)

    # Normalize the exponentials to obtain softmax action probabilities.
    probs = exp_theta / np.sum(exp_theta)

    return probs.astype(float)

# Step 15 - corridor_true_value
def corridor_true_value(p_right):
    # Compute the exact start-state value using the closed-form solution
    # for Sutton and Barto's short corridor (Example 13.1).
    value = -2.0 * (2.0 - p_right) / (p_right * (1.0 - p_right))

    return float(value)

# Step 16 - reinforce_episode
def reinforce_episode(theta, alpha, gamma, rng, max_steps=10000):
    # Copy the preferences so the input array remains unchanged.
    theta = theta.copy()

    # Generate one episode, storing the actions and rewards.
    state = 0
    actions = []
    rewards = []
    done = False

    while not done and len(actions) < max_steps:
        # Sample an action from the current softmax policy.
        probs = corridor_policy(theta)
        action = int(rng.choice(2, p=probs))

        # Take one transition in the short corridor.
        next_state, reward, done = corridor_step(state, action)

        actions.append(action)
        rewards.append(reward)
        state = next_state

    # Compute Monte Carlo returns using a backward pass.
    returns = [0.0] * len(rewards)
    G = 0.0

    for t in range(len(rewards) - 1, -1, -1):
        G = rewards[t] + gamma * G
        returns[t] = G

    # Apply REINFORCE updates in chronological order.
    # Recompute the policy after each update using the current preferences.
    for t, (action, G_t) in enumerate(zip(actions, returns)):
        probs = corridor_policy(theta)

        # Score function: one-hot(action) - policy probabilities.
        grad = np.zeros(2, dtype=float)
        grad[action] = 1.0
        grad -= probs

        # Apply the gamma^t-weighted policy-gradient update.
        theta += alpha * (gamma ** t) * G_t * grad

    # Return the updated preferences and undiscounted episode reward.
    total_reward = float(sum(rewards))
    return theta, total_reward

# Step 17 - reinforce_baseline_episode
def reinforce_baseline_episode(theta, w, alpha_theta, alpha_w, gamma, rng, max_steps=10000):
    # Copy the policy preferences so the input array remains unchanged.
    theta = theta.copy()
    w = float(w)

    # Generate one episode, storing actions and rewards.
    state = 0
    actions = []
    rewards = []
    done = False

    while not done and len(actions) < max_steps:
        # Sample an action from the current softmax policy.
        probs = corridor_policy(theta)
        action = int(rng.choice(2, p=probs))

        # Execute one transition in the short corridor.
        next_state, reward, done = corridor_step(state, action)

        actions.append(action)
        rewards.append(reward)
        state = next_state

    # Compute Monte Carlo returns using a backward pass.
    returns = [0.0] * len(rewards)
    G = 0.0

    for t in range(len(rewards) - 1, -1, -1):
        G = rewards[t] + gamma * G
        returns[t] = G

    # Update the baseline and policy in chronological order.
    for t, (action, G_t) in enumerate(zip(actions, returns)):
        # Compute the advantage using the current baseline estimate.
        delta = G_t - w

        # Update the state-value baseline.
        w += alpha_w * delta

        # Recompute action probabilities using the current preferences.
        probs = corridor_policy(theta)

        # Compute the policy score: one-hot(action) - probabilities.
        grad = np.zeros(2, dtype=float)
        grad[action] = 1.0
        grad -= probs

        # Apply the gamma^t-weighted policy-gradient update.
        theta += alpha_theta * (gamma ** t) * delta * grad

    # Return updated preferences, baseline, and undiscounted episode reward.
    total_reward = float(sum(rewards))
    return theta, float(w), total_reward

# Step 18 - actor_critic_corridor_episode
def actor_critic_corridor_episode(theta, w, alpha_theta, alpha_w, gamma, rng, max_steps=10000):
    # Copy the actor and critic parameters so the input arrays remain unchanged.
    theta = theta.copy()
    w = w.copy()

    # Start at the initial corridor state.
    state = 0
    discount = 1.0
    total_reward = 0.0
    done = False
    n_steps = 0

    # Run one episode, subject to the safety limit.
    while not done and n_steps < max_steps:
        # Select an action using the actor's softmax policy.
        probs = corridor_policy(theta)
        action = int(rng.choice(2, p=probs))

        # Execute one transition in the corridor.
        next_state, reward, done = corridor_step(state, action)

        # The terminal state's value is zero.
        next_value = 0.0 if done else w[next_state]

        # Compute the one-step TD error.
        delta = reward + gamma * next_value - w[state]

        # Update the critic for the current state.
        w[state] += alpha_w * delta

        # Compute the policy-gradient score using the sampled policy.
        grad = np.zeros(2, dtype=float)
        grad[action] = 1.0
        grad -= probs

        # Update the actor, weighting the gradient by the accumulated discount.
        theta += alpha_theta * discount * delta * grad

        # Update the accumulated discount and episode statistics.
        discount *= gamma
        total_reward += reward
        n_steps += 1

        # Advance to the next state.
        state = next_state

    # Return the updated actor, critic, and undiscounted episode reward.
    return theta, w, float(total_reward)

# Step 19 - access_control_step
def access_control_step(state, action, rng, n_servers=10, free_prob=0.06):
    # Unpack the state: number of free servers and current customer priority.
    free, prio = state
    free = int(free)

    # Customer priorities from Sutton and Barto's Example 10.2.
    priorities = (1.0, 2.0, 4.0, 8.0)

    # Accept the customer only if action is 1 and a server is available.
    if action == 1 and free > 0:
        reward = priorities[prio]
        free -= 1
    else:
        # Rejecting the customer, or accepting with no free server,
        # yields zero reward and does not occupy a server.
        reward = 0.0

    # Each busy server independently becomes free with probability free_prob.
    busy_servers = n_servers - free
    freed_servers = int(rng.binomial(busy_servers, free_prob))
    free += freed_servers

    # Draw the next customer's priority after updating server availability.
    next_prio = int(rng.integers(0, 4))

    # Return the new state with Python integers and a float reward.
    return (int(free), next_prio), float(reward)

# Step 20 - differential_sarsa_update
def differential_sarsa_update(q_table, avg_reward, state, action, reward, next_state, next_action, alpha, beta):
    # Copy the Q-table so the input array remains unchanged.
    updated_q_table = q_table.copy()

    # Compute the differential TD error for the average-reward setting.
    delta = (
        reward
        - avg_reward
        + q_table[next_state][next_action]
        - q_table[state][action]
    )

    # Update the average-reward estimate.
    avg_reward = avg_reward + beta * delta

    # Update the selected state-action value.
    updated_q_table[state][action] += alpha * delta

    # Return the updated Q-table and average-reward estimate.
    return updated_q_table, float(avg_reward)

# Step 21 - run_access_control (not yet solved)
# TODO: implement

# Step 22 - access_control_policy_table (not yet solved)
# TODO: implement

