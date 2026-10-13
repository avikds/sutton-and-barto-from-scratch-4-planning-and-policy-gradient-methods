# Sutton and Barto from Scratch 4: Planning and Policy-Gradient Methods

Implement planning and policy-gradient methods from Sutton and Barto. Learn a tabular model of the Dyna maze and watch planning cut the episodes needed to find the 14-step path, repair a wrong model with Dyna-Q+ and prioritized sweeping, then train REINFORCE, a learned baseline and one-step actor-critic on the short corridor and solve the access-control queue with differential Sarsa.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** build_maze
- [x] **2.** maze_step
- [x] **3.** epsilon_greedy_tabular
- [x] **4.** q_learning_update
- [x] **5.** planning_updates
- [x] **6.** dyna_q_episode
- [x] **7.** greedy_path_length
- [x] **8.** update_model_plus
- [x] **9.** dyna_q_plus_planning
- [x] **10.** run_blocking_maze
- [x] **11.** push_priority
- [x] **12.** prioritized_sweeping_episode
- [x] **13.** corridor_step
- [x] **14.** corridor_policy
- [x] **15.** corridor_true_value
- [x] **16.** reinforce_episode
- [x] **17.** reinforce_baseline_episode
- [x] **18.** actor_critic_corridor_episode
- [x] **19.** access_control_step
- [x] **20.** differential_sarsa_update
- [x] **21.** run_access_control
- [x] **22.** access_control_policy_table

## Results

```
Maze: 6x9, start (2, 0), goal (0, 8), optimal path 14 steps
Dyna-Q n= 0 | mean steps/episode at episodes 2, 3, 5, 10, 30: [1048.8, 257.8, 250.8, 229.6, 18.2] | greedy path after 30 episodes: [16, 14, 16, 14, 18]
Dyna-Q n= 5 | mean steps/episode at episodes 2, 3, 5, 10, 30: [134.6, 62.6, 26.0, 18.0, 17.6] | greedy path after 30 episodes: [16, 14, 16, 16, 16]
Dyna-Q n=50 | mean steps/episode at episodes 2, 3, 5, 10, 30: [37.2, 15.8, 16.0, 16.2, 14.8] | greedy path after 30 episodes: [14, 14, 14, 16, 14]
Dyna-Q  | mean cumulative reward at step 1000 / 2000 / 3000 (wall moves at 1000): [33.6, 39.2, 53.4]
Dyna-Q+ | mean cumulative reward at step 1000 / 2000 / 3000 (wall moves at 1000): [62.6, 84.8, 137.8]
Dyna-Q (n=5)                | updates until the greedy path is optimal (5 runs, cap 50 episodes): [7578, 11784, 8112, 3222, 3270] | median 7578
Prioritized sweeping (n=5)  | updates until the greedy path is optimal (5 runs, cap 50 episodes): [871, 685, 531, 485, 416] | median 531
Corridor true value v(start): p=0.05 -> -82.11, p=0.5 -> -12.00, p*=2-sqrt(2)=0.5858 -> -11.66
REINFORCE alpha=2^-13                      | mean total reward at episodes 1, 100, 300, 1000: [-87.3, -34.8, -29.1, -11.9] | mean of last 100: -12.03 | final P(right): 0.464
REINFORCE+baseline a_theta=2^-9 a_w=2^-6   | mean total reward at episodes 1, 100, 300, 1000: [-87.3, -13.8, -6.9, -14.0] | mean of last 100: -13.24 | final P(right): 0.576
Actor-critic a_theta=2^-9 a_w=2^-5         | mean total reward at episodes 1, 100, 300, 1000: [-68.8, -13.1, -17.5, -10.0] | mean of last 100: -11.88 | final P(right): 0.583
Last-run critic values v(s0), v(s1), v(s2): [-13.79, -11.78, -6.51]
Access control after 200000 steps: average reward estimate 2.432
Policy (A=accept, R=reject) by priority, columns = free servers 0..10 (cf. Figure 10.5):
  priority 1: R R R R A A A A A A R | accept from 4 free servers
  priority 2: R R R A A A A A A A R | accept from 3 free servers
  priority 4: R A A A A A A A A R A | accept from 1 free servers
  priority 8: R A A A A A A A A A R | accept from 1 free servers
```
