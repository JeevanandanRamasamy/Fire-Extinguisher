# Fire Extinguisher Simulation and Bot Strategies

This repository contains a simulation game where bots navigate a grid to reach a button while avoiding fire. The bots have different strategies to handle the fire and obstacles. The game is built using Python, and it supports multiple bots with varying behaviors, as well as fire spread mechanics and grid initialization.

## Features

- **Grid-based simulation**: The grid size is configurable, and the grid has both open and blocked cells. Open cells are traversable, while blocked cells are not.
- **Fire mechanics**: Fire spreads across open cells based on a flammability parameter. Each non-burning open cell has a chance to catch fire based on its number of burning neighbors.
- **Bot strategies**: Multiple bots with different movement strategies, including:
  - **Bot 1**: Follows a pre-planned shortest path, ignoring fire spread.
  - **Bot 2**: Replans the path to the button at each time step, considering fire positions.
  - **Bot 3**: Avoids both fire and adjacent cells while planning the path, recalculating if necessary.
  - **Bot 4**: Uses the A* algorithm to find the optimal path, considering fire risk.
- **Dynamic fire spread**: Fire spreads across the grid at each time step, affecting bot movement.
- **Parallelized simulation**: The simulation runs multiple iterations concurrently to test bot strategies under different conditions.

## Usage

To run a test with the bots using the default grid size and flammability parameter:
```bash
python fire_extinguisher.py
```
You can modify the parameters such as the grid size, flammability parameter, and the number of iterations for testing by modifying the arguments in the main method.

## Bot Strategies

- **Bot 1**: This bot follows a pre-planned shortest path to the button but does not adapt to the spread of fire.
- **Bot 2**: This bot recalculates the shortest path to the button at each time step, accounting for fire spread.
- **Bot 3**: This bot recalculates the path every time step, avoiding fire cells and adjacent cells. If no such path is found, it recalculates based on the fire cells.
- **Bot 4**: Uses A* search, incorporating both a Manhattan distance heuristic and fire risk, where the fire risk cost is inversely proportional to the distance from the nearest fire.

## Grid Initialization

The grid is initialized with open cells (traversable) and blocked cells (non-traversable). The grid is constructed such that:
- A random start position is chosen.
- Cells are iteratively opened based on a rule: a cell is opened if it has exactly one open neighbor.
- Dead-end cells are opened with a 50% probability.
- The positions of the bot, fire, and button are randomly assigned, with the condition that the button is reachable.

## Game Process

1. **Grid Initialization**: The game begins by initializing the grid with a random configuration.
2. **Bot Movement**: At each time step, the bots try to move one step closer to the button according to their respective strategies.
3. **Fire Spread**: The fire spreads across the grid based on a probability function determined by the flammability parameter.
4. **Game Status**: The game checks whether the bots succeed in reaching the button, get caught in fire, or their path is blocked. The game ends when all bots either succeed, fail, or are blocked.

## Testing

The testing function runs simulations of the bots with different strategies over multiple iterations. The success rate for each bot is calculated and displayed, both with and without accounting for impossible cases (where the button was unreachable due to fire spread or blocked paths).

Example:
```python
success_counts, impossible_count = test_bots(D=40, q=0.3, iterations=100, method='random')
```
This function runs the simulation iterations times with the specified grid size D and flammability parameter q. It prints the success rate for each bot.

## Plotting Results

The simulation also includes functionality to plot success rates for each bot over different values of the flammability parameter. This helps visualize how the bots perform under varying fire conditions.

```python
plot_success_rates(q_values, success_list, impossible_counts, iter)
```
This function generates plots of the success rates for each bot, including both the raw success rate and the success rate excluding impossible cases.
