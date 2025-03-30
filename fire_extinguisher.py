import heapq
import random
import numpy as np
import concurrent.futures
from collections import deque
import matplotlib.pyplot as plt

class FireExtinguisher:
    def __init__(self, D=40, q=0.3):
        self.D = D # Grid size
        self.q = q # Flammability parameter
        self.grid = np.full((self.D, self.D), False)  # False is blocked, True is open
        self.past_grids = [] # Store past grids for visualization
        self.open_cells = set() # Set of open cells (excluding fire cells)
        self.fire_positions = set() # Set of fire cells (open cells that are on fire)
        self.bot_positions = {} # Store positions of bots
        self.optimal_paths = {} # Store optimal paths for bots
        self.neighbors_cache = self.precompute_neighbors()

    def precompute_neighbors(self):
        """Precompute neighbors for each cell to avoid recalculating repeatedly."""
        neighbors_cache = {}
        for x, y in np.ndindex(self.D, self.D):
            neighbors = []
            if x > 0: neighbors.append((x - 1, y))
            if x < self.D - 1: neighbors.append((x + 1, y))
            if y > 0: neighbors.append((x, y - 1))
            if y < self.D - 1: neighbors.append((x, y + 1))
            neighbors_cache[(x, y)] = neighbors
        return neighbors_cache
    
    def initialize_grid(self):
        """Initialize the grid, open cells, and positions of bot, fire, and button."""
        while True:
            # Step 1: Randomly select a start position and open it
            start_x, start_y = random.randint(0, self.D - 1), random.randint(0, self.D - 1)
            self.grid[start_x, start_y] = True

            # Step 2: Iteratively open cells with exactly one open neighbor
            self.open_cells = {(start_x, start_y)}
            self.iteratively_open_cells()

            # Step 3: Open half of the dead-end cells
            self.open_dead_ends()

            # Step 4: Place bot, fire, and button
            if self.place_bot_fire_button():    # Ensure the button is reachable
                break
            else:   # Reset if button is not reachable
                self.grid = np.full((self.D, self.D), False)

    def iteratively_open_cells(self):
        """Open cells iteratively that have exactly one open neighbor."""
        while True:
            options = []
            for x, y in np.ndindex(self.D, self.D):
                if not self.grid[x, y]:
                    open_neighbors = sum(self.grid[nx, ny] for nx, ny in self.neighbors_cache[(x, y)])
                    if open_neighbors == 1:
                        options.append((x, y))
            if not options: # No more cells to open
                break
            new_open_cell = random.choice(options)
            self.grid[new_open_cell] = True # Open the cell
            self.open_cells.add(new_open_cell)

    def open_dead_ends(self):
        """Open half of the dead-end cells with exactly one open neighbor."""
        for x, y in list(self.open_cells):
            open_neighbors = sum(self.grid[neighbor] for neighbor in self.neighbors_cache[(x, y)])
            # Open random cell with exactly one open neighbor with 50% probability
            if open_neighbors == 1 and random.random() < 0.5:
                closed_neighbors = [(nx, ny) for nx, ny in self.neighbors_cache[(x, y)] if not self.grid[nx, ny]]
                chosen = random.choice(closed_neighbors)
                self.grid[chosen] = True
                self.open_cells.add(chosen)

    def place_bot_fire_button(self):
        """Randomly place the bot, fire, and button, ensuring the button is reachable."""
        open_cells_list = list(self.open_cells) # Convert set to list for random choice
        self.start_position = random.choice(open_cells_list)
        open_cells_list.remove(self.start_position) # Remove start position choice from list
        fire_cell = random.choice(open_cells_list)
        open_cells_list.remove(fire_cell) # Remove fire cell choice from list
        self.button_position = random.choice(open_cells_list)
        self.fire_positions = {fire_cell} # Set of fire cells
        self.open_cells.discard(fire_cell) # Remove fire cell from open cells

        # Check if the button is reachable from the bot
        path = self.breadth_first_search(self.open_cells, self.start_position, self.button_position)
        if path:
            self.optimal_paths['bot1'] = path
            self.bot_positions['bot1'] = self.bot_positions['bot2'] = self.bot_positions['bot3'] \
                = self.bot_positions['bot4'] = self.start_position
            self.optimal_paths['bot2'] = self.optimal_paths['bot3'] = self.optimal_paths['bot4'] = path.copy()
            return True
        return False # Button is not reachable
    
    def breadth_first_search(self, choices, start, goal):
        """Find the shortest path from start to goal using BFS."""
        queue = deque([(start, [])])
        visited = set([start])
        while queue:
            curr, path = queue.popleft()
            if curr == goal:
                return path
            for neighbor in self.neighbors_cache[curr]:
                if neighbor in choices and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None
    
    def spread_fire(self):
        """
        Spread fire based on the given probability at each time step.
        Each non-burning open cell catches fire with probability 1 - (1 - q)^K, 
        where q is the flammability parameter and K is the number of burning neighbors.
        """
        new_fires = [] # List to store new fire positions
        visited = set() # Set to track visited cells and avoid redundant checks
        for (fx, fy) in self.fire_positions:
            for (x, y) in self.neighbors_cache[(fx, fy)]:
                if (x, y) not in visited and (x, y) in self.open_cells and (x, y) not in self.fire_positions:
                    K = sum(neighbor in self.fire_positions for neighbor in self.neighbors_cache[(x, y)])
                    prob_spread = 1 - (1 - self.q) ** K # Probability of catching fire
                    if random.random() < prob_spread:
                        new_fires.append((x, y)) # Mark cell to catch fire
                    visited.add((x, y))
        # Update fire positions and open cells
        self.fire_positions.update(new_fires)
        self.open_cells.difference_update(new_fires)

    def apply_bot3_strategy(self, bot_name):
        """Bot 3 strategy to avoid fire and adjacent cells while finding the optimal path."""
        # Remove cells adjacent to fire from the open cells
        choices = self.open_cells.copy()
        for (fx, fy) in self.fire_positions:
            for neighbor in self.neighbors_cache[(fx, fy)]:
                choices.discard(neighbor)
        self.optimal_paths[bot_name] = self.breadth_first_search(choices, self.bot_positions[bot_name], self.button_position)
        if not self.optimal_paths[bot_name]: # Recalculate path based only on fire cells
            self.optimal_paths[bot_name] = self.breadth_first_search(self.open_cells, self.bot_positions[bot_name], self.button_position)

    def a_star_search(self, start, goal):
        """
        A* search algorithm to find the optimal path from start to goal
        for bot4. The bot calculates the path based on the heuristic function, 
        distance from fire, and fire risk cost. The fire risk cost is inversely
        proportional to the distance from the nearest fire cell.
        """
        def heuristic(cell):
            """Manhattan distance heuristic for A* search."""
            return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])
        
        def distance_from_fire(cell):
            """BFS to find the distance from the nearest fire cell."""
            queue = deque([(cell, 0)])
            visited = set([cell])
            while queue:
                curr, dist = queue.popleft()
                if curr in self.fire_positions:
                    return dist
                for neighbor in self.neighbors_cache[curr]:
                    if self.grid[neighbor] and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        # Initialize A* search
        open_set = [(0, start)]
        came_from = {}
        g_score = {cell: float('inf') for cell in self.open_cells}
        g_score[start] = 0
        f_score = {cell: float('inf') for cell in self.open_cells}
        f_score[start] = heuristic(start)

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                # Reconstruct the path from start to goal
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
            
            for adj in self.neighbors_cache[current]:
                if adj not in self.open_cells or adj in [i[1] for i in open_set]:
                    continue
                fire_risk_cost = 10 / distance_from_fire(adj)
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score[adj]:
                    # Update the optimal path to the cell
                    came_from[adj] = current
                    g_score[adj] = tentative_g_score
                    f_score[adj] = g_score[adj] + heuristic(adj) + self.q * fire_risk_cost
                    heapq.heappush(open_set, (f_score[adj], adj))
        return None # No path found

    def move_bot(self, bot_name):
        """Make one move for the bot based on its strategy."""
        if bot_name == 'bot2':
            self.optimal_paths[bot_name] = self.breadth_first_search(self.open_cells, self.bot_positions[bot_name], self.button_position)
        elif bot_name == 'bot3':
            self.apply_bot3_strategy(bot_name)
        elif bot_name == 'bot4':
            if self.q < 0.6:
                self.optimal_paths[bot_name] = self.a_star_search(self.bot_positions[bot_name], self.button_position)
            else:
                self.apply_bot3_strategy(bot_name) # Use bot3 strategy for high flammability
        if self.optimal_paths[bot_name] and self.bot_positions[bot_name] not in self.fire_positions \
            and self.button_position not in self.fire_positions:
            self.bot_positions[bot_name] = self.optimal_paths[bot_name].pop(0) # Move bot one step
            return True
        return False # Bot could not move
        
    def get_grid_string(self):
        """
        Return a string representation of the grid, with the bots as '1', '2', '3', and '4' respectively.
        Additionally, start is 'S', fire is 'F', button is 'B', open cells are ' ', and blocked cells are '*'.
        """
        grid_display = [[' ' if cell else '*' for cell in row] for row in self.grid]

        # Add symbols for bots, fire, and button
        bot_symbols = {'bot1': '1', 'bot2': '2', 'bot3': '3', 'bot4': '4'}
        for bot, position in self.bot_positions.items():
            x, y = position
            grid_display[x][y] = bot_symbols.get(bot, 'S')
        grid_display[self.start_position[0]][self.start_position[1]] = 'S'
        grid_display[self.button_position[0]][self.button_position[1]] = 'B'
        for fire_x, fire_y in self.fire_positions:
            grid_display[fire_x][fire_y] = 'F'

        # Build the string for the grid
        grid_str = '\n'.join(''.join(row) for row in grid_display)
        grid_str += '\n' + '-' * self.D
        return grid_str
    
    def check_success_possible(self):
        """
        Check if the button is reachable from the bot using BFS
        on a 3D grid generated from past grids. If the button is reachable,
        return the path to the button. Otherwise, return None.
        """
        queue = deque([(self.start_position, [], 0)])
        visited = set([self.start_position])

        while queue:
            curr, path, grid_num = queue.popleft() # grid_num is the step number
            if curr == self.button_position:
                return path
            if grid_num < len(self.past_grids):
                for nx, ny in self.neighbors_cache[curr]:
                    if (nx, ny) not in visited and self.past_grids[grid_num][nx][ny]:
                        visited.add((nx, ny))
                        queue.append(((nx, ny), path + [(nx, ny)], grid_num + 1))
        return None

    def initialize_optimal_grid(self):
        """Generate the optimal grid that minimizes failure cases for bots."""
        alternate = False # Alternate between opening the first and last column
        for row in range(self.D):
            if row % 2 == 0:
                self.grid[row] = True
                self.open_cells.update((row, col) for col in range(self.D))
                alternate = not alternate
            else:
                if alternate:
                    self.grid[row][0] = True
                    self.open_cells.add((row, 0))
                else:
                    self.grid[row][-1] = True
                    self.open_cells.add((row, self.D - 1))
        if not self.place_bot_fire_button(): # Ensure the button is reachable
            self.grid = np.full((self.D, self.D), False)
            self.open_cells.clear()
            self.initialize_optimal_grid() # Retry if button is not reachable

def print_bot_status(bot, game, bot_end, any_bot_succeeded, any_bot_failed, bot_success):
    """Print the status of the bot after moving one step."""
    if bot_end[bot]: # Bot already reached the button, caught in fire, or path blocked
        return None
    if not game.move_bot(bot): # Bot could not move
        bot_end[bot] = True
        any_bot_failed = True
        return any_bot_succeeded, any_bot_failed, f"{bot} path blocked"
    if game.bot_positions[bot] in game.fire_positions:
        bot_end[bot] = True
        any_bot_failed = True
        return any_bot_succeeded, any_bot_failed, f"{bot} caught in fire"
    if game.bot_positions[bot] == game.button_position:
        bot_end[bot] = True
        bot_success[bot] = True
        any_bot_succeeded = True
        return any_bot_succeeded, any_bot_failed, f"{bot} reached the button"

def process_game(game):
    """Process one game instance and check if the button was reachable."""
    num_turns = 0
    possible_path = None
    any_bot_succeeded = any_bot_failed = False
    bot_end = {'bot1': False, 'bot2': False, 'bot3': False, 'bot4': False}
    bot_success = bot_end.copy()
    game_str = ""

    # Continue moving bots until all of them reach the button or fail
    while not all(bot_end.values()):
        game_str += game.get_grid_string() + '\n'
        num_turns += 1
        for bot in bot_end:
            result = print_bot_status(bot, game, bot_end, any_bot_succeeded, any_bot_failed, bot_success)
            if result:
                any_bot_succeeded, any_bot_failed, status = result
                game_str += status + '\n'
        game.spread_fire()

        # Store the grid after each turn to check if the button was reachable
        curr_grid = game.grid.copy()
        for fx, fy in game.fire_positions:
            curr_grid[fx][fy] = False
        game.past_grids.append(curr_grid)

    # If no bot succeeded, check if it was ever possible to reach the button
    if not any_bot_succeeded:
        print(game_str)
        # Continue spreading fire until the game ends (in case solution exists after bots were blocked)
        while game.button_position not in game.fire_positions:
            game.spread_fire()
            curr_grid = game.grid.copy()
            for fx, fy in game.fire_positions:
                curr_grid[fx][fy] = False
            game.past_grids.append(curr_grid)
        possible_path = game.check_success_possible()
        if possible_path:
            # Print the path to the button
            print("Button was reachable but no bot succeeded")
            for past_grid in game.past_grids:
                print('\n'.join(''.join('S' if (x, y) == game.start_position else
                                        'B' if (x, y) == game.button_position else 
                                        'O' if (x, y) in possible_path else 
                                        ' ' if past_grid[x][y] else '*' for y in range(game.D)) for x in range(game.D)))
        else:
            return None # Button was never reachable
    return bot_success

def test_bots(D=40, q=0.3, iterations=100, method='random'):
    """
    Test the strategies for all bots over multiple iterations and
    print the success rate for each bot.
    Bot 1 Strategy:
    - This bot computes and follows a pre-planned shortest path to the button, 
    ignoring the spread of the fire.
    - It avoids the initial fire location when planning but does not adapt to 
    changes in fire positions after the plan is made.
    Bot 2 Strategy:
    - This bot re-plans the shortest path to the button at each time step,
    accounting for the current positions of fire cells.
    Bot 3 Strategy:
    - This bot re-plans the shortest path to the button at every time step,
    trying to avoid both the current fire cells and cells adjacent to fire, if possible.
    - If no such path can be found that avoids fire and adjacent cells, 
    the bot recalculates a path based only on the current fire cells.
    Bot 4 Strategy:
    - This bot uses the A* search algorithm to find the optimal path to the button.
    - It calculates the heuristic function, distance from fire, and fire risk cost. 
    - The fire risk cost is inversely proportional to the distance
    The bots move one step along the path each time step, checking if any
    reach the button or get caught in the fire.
    """
    # Generate games with the same initial conditions for comparison
    random.seed(520)
    games_list = [FireExtinguisher(D=D, q=q) for _ in range(iterations)]
    if method == 'optimal':
        for game in games_list:
            game.initialize_optimal_grid()
    else:
        for game in games_list: 
            game.initialize_grid()

    # Parallelize game processing using ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_game, games_list))
    impossible_count = sum(1 for result in results if not result)

    # Count the number of successful attempts for each bot
    success_counts = {'bot1': 0, 'bot2': 0, 'bot3': 0, 'bot4': 0}
    for result in results:
        if result:
            for bot, success in result.items():
                if success:
                    success_counts[bot] += 1

    # Print success rates for each bot
    for bot, success in success_counts.items():
        print(f"{bot} success rate: {success}/{iterations} = {100 * success/iterations}%")
        print(f"{bot} success rate excluding impossible cases: {success}/{iterations - impossible_count}",
              f"= {100 * success/(iterations - impossible_count)}%")
    return success_counts, impossible_count

def plot_success_rates(q_values, success_list, impossible_counts, iter):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Compute success rates for each bot
    bot1_counts = [d['bot1'] / iter for d in success_list]
    bot1_possible_counts = [(d['bot1'] / (iter - imp)) for d, imp in zip(success_list, impossible_counts)]
    bot2_counts = [d['bot2'] / iter for d in success_list]
    bot2_possible_counts = [(d['bot2'] / (iter - imp)) for d, imp in zip(success_list, impossible_counts)]
    bot3_counts = [d['bot3'] / iter for d in success_list]
    bot3_possible_counts = [(d['bot3'] / (iter - imp)) for d, imp in zip(success_list, impossible_counts)]
    bot4_counts = [d['bot4'] / iter for d in success_list]
    bot4_possible_counts = [(d['bot4'] / (iter - imp)) for d, imp in zip(success_list, impossible_counts)]

    # Bot 1 subplot
    axs[0, 0].plot(q_values, bot1_counts, marker='.', color='blue')
    axs[0, 0].plot(q_values, bot1_possible_counts, marker='x', color='blue', linestyle='--')
    axs[0, 0].set_title('Bot 1 Success Rate')
    axs[0, 0].set_xlabel('Flammability parameter (q)')
    axs[0, 0].set_ylabel('Success rate')
    axs[0, 0].set_ylim(0, 1)
    axs[0, 0].grid(True)

    # Bot 2 subplot
    axs[0, 1].plot(q_values, bot2_counts, marker='.', color='green')
    axs[0, 1].plot(q_values, bot2_possible_counts, marker='x', color='green', linestyle='--')
    axs[0, 1].set_title('Bot 2 Success Rate')
    axs[0, 1].set_xlabel('Flammability parameter (q)')
    axs[0, 1].set_ylabel('Success rate')
    axs[0, 1].set_ylim(0, 1)
    axs[0, 1].grid(True)

    # Bot 3 subplot
    axs[1, 0].plot(q_values, bot3_counts, marker='.', color='red')
    axs[1, 0].plot(q_values, bot3_possible_counts, marker='x', color='red', linestyle='--')
    axs[1, 0].set_title('Bot 3 Success Rate')
    axs[1, 0].set_xlabel('Flammability parameter (q)')
    axs[1, 0].set_ylabel('Success rate')
    axs[1, 0].set_ylim(0, 1)
    axs[1, 0].grid(True)

    # Bot 4 subplot
    axs[1, 1].plot(q_values, bot4_counts, marker='.', color='purple')
    axs[1, 1].plot(q_values, bot4_possible_counts, marker='x', color='purple', linestyle='--')
    axs[1, 1].set_title('Bot 4 Success Rate')
    axs[1, 1].set_xlabel('Flammability parameter (q)')
    axs[1, 1].set_ylabel('Success rate')
    axs[1, 1].set_ylim(0, 1)
    axs[1, 1].grid(True)

    # Adjust layout to avoid overlap and show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dim, iter = 40, 500 # Change the grid size and number of iterations as needed
    success_list, impossible_counts = [], []
    # List of flammability parameters to test
    q_values = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    
    for q in q_values:
        print(f"Flammability parameter q = {q}")
        # Use random or optimal method to generate the grid
        success_counts, impossible_count = test_bots(D=dim, q=q, iterations=iter, method='random')
        success_list.append(success_counts)
        impossible_counts.append(impossible_count)
    plot_success_rates(q_values, success_list, impossible_counts, iter=iter)
