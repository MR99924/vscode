# Quant dev assignment 2023 - Matthew Rodger attempt

# Q2 - Battleships game

# We produce our base expression on how to count battleships
def count_battleships(grid):
    
    # Assume idiot non-grid
    if not grid or not grid[0]:
        return 0
    
    # Bare essentials
    n = len(grid)
    print("Grid size:", n)
    ships = 0
    visited = set()

    #Sniffing out some battleships
    def explore_ship(row, col):

        # We can't have anything outside the cellgrid, or indeed one that we have seen before
        if (row<0 or row>=n or col<0 or col>=n or
            grid[row][col] != 'x' or (row, col) in visited):
            return
        
        visited.add((row, col))

        # We circle round the adjacent cells in our example - can probably generalise this but for our N*N sample of examples it should be okay
        explore_ship(row-1, col)
        explore_ship(row+1, col)
        explore_ship(row, col-1)
        explore_ship(row, col+1)
        

    #Now we rattle through the various series over the grid until exhausted
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 'x' and (i,j) not in visited:
                ships += 1
                explore_ship(i,j)

    return ships

# We run our series of tests - over first example and a series of N*N grids (example, 2*2 and 3*3 respectively)
def test_battleships():
    grid1 = [
        ['x', '0', 'x', '0'],
        ['0', 'x', 'x', '0'],
        ['x', '0', '0', '0']
    ]
    
    grid2 = [
        ['0', 'x'],
        ['0', '0']
    ]

    grid3 = [
        ['x', '0', 'x'],
        ['0', '0', '0'],
        ['x', '0', '0']
    ]

    print("\nTest Results")
    print(f"Grid 1 - ships found:{count_battleships(grid1)}")
    print("Grid 1 layout:")
    for row in grid1:
        print(row)

    print(f"Grid 2 - ships found:{count_battleships(grid2)}")
    print("Grid 2 layout:")
    for row in grid2:
        print(row)

    print(f"Grid 3 - ships found:{count_battleships(grid3)}")
    print("Grid 3 layout:")
    for row in grid3:
        print(row)

if __name__ == "__main__":
    test_battleships()