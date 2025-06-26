# Quant dev assignment 2023 - Matthew Rodger attempt

import random

# Q1 - Jigsaw's coin flipping game

# What is a flip? - Start by functionising this
def flip():
    result = random.choice(["H", "T"])
    if result == "H":
        return "H"
    else:
        return "T"

# We imagine playing a single instance of our game, let's play Jigsaw's game
def single_game(initial_money = 10):
    money = initial_money
    flips = 0
    MAX_FLIPS = 1000

    while money > 0 and flips < MAX_FLIPS:
        current_flip = flip()
        
        if current_flip =='H':
            money += 2 # could adjust to get a generalised function for payoff versus non-payoff
        else:
            money -= 1 # Same here - can make negative payoff super lopsided if you like
        flips += 1

    return money <= 0, flips

# Now we macro-ise out further, what if we widen it to many thousands of people playing coin flips with Jigsaw
def simulated_games(num_games = 100000):
    went_broke_count = 0
    total_flips = 0
    
    # Defining "went_broke" as losing the game and having no money
    for _ in range(num_games):
        went_broke, num_flips = single_game()
        if went_broke:
            went_broke_count += 1
        total_flips += num_flips

    probability = went_broke_count/num_games
    avg_flips = total_flips/num_games

    return probability, avg_flips


# "Do you want to play a game? Or like a billion games??"
def main():
    num_games = 10000 # adjust this for heuristic on smaple size versus payoff.
    probability, avg_flips = simulated_games(num_games)

    print(f"\n Simulation results {num_games:,} games:")
    print(f"Probability of going broke: {probability:.4f} ({probability * 100:.1f}%)")
    print(f"Average number of flips before ending: {avg_flips:.1f}")

    print(f"Heads (P=0.5): +£2")
    print(f"Heads (P=0.5): -£1")
    ev_per_flip = 2 * 0.5 + -1 * 0.5

    print(f"Average per flip: £{ev_per_flip:.2f}")

if __name__ == "__main__":
    main()
