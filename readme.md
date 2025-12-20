# Formula 1 Tyre-Fuel Strategy Optimizer
This project is about optimizing tyre strategies for the lowest forecasted race time. This project follows the 2022-2025 ground effect regulations for consistent car performance.

## Decision Variables
- Available tyre compounds (with tyre age)

## Constraints
- Max laps for each compound
- Race laps
- Minimum 2 stints

## Objective 
Minimizing the time needed to complete the race

$$
    \text{Min Race Time} =(N-1) \times P + \sum^{N}_{i=1} \space F(T_i, L_i)
$$

Where:
- $P =$ Pit penalty (s)
- $N =$ Amount of stints
- $F =$ Function to forecast stint time
- $T_i =$ Chosen Tyre
- $L_i =$ Length of stint