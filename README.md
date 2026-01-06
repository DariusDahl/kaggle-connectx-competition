# Kaggle ConnectX Competition

This repository contains the work of our team for the **Kaggle ConnectX** competition, which is based on creating AI agents to play Connect Four in a reliable and strategic manner. Our best submission achieved a **skill rating of 714.5**, which, compared against a current snapshot of the public leaderboard (12/12/2025), would be roughly **top 10%** (tied for about 22nd out of 224 entries at the time of review).

🔗 **[Official Kaggle ConnectX Competition](https://www.kaggle.com/c/connectx)**

## What is ConnectX?

ConnectX is Kaggle's variation of the classic Connect Four game. Each player takes turns placing markers on a grid, aiming to align a specific number of markers horizontally, vertically, or diagonally before the opponent. The ConnectX variation introduces a few twists, such as different grid sizes or numbers of markers required to win, which makes the competition challenging and diverse.

## Repository Overview

This repository is structured to reflect the three major deliverables we worked on during the project. Each deliverable focuses on distinct aspects of developing a high-performing ConnectX bot. Detailed descriptions for each deliverable can be found in their respective READMEs.

### Deliverables

#### 1. Baseline Bot
- **Goal**: Create a simple, rule-based bot (heuristic-based) to serve as the foundation for further development.
- **Approach**: Focus on basic strategies like blocking opponents, prioritizing immediate wins, and choosing the first free column when no strategic moves are available.
- **README**: [Baseline Bot README](deliverable-one/README.md)

#### 2. Reinforcement Learning Agent
- **Goal**: Train a bot using reinforcement learning techniques to optimize moves and strategies based on gameplay experience.
- **Approach**: Developed a training pipeline using Python and popular machine learning libraries. Our agent learned from self-play and improved over time.
- **README**: [Reinforcement Learning Agent README](deliverable-two/README.md)

#### 3. Submission Bot (Final Model)
- **Goal**: Construct and refine the bot submitted to the competition leaderboard, combining the strengths of both the baseline bot and reinforcement learning agent.
- **Approach**: Used experimentation and analysis from earlier deliverables to create the best-performing bot version. Incorporated tactics such as depth-limited tree search and optimized heuristics to complement learned strategies.
- **README**: [Submission Bot README](deliverable-three/README.md)

## Language Composition

- **Python (81.1%)**: Core development of the AI agents and simulations.
- **JavaScript (8%) / HTML (5.6%) / TypeScript (5.3%)**: Supplementary scripts from a web-based interface for visualization.

## Learnings and Outcomes

- **Strategic AI Development**: Explored a range of AI methodologies, from basic heuristics to reinforcement learning, for developing competitive AI agents.
- **Performance Highlights**: Our bot achieved a score of **714.5**, performing reliably against various opponents in the competition.
- **Iteration and Experimentation**: Balanced hand-crafted strategies with machine-learned models to develop a hybrid and competitive approach.

## Future Work

- **Further Training**: Slight adjustments to our RL training pipeline may yield bots capable of consistently achieving scores above 800.
- **Extensions**: Test the bot’s robustness on slightly altered rule sets (e.g., varying grid sizes or custom win conditions).

Feel free to explore the individual deliverables for technical details, code, and experimental results!

