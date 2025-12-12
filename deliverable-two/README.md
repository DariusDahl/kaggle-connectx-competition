# ConnectX Deep Reinforcement Learning Agent

---

## Purpose

This project implements a Deep Reinforcement Learning (DRL) bot for the Kaggle ConnectX competition. The agent is designed to learn how to play ConnectX by playing against itself, using a reward system to optimize its gameplay.

---

## Key Objectives

1. Develop a reinforcement learning agent that uses a neural network to evaluate the board and predict the best move.
2. Train the bot using self-play, allowing it to improve iteratively without relying on predefined heuristics.
3. Implement a reward system that encourages the bot to:
   - Win: **+1**
   - Avoid invalid moves: **-10**
   - Avoid letting the opponent win next turn: **-1**
   - Make valid moves (non-winning): **1/42 bonus per valid move**

---

## Agents Implemented

### Deep Reinforcement Learning Agent
- Utilizes a neural network to predict Q-values for each possible move.
- Trained using the Deep Q-Learning algorithm.
- Rewarded based on the outlined reward conditions.

---

## Files Included

- **`Deep_Reinforcement_Learning.py`**: The main file containing the DRL agent implementation.
- **`submission.py`**: The file generated after training for submission to the Kaggle competition.
- **`requirements.txt`**: A list of dependencies required to run the project.

---

## Setup Instructions

### Step 1: Ensure Required Dependencies
1. **Python**: Make sure Python is installed. [Download it here.](https://www.python.org/downloads/)
2. **Kaggle Environments**: Install this dependency to simulate the ConnectX environment.  
   ```bash
   pip install kaggle-environments
   ```

### Step 2: Download the Project
Clone or download this repository to your local machine.

### Step 3: Install Dependencies
1. Locate the `requirements.txt` file in the directory for this project.
2. Open your terminal and navigate to the project directory.
3. Run the following command:  
   ```bash
   pip install -r requirements.txt
   ```

### Step 4: Run the Agents
To train the DRL agent, run the following command:
```bash
python Deep_Reinforcement_Learning.py
```
Alternatively, use the **Run Code** button in your IDE if supported.

---

## Generating the Submission File

After training, the script will generate a `submission.py` file. This file contains a fully self-contained agent, including all necessary imports and code to compete in the Kaggle ConnectX environment.

---

## How to Submit to Kaggle

1. Navigate to the [Kaggle ConnectX competition page](https://www.kaggle.com/competitions/connectx#).
2. Upload the `submission.py` file as your entry.
3. The competition platform will evaluate your bot against other submitted bots.

---

## Generating and Running HTML to Visualize the Game Locally

You can visualize the game in your browser by generating HTML using the environment render method.

### Steps:
1. Uncomment the following code in the agent file, located about 13–14 lines above the bottom:
   ```python
   htmloutput = env.render(mode="html")
   print(htmloutput)
   ```
2. Run the file to generate HTML code in your terminal.
3. **Copy the HTML code**:
   - Select the terminal window and press `CTRL+A` to select all text, then copy it.
4. **Paste the HTML code** into a new HTML file. (PyCharm and VSCode both support HTML documents.)
5. **Clean the HTML file**:
   - Delete all lines above the line starting with `<!DOCTYPE html>` (this line must remain, as it’s the start of the HTML document).
   - Remove unnecessary lines at the end, ensuring `</html>` is the last line in the file.
6. **Run the HTML file**:
   - In PyCharm: Hover over the top-right corner of the file to see an option for launching the document in your browser.
   - In VSCode: Use the **Run** option in the top-left corner and choose **Run Without Debugging**.

This process will launch the game in your default browser, allowing you to visualize the gameplay.

---

## Acknowledgments

This project was developed using resources from the Kaggle ConnectX competition. The reinforcement learning tutorials by **Alexis Cook** and contributions from the Kaggle community were instrumental in guiding this project.  

For more details about Kaggle ConnectX, [visit the competition page here](https://www.kaggle.com/competitions/connectx).

---

## License

This project incorporates code provided by Kaggle.  
```
Copyright 2020 Kaggle Inc  
Licensed under the Apache License, Version 2.0 (the "License");  
you may not use this file except in compliance with the License.  
You may obtain a copy of the License at  
    http://www.apache.org/licenses/LICENSE-2.0  
Unless required by applicable law or agreed to in writing, software  
distributed under the License is distributed on an "AS IS" BASIS,  
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
See the License for the specific language governing permissions and limitations under the License.  
```
