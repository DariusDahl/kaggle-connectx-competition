# ConnectX Alpha-Beta Pruning Agent

---

## Purpose

This project implements an improved bot for the Kaggle ConnectX competition using heuristic-based strategies, dynamic depth adjustment, and alpha-beta pruning. The agent is designed to evaluate the game state efficiently and make optimal decisions within limited computation time.

---

## Key Objectives

1. Implement an optimized heuristic-based bot that uses advanced strategies such as:
   - **Alpha-beta pruning** to enhance minimax search efficiency.
   - **Dynamic depth adjustment** based on the game state.
   - **Refined heuristics** to balance offensive and defensive play.
2. Ensure the agent can handle the game effectively with limited computation time while maximizing its performance.
3. Generate a fully functional, self-contained `submission.py` file for the Kaggle competition.

---

## Agents Implemented

### Alpha-Beta Pruning Agent
- **Algorithm**: Utilizes a minimax algorithm with alpha-beta pruning to optimize the decision-making process.
- **Dynamic Depth Adjustment**: Adapts the search depth based on the number of empty cells in the game board.
- **Advanced Heuristic Evaluation**:
  - Rewards moves that create winning opportunities.
  - Penalizes moves that allow the opponent to win.
  - Encourages control of the center column and other strategic positions.

---

## Files Included

- **`Alpha_Beta.py`**: The main Python file containing the implementation of the alpha-beta pruning agent.
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
To run the enhanced agent, execute the following command:
```bash
python Alpha_Beta.py
```
Alternatively, use the **Run Code** button in your IDE if supported.

---

## Generating the Submission File

After running the alpha-beta pruning agent, the script will generate a `submission.py` file. This file contains a fully self-contained agent, including all necessary imports and code to compete in the Kaggle ConnectX environment.

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
4. **Paste the HTML code** into a new HTML file. (Both PyCharm and VSCode support HTML documents.)
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
