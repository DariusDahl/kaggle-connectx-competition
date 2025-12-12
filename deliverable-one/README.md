# ConnectX Agents: One-Step and N-Step Lookahead Strategies

This project focuses on implementing and comparing multiple AI agents for the ConnectX competition on Kaggle. The agents utilize different strategies, including one-step lookahead, n-step lookahead, and minimax.

---

## Agents Implemented

The following agents are included in this project:

1. **Default Agent (`Default_Given.py`)**  
   The default agent provided by Kaggle to test that submissions are working.

2. **Q4 Agent (`Q4_Given.py`)**  
   Another agent using strategies from the ConnectX competition. Slightly more advanced.

3. **One-Step Lookahead Agent (`One_Step_Lookahead_Given.py`)**  
   An agent that looks one move ahead to decide the next move.

4. **N-Step Lookahead Agent (`N_Step_Lookahead_Given.py`)**  
   A direct implementation from Kaggle, using a multi-step lookahead strategy.

5. **N-Step Lookahead Minimax Agent (`N_Step_Lookahead_Minimax.py`)**  
   An enhancement of the n-step lookahead agent using the Minimax algorithm for better decision-making.

These agents can be used to compete in the Kaggle ConnectX environment by submitting them through the provided `submission.py` file.

> **Note:**  
> The files with "Given" in the title are direct copies from Kaggle and are meant to be run as they are. When executed, they will overwrite the current `submission.py` file, which can then be submitted directly.

---

## Setup Instructions

### Step 1: Ensure Required Dependencies

1. **Python**: Make sure Python is installed. [Download it here.](https://www.python.org/downloads/)
2. **Kaggle Environments**: Install this dependency to simulate the ConnectX environment.  
   Run the following command:  
   ```bash
   pip install kaggle-environments
   ```

### Step 2: Download the Project

Clone or download this repository to your local machine.

### Step 3: Install Dependencies - `requirements.txt`

1. Locate the `requirements.txt` file in the directory for this project.
2. In your terminal, navigate to that directory.
3. Run the following command to install all dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
   After installing the dependencies, you should be able to work with the files included.

### Step 4: Run the Agents

To run any of the provided agents, execute their respective Python files. For example:

- Type the following in the terminal:  
  ```bash
  python N_Step_Lookahead_Minimax.py
  ```  
- Alternatively, if your IDE supports it, use the **Run Code** button, typically located in the top-right corner of your screen.

---

## Generating and Running HTML to Visualize the Game

To visualize the game in your browser, you can generate HTML code using the environment render method. Here’s how:

1. Uncomment the following lines in any of the agent files (`Default_Given.py`, `One_Step_Lookahead.py`, etc.), located around 13–14 lines above the bottom:
   ```python
   htmloutput = env.render(mode="html")
   print(htmloutput)
   ```
2. Run the file. The HTML code will be output to the terminal.
3. **Copy the HTML code**:
   - Select the terminal window and press `CTRL+A` to select all the text, then copy it.
4. **Paste the HTML code** into a new HTML file. (Both PyCharm and VSCode support HTML documents.)
5. **Clean the HTML file**:
   - Delete all lines above the line starting with `<!DOCTYPE html>` (this line must remain, as it is the start of the HTML document).
   - Delete a few lines at the bottom, ensuring that the last line in the file is `</html>`.
6. **Run the HTML file**:
   - In PyCharm: Hover over the top-right corner of the file. You should see an option to launch the document in your browser.
   - In VSCode: Go to the **Run** button in the top-left corner and select **Run Without Debugging**.

This process should launch the game in your default browser, allowing you to visualize the gameplay.

---

## Acknowledgments

This project was developed using code from the Kaggle ConnectX competition. Specifically, the `Default_Given.py`, `Q4_Given.py`, `One_Step_Lookahead_Given.py`, and `N_Step_Lookahead_Given.py` files are direct copies provided by Kaggle.  

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
