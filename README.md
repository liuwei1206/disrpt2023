# disrpt2023
The code for competition disrpt2023.

## Requirement
Our working environment is Python 3.8. Before you run the code, please make sure you have installed all the required packages. You can achieve it by simply execute the shell as `sh requirements.sh`

## Task 1 and 2
For running Task 1 and Task 2：
- activate your Conda environment and make sure your environment matches the requirements.txt. 

- In a folder called **data**, put the all corpora data in the folder called **dataset**. For example, the data path for the corpus eng.pdtb.pdtb will be **data/dataset/eng.pdtb.pdtb**.
- You need to create a folder called **result** within the **data** folder, the path for this folder is **data/result**. This folder will store the final trained model and apply some of them for the corpora without training data. 
- Now, you can use the command **sh run_task12.sh** in your terminal to run the code for Task 1 and Task 2.

## Task 3
To run the code of task 3, you should do like the follows:
1. prepare data. Put all the raw corpora under the folder "data/dataset".
2. preprocessing. Convert the raw corpora into matched format via `python3 preprocessing.py`.
3. run. Execute the shell file as `sh run_task3.sh`.
