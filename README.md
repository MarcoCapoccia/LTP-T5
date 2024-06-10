# LTP-T5

## All

### Prerequisites

- Python 3.7 or higher
- pip

### Dependencies

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

If this does not work try running these lines.

For t5.py:
```
pip install torch
pip install transformers
pip install json
pip install tqdm
```
For evaluation.py:
```
pip install datasets
pip install json
pip install matplotlib
```
## T5
Clone this repo using:
```bash
git clone https://github.com/MarcoCapoccia/LTP-T5.git
```
Then unzip the dataset and replace all the 'PATH TO' (find using ctrl+f or cmd+f) in the t5.py file with the path to the corresponding files.

Now you can run the t5.py script.

## Evaluation
We have put our results in this google drive:
```
https://drive.google.com/drive/folders/19cUqDr-MEjYCdANas_7L0cP-5pi6DBag?usp=sharing
```
If you want to test evaluation.py you only have to download:
1. **training_losses.json**: for the loss plot
2. **predictions_and_references.json**: for the metrics

Then you should replace all the 'PATH TO' (find using ctrl+f or cmd+f) in the evaluation.py file with the path to the corresponding files.

Now you can run the evaluation.py script

## Known Issues, Limitations
The training process is time-consuming. Try running fewer epochs or maybe use the Hábrók server.

## Areas for Improvement
Our model tends to overfit a lot, despite our efforts to solve it. We have identified several additional strategies that could further address this issue. Even though we tried a lot already.
- **Data Handling:** Having a larger, more diverse dataset could help solve overfitting.
- **Regularization Techniques:** Implement more advanced regularization techniques next to the weight decay and dropout we already implemented.
- **Hyperparameter Tuning:** Conduct a more thorough hyperparameter search to find the optimal settings that might prevent overfitting.
- **Model Complexity:** Investigate if a simpler model could achieve better generalization for this specific task. We had to use the T5 model, so simplification was not an option.
- **Computational Efficiency:** Explore ways to optimize the training process to make it more efficient and less resource-intensive.

## Contact
If you encounter any problems at all please contact us at e.b.baho@student.rug.nl or m.capoccia@student.rug.nl.
