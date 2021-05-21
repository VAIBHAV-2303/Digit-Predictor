# Digit-Predictor
Recognising digits using machine learning

Made using pytorch and flask and basic html and javascript, the user draws a single digit number on the canvas and the model predicts it. The average accuracy of the model is around 0.75. Pickle has also been used to save the model instead of generating a new one everytime. 

## How To Run

```bash
bar@foo:~/Digit-Predictor$ python3 run.py
```

To retrain the model
```bash
bar@foo:~/Digit-Predictor$ python3 classifier.py
```
