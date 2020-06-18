# Twitter Sentiment Prediction
A simple sentiment prediction model trained using the [sentiment140 dataset](https://www.kaggle.com/kazanova/sentiment140).</br>

## Software Requirements:
+ [PyTorch](https://pytorch.org/) 1.5 or later.
+ [Pickle](https://docs.python.org/3.8/library/pickle.html).
+ [Python](https://www.python.org/) 3.6 or later.


## How to use:
1. Download the `config.pickle`, `twitter_sentiment_state_dict.pt` and `sentiment.py` files. Make sure the other two files are in the same directory as that of `sentiment.py`.
2. Import the sentiment.py script in your program.
3. Make an object of `SentimentPredictor` and call the `predict()` function on it.

*For Example:*
```
predictor = SentimentPredictor()
sentiment = predictor.predict('@AndrewYNg I love you, no homo')
```
This returns a float in the range (0, 1), 0 being negative sentiment and 1 being positive.
