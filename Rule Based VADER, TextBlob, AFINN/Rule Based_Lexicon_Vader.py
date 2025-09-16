import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm

class SentimentAnalysis:
    def __init__(self):
        self.Dataset =pd.read_csv("IMDB Dataset.csv")
        self.VaderInitialization()

    def VaderInitialization(self):
        # Initialize analyzer
        sia = SentimentIntensityAnalyzer()

        # Apply sentiment analysis with tqdm progress bar
        tqdm.pandas(desc="Analyzing Sentiment")

        self.Dataset["scores"] = self.Dataset["review"].apply(lambda x: sia.polarity_scores(x))
        self.Dataset["compound"] = self.Dataset["scores"].apply(lambda x: x["compound"])

        # Classify as Positive / Negative / Neutral
        self.Dataset["sentiment"] = self.Dataset["compound"].apply(
            lambda c: "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")
        )

        print(self.Dataset[["review", "compound", "sentiment"]])
        self.Dataset.to_csv("Results_Using_Vader.csv")


SentimentAnalysisObj = SentimentAnalysis()
