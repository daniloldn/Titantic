
import pandas as pd
from Titantic.feature_engineering import feature_eng
from Titantic.model import model

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df = feature_eng(train_df)
test_df = feature_eng(test_df, df_type="Test")

model(train_df)
