#%%
import pandas as pd
from Titantic.feature_engineering import feature_eng

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df = feature_eng(train_df)
all_df = feature_eng(train_df, df2 = test_df, agg=True)
# %%
