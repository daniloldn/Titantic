import pandas as pd
def agg_data(df1, df2):
    all_df = pd.concat([df1, df2], axis=0)
    all_df["set"] = "train"
    all_df.loc[all_df.Survived.isna(), "set"] = "test"
    return all_df


def family_size(df):
    df["Family Size"] = df["SibSp"] + df["Parch"] + 1
    
    return df

def label_fam(df):
    df["Family Type"] = df["Family Size"]
    df.loc[df["Family Size"] == 1, "Family Type"] = "Single"
    df.loc[(df["Family Size"] > 1) & (df["Family Size"] < 5), "Family Type"] = "Small"
    df.loc[(df["Family Size"] >= 5), "Family Type"] = "Large"
    return df

def label_titles(df):
    df["Titles"] = df["Title"]
    #unify `Miss`
    df['Titles'] = df['Titles'].replace('Mlle.', 'Miss.')
    df['Titles'] = df['Titles'].replace('Ms.', 'Miss.')
    #unify `Mrs`
    df['Titles'] = df['Titles'].replace('Mme.', 'Mrs.')
    # unify Rare
    df['Titles'] = df['Titles'].replace(['Lady.', 'the Countess.','Capt.', 'Col.',\
     'Don.', 'Dr.', 'Major.', 'Rev.', 'Sir.', 'Jonkheer.', 'Dona.'], 'Rare')
    return df



def age_interval(df):
    df["Age Interval"] = 0.0
    df.loc[ df['Age'] <= 16, 'Age Interval']  = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age Interval'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age Interval'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age Interval'] = 3
    df.loc[ df['Age'] > 64, 'Age Interval'] = 4

    return df

def fare_interval(df):
    df['Fare Interval'] = 0.0
    df.loc[ df['Fare'] <= 7.91, 'Fare Interval'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare Interval'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare Interval']   = 2
    df.loc[ df['Fare'] > 31, 'Fare Interval'] = 3

    return df

def Pclass_sex(df):
    df["Sex_Pclass"] = df.apply(lambda row: row['Sex'][0].upper() + "_C" + str(row["Pclass"]), axis=1)

    return df


def parse_names(row):
    try:
        text = row["Name"]
        split_text = text.split(",")
        family_name = split_text[0]
        next_text = split_text[1]
        split_text = next_text.split(".")
        title = (split_text[0] + ".").lstrip().rstrip()
        next_text = split_text[1]
        if "(" in next_text:
            split_text = next_text.split("(")
            given_name = split_text[0]
            maiden_name = split_text[1].rstrip(")")
            return pd.Series([family_name, title, given_name, maiden_name])
        else:
            given_name = next_text
            return pd.Series([family_name, title, given_name, None])
    except Exception as ex:
        print(f"Exception: {ex}")

    
def apply_parse(df, col_array):
    df[col_array] = df.apply(lambda row: parse_names(row), axis=1)

    return df


def feature_eng(df, col_array = ["Family Name", "Title", "Given Name", "Maiden Name"], agg = False, df2 = None):
    if agg == True:
        df = agg_data(df, df2)
    df = apply_parse(df, col_array)
    df = family_size(df)
    df = label_fam(df)
    df = label_titles(df)
    df = age_interval(df)
    df = fare_interval(df)
    df = Pclass_sex(df)
    
    return df