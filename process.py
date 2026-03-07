import pandas as pd
from sklearn.model_selection import train_test_split

#101,766 rows 50 cols
#drop weight
#fill medical_specialty,payer_code, race, drop diag_1,Diag_2_diag3 w Unknown
# print(df.shape)
# print(df.dtypes)
# print(df.describe())
# print((df == '?').sum())
#Here i just removed the data that had a ton of empty slots like weight
# I also reformated the ? into Unknown
def clean_data(files):
    # df = pd.read_csv('./diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv')
    df = pd.read_csv(files)
    df.drop(["weight","encounter_id", "patient_nbr"], axis=1, inplace=True)
    df = df[df["diag_1"] !='?']
    df = df[df["diag_2"] !='?']
    df = df[df["diag_3"] !='?']
    df.replace('?', "Unknown", inplace=True)
    #now turn the readmition into number 0 for NOT readdmited, 1 for Readmiited less than 30 days
    #2 for READMITTED AFTER 30
    df['readmitted'] = df['readmitted'].map({'NO':0, '>30':1,'<30':2})
    # print(df['readmitted'].value_counts())
    #TURNS EVERYTHING  stringINTO 1 0 T F etc instead of words
    df = pd.get_dummies(df)
    # print(df.shape)
    #TO DOOO MAKE THE SPLIT
    X = df.drop(["readmitted"], axis=1)
    y = df["readmitted"]
    x_rest, x_te, y_rest, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    x_tr, x_val, y_tr, y_val = train_test_split(x_rest, y_rest, test_size=0.25, random_state=42)
    return x_tr, x_val,x_te, y_tr, y_val, y_te
if __name__ == "__main__":
    x_tr, x_val,x_te, y_tr, y_val, y_te = clean_data("./diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv")
    print(x_tr.shape, x_val.shape, x_te.shape)
