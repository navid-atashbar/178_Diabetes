import pandas as pd

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

if __name__ == "__main__":
    clean_data("./diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv")

