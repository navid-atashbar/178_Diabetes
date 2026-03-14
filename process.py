import pandas as pd
from sklearn.model_selection import train_test_split

#101,766 rows 50 cols
#drop weight
#fill medical_specialty,payer_code, race, drop diag_1,Diag_2_diag3 w Unknown

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
    df['admission_type_id'] = df['admission_type_id'].map({1:'Emergency', 2:'Urgent',3:'Elective',4:'Newborn',5:'NotAvailable', 6:'Null',7:'TraumaCenter',8:'NotMapped'})


    df['discharge_disposition_id'] = df['discharge_disposition_id'].map({
        1: 'Home', 2: 'TransferShortTerm', 3: 'TransferSNF',
        4: 'TransferICF', 5: 'TransferOther', 6: 'HomeWithHealth',
        7: 'LeftAMA', 8: 'HomeIV', 9: 'Inpatient',
        10: 'NeonateTransfer', 11: 'Expired', 12: 'Outpatient',
        13: 'HospiceHome', 14: 'HospiceFacility', 15: 'SwingBed',
        16: 'OutpatientOther', 17: 'OutpatientSame', 18: 'NULL',
        19: 'ExpiredHome', 20: 'ExpiredFacility', 21: 'ExpiredUnknown',
        22: 'TransferRehab', 23: 'TransferLongTerm', 24: 'TransferMedicaidNursing',
        25: 'NotMapped', 26: 'Unknown', 27: 'TransferFederal',
        28: 'TransferPsych', 29: 'TransferCAH',
        30: 'TransferOtherHealth'
    })

    df['admission_source_id'] = df['admission_source_id'].map({
        1: 'PhysicianReferral', 2: 'ClinicReferral', 3: 'HMOReferral',
        4: 'TransferHospital', 5: 'TransferSNF', 6: 'TransferOther',
        7: 'EmergencyRoom', 8: 'CourtLaw', 9: 'NotAvailable',
        10: 'TransferCAH', 11: 'NormalDelivery', 12: 'PrematureDelivery',
        13: 'SickBaby', 14: 'ExtramuralBirth', 15: 'NotAvailable',
        17: 'NULL', 18: 'TransferHomeHealth', 19: 'ReadmitHomeHealth',
        20: 'NotMapped', 21: 'Unknown', 22: 'TransferInpatient',
        23: 'BornInside', 24: 'BornOutside', 25: 'TransferAmbulatory',
        26: 'TransferHospice'
    })

    #now turn the readmition into number 0 for NOT readdmited, 1 for Readmiited less than 30 days
    #2 for READMITTED AFTER 30
    df['readmitted'] = df['readmitted'].map({'NO':0, '>30':1,'<30':2})
    # print(df['readmitted'].value_counts())
    #TURNS EVERYTHING  stringINTO 1 0 T F etc instead of words
    df.dropna(subset=['admission_type_id', 'discharge_disposition_id',
                      'admission_source_id', 'readmitted'], inplace=True)
    df = pd.get_dummies(df)

    # print(df.shape)
    #TO DOOO MAKE THE SPLIT
    X = df.drop(["readmitted"], axis=1)
    y = df["readmitted"]
    x_rest, x_te, y_rest, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    x_tr, x_val, y_tr, y_val = train_test_split(x_rest, y_rest, test_size=0.25, random_state=42)
    return x_tr, x_val,x_te, y_tr, y_val, y_te
def diagnoses(codes):
    if codes == 'Unknown':
        return 'Unknown'
    try:
        number = float(str(codes).replace('V', '').replace('E',""))
    except:
        if str(codes).startswith('V'):
            return 'Other'


if __name__ == "__main__":
    x_tr, x_val,x_te, y_tr, y_val, y_te = clean_data("./diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv")
    print(x_tr.shape, x_val.shape, x_te.shape)
