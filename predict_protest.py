#This is the protest predictions python program.
#The user will be asked for several parameters in regards to the protest they are trying
#to get information on.  The entries will be validated and a model will be run.  
#Application by Nicholas A. Kovacs

##############################################################################################

"""
TABLE OF CONTENTS
    [1]  Imports
    [2]  Function declarations and initial data structures
    [3]  Introduction
    [4]  Region entry & validation
    [5]  Country entry & validation
    [6]  Violence entry & validation
    [7]  Participants entry & validation
    [8]  Length entry & validation
    [9]  Demands entry & validation
    [10] Notes entry & validation
    [11] Data modification for the model
    [12] Function declarations for composing a dataframe for the model
    [13] Dataframe read-in and composition
    [14] Model function declaration
    [15] For-loops for running the models per response
    [16] Program output

"""

##############################################################################################
#    [1] Imports

#core imports
import numpy as np
import pandas as pd
import os

#preprocessing & data modification
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler

#models & the pipeline
from sklearn.pipeline import Pipeline, FeatureUnion
from xgboost import XGBClassifier


##############################################################################################
#    [2] Function declarations and initial data structures

def validate_input(input_str, valids):
    while True:
        entry = input(input_str)
        if entry not in valids:
            print("That is not an acceptable input.")
            continue
        else:
            break
    return entry



def validate_numeric(input_str, valids):
    valid = False
    while valid == False:
        entry = input(input_str)
        for c in entry:
            if c not in valids:
                print("That is not an acceptable input.")
                valid = False
                break
            else:
                valid = True
                continue
    return entry


region_input = "Enter a number corresponding to the region:  "
region_valids = {"1":"Central America", "2":"South America", "3":"Europe", "4":"Middle East/North Africa", "5":"Sub-Saharan Africa", "6":"Asia"}

violence_input = "Enter 1 for yes, and 0 for no:  "
violence_valids = ["1", "0"]

participants_input = "It's okay if it's not accurate, give it your best guess:  "
length_input = "Enter here:  "
numeric_valids = "1234567890"


##############################################################################################
#    [3] Introduction

os.system("clear")

print("Welcome to Protest Outcome Predictor.  For this exercise, please ensure that all of your entries precisely correct (to the best of your knowledge) and that everything is spelled correctly where applicable, or the model may not run as intended.\n\n")

print("We will ask the following information from you:\n(A) What region the protest will be in;\n(B) What country the protest will be in;\n(C) Whether you have reason to believe there will be violence by the protestors;\n(D) How many protesters you think there will be;\n(E) How long you think the protest will take (in days);\n(F) What the protest demands are;\n(G) And finally, a short paragraph of notes describing the protest."
)

input("\nPress enter when you're ready to proceed.")    



##############################################################################################
#    [4]  Region entry & validation

os.system("clear")

print("What region best describes your location?  (Note: North America and Oceania are currently not able to be modelled due to availability of data.)\n1. Central America\n2. South America\n3. Europe\n4. Middle East/North Africa\n5. Sub-Saharan Africa\n6. Asia")

region = validate_input(region_input, region_valids.keys())


##############################################################################################
#    [5]  Country entry & validation

os.system("clear")

country = input("What country will the protest occur in?  (Please double check spelling and spacing where necessary):  ")


##############################################################################################
#    [6]  Violence entry & validation

os.system("clear")

print("Do you have reason to believe there will be violence perpetrated by the protestors?")

violence = validate_input(violence_input, violence_valids)


##############################################################################################
#    [7]  Participants entry & validation

os.system("clear")

print("How many participants do you believe there will be in the protest?  When entering a number, only use numeric characters - no commas, letters, or spaces.")

participants = validate_numeric(participants_input, numeric_valids)


##############################################################################################
#    [8]  Length entry & validation

os.system("clear")

print("How long do you think the protest will last?  Enter in the number of days.  Once again, just make your best guess and use only numeric characters.")

length = validate_numeric(length_input, numeric_valids)


##############################################################################################
#    [9]  Demands entry & validation

os.system("clear")


demands_display_dict = {
    'demand_labor_wage_dispute':
        [1, 'Labor or wage dispute', '[Labor or wage dispute]  SELECTED'], 
        
    'demand_land_farm_issue':
        [2, 'Land or farm issue', '[Land or farm issue]  SELECTED'],
        
    'demand_police_brutality':
        [3, 'Police brutality', '[Police brutality]  SELECTED'], 
        
    'demand_political_behavior_or_process':
        [4, 'Political behavior or process', '[Political behavior or process]  SELECTED'],
        
    'demand_price_hike_or_tax_policy':
        [5, 'Price hike or tax policy', '[Price hike or tax policy]  SELECTED'], 
        
    'demand_removal_of_politician':
        [6, 'Removal of politician', '[Removal of politician]  SELECTED'],
        
    'demand_social_restrictions':
        [7, 'Social restrictions', '[Social restrictions]  SELECTED']
}

demands_comp_dict = {
    'demand_labor_wage_dispute':0,
    'demand_land_farm_issue':0,
    'demand_police_brutality':0, 
    'demand_political_behavior_or_process':0,
    'demand_price_hike_or_tax_policy':0, 
    'demand_removal_of_politician':0,
    'demand_social_restrictions':0
}

demands_input = "Enter here:  "
demands_valids = ["1", "2", "3", "4", "5", "6", "7", ""]


def display_demand_set():
    os.system("clear")
    print("Now, select however many demands your protest has.  Do this by entering a number corresponding to the demand or demands that most closely match.  You may enter the same number again to unselect the option.  When you are ready to proceed, enter a blank line.")
    for key in demands_display_dict.keys():
        if demands_comp_dict[key] == 1:
            print(f"{demands_display_dict[key][0]}. {demands_display_dict[key][2]}")
        else:
            print(f"{demands_display_dict[key][0]}. {demands_display_dict[key][1]}")


def swap_demand_truth(entry):
    for key in demands_display_dict.keys():
        if int(entry) == demands_display_dict[key][0]:
            if demands_comp_dict[key] == 0:
                demands_comp_dict[key] = 1
            else:
                demands_comp_dict[key] = 0



while True:
    display_demand_set()
    entry = validate_input(demands_input, demands_valids)
    if entry == "":
        break
    else:
        swap_demand_truth(entry)
        continue


##############################################################################################
#    [10] Notes entry & validation

os.system("clear")



print("Finally, add some notes about your protest.  The more descriptive you are, the better, but what we're looking for is a good 2-5 sentences describing what's going on.")



while True:
    os.system("clear")

    print("Finally, add some notes about your protest.  The more descriptive you are, the better, but what we're looking for is a good 2-5 sentences describing what's going on.")
    notes = input("Enter here:  ")
    
    os.system("clear")
    print(f"Your notes are: \n\n{notes}\n\n")
    notes_confirm_input = "Are you satisfied with this entry?  Press 1 for yes and 0 for no."
    confirm = validate_input(notes_confirm_input, ["0", "1"])
    
    if confirm == "0":
        continue
    else:
        break





##############################################################################################
#    [11] Data modification for the model

os.system("clear")
print("Please wait while the model runs...")


#Modifying the data to be compatible for the model.
region_valids["4"] = "MENA"
region_valids["5"] = "Africa"
region = region_valids[region]

country = f"country_{country}"
violence = int(violence)
participants = int(participants)
length = int(length)

##############################################################################################
#    [12] Function declarations for composing a dataframe for the model

def populate_dict(region, country, violence, participants, length, demands_comp_dict, notes, model_ready_features):
    #Accepts all of the user values entered thus far and creates a dictionary for the purposes of passing it to the populate_df function.
    pop_dict = {}
    if country in model_ready_features:
        pop_dict[country] = 1
    pop_dict["year_2019"] = 1
    pop_dict["protesterviolence"] = violence
    pop_dict["participants"] = participants
    pop_dict["notes"] = notes
    pop_dict["protest_length"] = length
    pop_dict.update(demands_comp_dict)
    return pop_dict
    

def populate_df(dct, features):
    #This function accepts a dictionary and assigns the values of the dictionary to a model-ready dataframe.
    forbiddens = ["region", "protesteridentity", "sources", "y_accomodation", "y_ignore", "y_adverse_reaction", "y_state_violence"]
    model_ready = pd.DataFrame(columns=features)
    
    model_ready_dict = {}
    for f in features:
        if f not in forbiddens:
            model_ready_dict[f] = 0
    model_ready_dict.update(dct)
    
    model_ready = pd.DataFrame()
    model_ready = model_ready.append(model_ready_dict, ignore_index=True)
    
    return model_ready



##############################################################################################
#    [13] Dataframe read-in and composition

#Read in the cleaned protest CSV, drop the excess column, and subdivide the dataframe into
#the region as specified by the user.
df = pd.read_csv("./cleaned_protests.csv")
df.drop(columns=["Unnamed: 0"])
df = df[df["region"]==region]

#Extract the list of model ready features for the given region we're working with.
dummy_df = pd.get_dummies(df, columns=["year", "country"])
model_ready_features = list(dummy_df.columns)


#This creates a one-line dataframe that will be fed into the model we construct.  Then it will get probabilistic outputs.
entry_df = populate_df(
    populate_dict(
        region, 
        country, 
        violence, 
        participants, 
        length, 
        demands_comp_dict, 
        notes,
        model_ready_features),
    model_ready_features
)


##############################################################################################

possible_responses = ["y_accomodation", "y_ignore", "y_adverse_reaction", "y_state_violence"]

models_dict = {}



def model_trainer(df, response):
    #This takes the regional data and runs a model for a given possible response fed into it.
    #It returns the pipeline which will then be used to predict outcomes based on protest features.
    df = pd.get_dummies(df, columns=["year", "country"])
    
    forbiddens = ["region", "protesteridentity", "sources", "y_accomodation", "y_ignore", "y_adverse_reaction", "y_state_violence"]
    xgb_features = [i for i in df.columns if i not in forbiddens]
    
    numerics = ['int64', 'float64', 'uint8']
    num_features = [i for i in df.select_dtypes(include=numerics).columns if i not in forbiddens]
    
    X = df[xgb_features]
    y = df[response]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    
    get_numeric_data = FunctionTransformer(lambda df: df[num_features], validate=False)
    get_text_data = FunctionTransformer(lambda df: df["notes"], validate=False)
    
    
    pipe = Pipeline([
    ('features', FeatureUnion([
            ('numeric_features', Pipeline([
                ('selector', get_numeric_data),
                ('ss', StandardScaler())
            ])),
             ('text_features', Pipeline([
                ('selector', get_text_data),
                ('cvec', CountVectorizer(stop_words='english', max_df=0.8, max_features=2000))
            ]))
         ])),
    ('xg', XGBClassifier(max_depth=2, learning_rate=0.05))
    ])
    
    pipe.fit(X_train, y_train)


    return pipe, pipe.score(X_test, y_test)

##############################################################################################
#    [15] For-loops for running the models per response

#Populate the models dictionary with a model trained for each possible response.
#Also populate a dictionary of accuracy.
acc = {}
for response in possible_responses:
    models_dict[response], acc[response]  = model_trainer(df, response)


#Get baselines and display them for the user.
baselines = {}
for response in possible_responses:
    baselines[response] = round(df[response].mean(), 3) * 100

##############################################################################################
#    [16] Program output

os.system("clear")

print("For your region, the following outcomes, on average, have the following probabilities:\n")

print(f"There is a {baselines['y_accomodation']}% chance the government will accommodate the protest.")

print(f"There is a {baselines['y_ignore']}% chance the government will ignore the protest.")

print(f"There is a {baselines['y_adverse_reaction']}% chance the protest will elicit an adverse reaction from the state. \n\t(in the form of crowd dispersal or arrests)")

print(f"There is a {baselines['y_state_violence']}% chance the state will react with violence.\n\t(in the form of beatings, shootings, or killings)")

print("\n\n")

#finally, use the set of four pipes returned in the previous function calls to create probabilitistic outputs with confidence intervals.
for response in possible_responses:
    pred = models_dict[response].predict_proba(entry_df)
    
    pred = pred[0][1]
    pred = round(pred, 3) * 100
    
    acc[response] = round(acc[response], 3) * 100

    if (acc[response] - max(baselines[response], 100-baselines[response])) < 5:
        confidence = "not "
    else:
        confidence = ""
    
    print(f"The model predicts that the state has a {pred}% of responding with: {response[2:]}")
    print(f"Based on how this model performs against the baseline, we are {confidence}confident about this outcome.")
    print("")









