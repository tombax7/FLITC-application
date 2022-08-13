import os
import joblib

# Load data
directory = os.getcwd()
Distance_Class_path = directory + r'\Distance_Output.joblib'

Distance_Class = joblib.load(Distance_Class_path)

def distributer(dist_class):
    new_class = dist_class[44580:]
    
    return new_class

Distance_Class_sorted = distributer(Distance_Class)

joblib.dump(Distance_Class_sorted, directory + r'\Distance_Class_sorted.joblib')
