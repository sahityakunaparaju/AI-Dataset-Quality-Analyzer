from sklearn.ensemble import RandomForestClassifier

def baseline_classifier():
    return RandomForestClassifier(n_estimators=200, random_state=42)
