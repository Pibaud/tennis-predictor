import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Charge le fichier Excel
df = pd.read_excel("data/atp_matches_2023.xlsx")

# Nettoie les données
df = df.dropna(subset=['WRank', 'LRank', 'Surface', 'AvgW', 'AvgL'])

# features
df['rank_diff'] = df['LRank'] - df['WRank']  # différence de classement
df['odds_diff'] = df['AvgL'] - df['AvgW']    # différence de cote moyenne
df['target'] = 1  # 1 = le joueur "Winner" a gagné (toujours vrai ici)

# Prise en compte de la surface
df = pd.get_dummies(df, columns=['Surface'])

# Colonnes d'entrée
features = ['rank_diff', 'odds_diff'] + [col for col in df.columns if col.startswith("Surface_")]
X = df[features]
y = df['target']

# Modélisation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2%}")