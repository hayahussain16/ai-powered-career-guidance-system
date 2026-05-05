import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import joblib

np.random.seed(42)

streams = ['Science', 'Commerce', 'Arts', 'Engineering']
primary_interests = ['AI','Data','Coding','Design','Business',
                     'Finance','Marketing','Healthcare','Law','Civil']

# Strong mapping: for each (stream, interest) pick ONE main career
career_map = {
    ('Science','AI'): 'AI Researcher',
    ('Science','Data'): 'Data Scientist',
    ('Science','Coding'): 'Software Developer',
    ('Science','Healthcare'): 'Doctor',
    ('Science','Civil'): 'Civil Engineer',

    ('Engineering','AI'): 'ML Engineer',
    ('Engineering','Data'): 'Data Scientist',
    ('Engineering','Coding'): 'Backend Developer',
    ('Engineering','Civil'): 'Civil Engineer',

    ('Commerce','Business'): 'Business Analyst',
    ('Commerce','Finance'): 'Finance Analyst',
    ('Commerce','Marketing'): 'Digital Marketer',

    ('Arts','Design'): 'UI/UX Designer',
    ('Arts','Marketing'): 'Content Strategist',
    ('Arts','Law'): 'Lawyer',
}

def generate_row():
    stream = np.random.choice(streams)
    prim_int = np.random.choice(primary_interests)

    # marks roughly consistent with difficulty
    base = np.random.uniform(60, 95)
    tenth = base + np.random.uniform(-5, 5)
    twelfth = base + np.random.uniform(-7, 7)
    grad = base + np.random.uniform(-10, 10)

    # optional fields
    if np.random.rand() < 0.3:
        twelfth = np.nan
    if np.random.rand() < 0.5:
        grad = np.nan

    # some extra random interests for realism
    extra = np.random.choice(primary_interests, size=np.random.randint(0,2),
                             replace=False)
    interests_list = [prim_int] + list(extra)
    interests_str = ','.join(interests_list)

    # main deterministic label
    career = career_map.get((stream, prim_int), 'Software Developer')

    # small noise so model has to learn a bit
    if np.random.rand() < 0.05:
        career = 'Software Developer'

    return [stream, interests_str, tenth, twelfth, grad, career]

rows = [generate_row() for _ in range(4000)]

df = pd.DataFrame(
    rows,
    columns=['stream','interests','tenth_pct','twelfth_pct','grad_pct','target_career']
)
X = df[['stream','interests','tenth_pct','twelfth_pct','grad_pct']]
y = df['target_career']

num_features = ['tenth_pct','twelfth_pct','grad_pct']
cat_features = ['stream','interests']

preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))]),
         num_features)
    ]
)

model = Pipeline(steps=[
    ('preprocess', preprocess),
    ('clf', RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print("Synthetic model accuracy:", acc)

joblib.dump(model, 'career_model.pkl')
df.to_csv('synthetic_career_data.csv', index=False)
