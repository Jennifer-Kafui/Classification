#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# # Dictionary:

# age; workclass; fnlwgt; education; education-num; marital-status; occupation; relationship; race; sex: 1 is female else 0; capital-gain; capital-loss; hours-per-week; native-country; target: 1 if>50K else 0,* age4;

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('Solarize_Light2')


# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/renfelo/datascience/master/others/data_science/notebooks/city_census/data/census.csv')


# In[3]:


fig, ax = plt.subplots(figsize=(10,5))

pd.crosstab(df['target'], df['sex']).plot(kind='bar', title='Relation between gender and annual salary', ax=ax)
ax.set_xticklabels(['<=50K', '>50K'],rotation=0)
ax.legend(['Male', 'Female'])


# In[4]:


fig, ax = plt.subplots(figsize=(10,5))

sns.barplot(df['hours-per-week'],df['education'], ax=ax) .set_title('Relation between education level and hours of work');


# # Data Preprocessing

# Features¶ age; workclass; fnlwgt; education; education-num; marital-status; occupation; relationship; race; sex: 1 if female else 0; capital-gain; capital-loss; hours-per-week; native-country; target: 1 if >50K else 0;

# In[5]:


import pandas as pd
import numpy as np
import re


# In[6]:


# Get data directly from UCI ML using 'read_table'
raw_data = pd.read_table('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', delim_whitespace=True, error_bad_lines=False)


# In[7]:


# Security copy (just in case...)
df = raw_data.copy()


# In[8]:


# Hmm, we have a problem here: there are no column names... let's fix that.
df = raw_data.T.reset_index().T.reset_index(drop=True) # Haaaaack!

# Let's put a name to it now
column_names = ['age','workclass','fnlwgt','education','education-num','marital-status',
                'occupation','relationship','race','sex','capital-gain','capital-loss',
                'hours-per-week','native-country', 'target']

df.columns = column_names
df.head(1)


# # Data Wrangling

# Let's verify how consistent are the column data and modify if necessary

# In[9]:


def verify_data_spread():
    for column in df.columns:
        print(column, ':\n\n')
        print(df[column].value_counts())

#verify_data_spread()


# In[10]:


# There are just ony thing that I want to modify:
# -> Some columns have an '?' for the missing parameters, let's modify that to 'missing'

def modify_symbol(symbol='?'):
    occurrences = []
    for column in df.columns:
        df[column] = df[column].apply(lambda x: re.sub('[,.;]', '', x))
        df[column] = df[column].apply(lambda x: x.replace('-', ' '))
        for index, value in df[column].items():
            if symbol in value and column not in occurrences:
                occurrences.append(column)
    
    for column in occurrences:
        df[column] = df[column].apply(lambda x: x if '?' not in x else 'missing')

    

modify_symbol()


# In[11]:


for column in df.columns:
        # Removing some commas from data
        numerical = ['age', 'fnlwgt', 'education-num', 'capital-loss', 'capital-gain', 'hours-per-week']
    
        if column in numerical:
            df[column] = df[column].apply(lambda x: x.split(',')[0] if ',' in x else x) 
            df[column] = df[column].astype('int64')


# # More Modifications

# Some columns can be already modified to boolean ones, like sex and target; There are a bunch of zeros inside some columns, but they are necessary;

# In[12]:


df.tail(1)


# In[13]:


df['sex'].value_counts(), df['target'].value_counts()


# In[14]:


# Booleanizing (huh) categories
df['sex'] = df.sex.apply(lambda x: 1 if 'Female' in x else 0)
df['target'] = df.target.apply(lambda x: 1 if '>50K' in x else 0)
df.head()


# In[16]:


df['sex'].value_counts(), df['target'].value_counts()


# # Estimators

# Features¶ age; workclass; fnlwgt; education; education-num; marital-status; occupation; relationship; race; sex: 1 if female else 0; capital-gain; capital-loss; hours-per-week; native-country; target: 1 if >50K else 0;

# # Let's create our X and y variables

# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


# Taking a sample from the entire population
sample = df.iloc[:10000]

X = sample.drop('target', axis=1)
y = sample['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# # Data Imputation

# In[20]:


from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Let's import our model

# In[21]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


# Pipeline to impute missing data and convert categorical columns into numbers

# In[22]:


categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
categorical_imputer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

numerical_features = ['age', 'fnlwgt', 'education-num', 'sex', 'capital-loss', 'capital-gain', 'hours-per-week']
numerical_imputer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=-1))])

preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_imputer, categorical_features),
    ('num', numerical_imputer, numerical_features)])


# # Functions to get the base scores for each model

# Please notice that my computer doesn't have enough power to fit all the dataset entries, so I've reduced to a sample of 10K only

# In[23]:


def get_base_prediction(model_dict):
    scores = {}
    for name, model in model_dict.items():
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)])

        model.fit(X_train, y_train)
        scores[name] = f'{model.score(X_test, y_test) * 100}%'

    return scores

get_base_prediction({'random_forest': RandomForestClassifier(),
                     'k_neighbors': KNeighborsClassifier(),
                     'linear_svc': LinearSVC()})


# # Hyperparameters tuning

# In[24]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# # RandomizedSearch CV

# In[25]:


rf_grid = {
    'model__n_estimators': np.arange(250, 350, 5),
    'model__max_depth': np.arange(30, 60, 5),
    'model__min_samples_split': np.arange(5, 15, 5),
    'model__min_samples_leaf': np.arange(0, 10, 1)
}

model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestClassifier(n_jobs=-1))])

rscv = RandomizedSearchCV(model, param_distributions=rf_grid, n_iter=30)

#rscv.fit(X_train, y_train)
#print(rscv.best_params_)
#print(f'{rscv.score(X_test, y_test) * 100}%')


# In[26]:


rsrf_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestClassifier(n_jobs=-1, n_estimators=285, min_samples_split=10,                                                                      min_samples_leaf=2, max_depth=45))])

rsrf_model.fit(X_train, y_train)
print(f'Tuned model score: {rsrf_model.score(X_test, y_test) * 100}%')


# # GridSearch CV

# In[27]:


rf_grid = {
    'model__n_estimators': [250, 280, 300],
    'model__max_depth': [45, 50, 55],
    'model__min_samples_split': [10, 12],
    'model__min_samples_leaf': [1, 3, 5]
}

model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestClassifier(n_jobs=-1))])

gscv = GridSearchCV(model, param_grid=rf_grid, n_jobs=-1)
gscv.fit(X_train, y_train)
print(gscv.best_params_)
print(f'{gscv.score(X_test, y_test) * 100}%')


# # Model Evaluation

# In[28]:


from sklearn.metrics import classification_report


# In[29]:


y_preds = rsrf_model.predict(X_test)
pd.DataFrame(classification_report(y_test, y_preds, output_dict=True))


# In[ ]:




