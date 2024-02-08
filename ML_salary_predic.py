# Importation des modules nécessaires
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from plotly.offline import iplot, plot
import plotly.offline as pyo
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ignorer tous les avertissements pour une sortie plus propre
warnings.filterwarnings("ignore")

# Lecture des données à partir du fichier Excel
df = pd.read_excel('jobs_in_data.xlsx')

# Affichage de quelques échantillons de données pour vérifier leur lecture
df.sample(10)

# Vérification des valeurs manquantes dans chaque colonne
df.isna().sum()

# Analyse de la distribution des catégories d'emplois
df_job_category_general = df['job_category'].value_counts()
df_job_category_general


# Affichage du top et du bas des catégories d'emplois
print(f"Top Job Needed in Four Year'{df_job_category_general.idxmax()}' with Value '{df_job_category_general.max()}")
print(f"Least Job Needed in Four Year'{df_job_category_general.idxmin()}' with Value '{df_job_category_general.min()}")

# Visualisation de la distribution des catégories d'emplois
iplot(
    px.bar(
        df_job_category_general,
        labels={'job_category': 'Job Category'},
        title=f'Needed of Job Category in 4 years ago',
        color_discrete_sequence=['#b3079c'],
        template='plotly_dark',
        text_auto=True
    )
)

# Description statistique des données
df.describe()

# Groupement des catégories d'emplois par année
df_job_category = df.groupby('work_year')['job_category'].value_counts()

# Création de graphiques de barres pour chaque année
colors = ['#ccaa14','#8807b3','#07b324','#1007b3']

j=0

for i in range(2020,2024):
    fig = px.bar(
        df_job_category.get(i),
        labels={
            'job_category':'Job Category',
            'value':'Numbre of Employees'
        },title=f'Needed of Job Category in {i}',
        color_discrete_sequence=[colors[j]],
        template='plotly_dark',text_auto=True
    )
    pyo.iplot(fig)
    j+=1

print(j)


df_job_category = df.groupby('work_year')['job_category'].value_counts()

colors = ['#ccaa14','#8807b3','#07b324','#1007b3']

j=0

for i in range(2020,2024):
    fig = px.bar(
        df_job_category.get(i),
        labels={
            'job_category':'Job Category',
            'value':'Numbre of Employees'
        },title=f'Needed of Job Category in {i}',
        color_discrete_sequence=[colors[j]],
        template='plotly_dark',text_auto=True
    )
    pyo.iplot(fig)
    j+=1

print(j)

df.groupby('work_year')['experience_level'].value_counts(normalize=True).sort_values(ascending=True).unstack('experience_level').plot(kind = 'bar', stacked=True)



df.sort_index().plot(kind='bar', stacked=True)




top_5_countries = df[df['work_year']==2023]['company_location'].value_counts(normalize = True).head(5)
top_5_countries




boxplot_dataframe = df[(df['work_year'] == 2023) & (df['company_location'].isin(top_5_countries.index))]
boxplot_dataframe



medians = boxplot_dataframe.groupby('company_location')['salary_in_usd'].median().sort_values(ascending=False).index
medians

plt.figure(figsize=(15,5))

sns.boxplot(
    data = boxplot_dataframe,
    x = 'company_location',
    y = 'salary_in_usd',
    hue = 'experience_level',
    order = medians
)
plt.show()

level_order = ['Entry-level', 'Mid-level', 'Senior', 'Executive']
top_paid_USA = boxplot_dataframe[boxplot_dataframe['company_location']=="United States"]
mean_pay = top_paid_USA.groupby(['job_category','experience_level'])['salary_in_usd'].mean().unstack('experience_level')
mean_pay = mean_pay[level_order].sort_values(by = 'Senior', ascending = False)
mean_pay_k = mean_pay/1000
plot_order = mean_pay_k.index



# Encodage des variables catégorielles pour la préparation des modèles
encoder = LabelEncoder()

df['job_category'] = encoder.fit_transform(df['job_category'])
df['experience_level'] = encoder.fit_transform(df['experience_level'])
df['job_title'] = encoder.fit_transform(df['job_title'])
df['salary_currency'] = encoder.fit_transform(df['salary_currency'])
df['employee_residence'] = encoder.fit_transform(df['employee_residence'])
df['employment_type'] = encoder.fit_transform(df['employment_type'])
df['work_setting'] = encoder.fit_transform(df['work_setting'])
df['company_location'] = encoder.fit_transform(df['company_location'])
df['company_size'] = encoder.fit_transform(df['company_size'])
df['company_size'] = encoder.fit_transform(df['company_size'])
df['company_size'] = encoder.fit_transform(df['company_size'])
df


df.describe()


df.columns.tolist()

df.shape

# Calcul de la matrice de corrélation entre les variables
correlation_matrix = df.corr()

# Créer une figure
plt.figure(figsize=(10, 8))

# Dessiner la heatmap de la matrice de corrélation
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

# Ajouter un titre
plt.title('Matrice de Corrélation')

# save the figure
plt.savefig('correlation_matrix.png')

# Afficher la figure
plt.show()

# Diviser les données en ensembles d'entraînement et de test
X = df.drop(['salary_in_usd'], axis=1)  # Features
y = df[['salary_in_usd']]  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)


# Initialiser les modèles pour chaque colonne cible
models_salary_in_usd = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
}

# Entraîner et évaluer de chaque modèle pour la colonne 'salary_in_usd'
for model_name, model in models_salary_in_usd.items():
    print(f"Training {model_name} for 'salary_in_usd' column...")
    model.fit(X_train, y_train['salary_in_usd'])
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test['salary_in_usd'], y_pred)
    mae = mean_absolute_error(y_test['salary_in_usd'], y_pred)
    r2 = r2_score(y_test['salary_in_usd'], y_pred)
    print(f"{model_name} - MSE: {mse}, MAE: {mae}, R²: {r2}")


# Supposons que vous ayez déjà vos données dans un DataFrame appelé df
# Par exemple, vos variables d'entrée sont dans une liste nommée X et votre variable de sortie est y

# Créer un graphique de dispersion pour chaque variable d'entrée par rapport à la variable de sortie
for variable in X:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=variable, y='salary_in_usd')
    plt.title(f"Relation entre {variable} et le salaire (USD)")
    # save figure
    plt.savefig(f'{variable}_vs_salary_in_usd.png')
    plt.xlabel(f"Variable d'entrée : {variable}")
    plt.ylabel("Salaire (USD)")
    plt.show()

