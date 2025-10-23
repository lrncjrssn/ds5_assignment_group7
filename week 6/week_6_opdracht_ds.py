import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# het volgende is gemaakt door laurence:
#opdracht 1
#functie om synthetische data te genereren voor lineaire regressie
#functie om een lineair regressiemodel te trainen en evalueren
#opdracht 2
#de dataset winequality-red.csv is ingeladen
#de data is verkend en gecontroleerd op missende waarden en outliers
#de verdeling van de wijnkwaliteit en de correlaties tussen de variabelen zijn gevisualiseerd
#de dataset is gesplitst in een training- en testset
#een lineair regressiemodel is getraind om de kwaliteit van de wijn te voorspellen
#het model is geëvalueerd met de r²-score en de mean squared error
#de prestaties van het model zijn besproken en er is aangegeven dat een complexer model mogelijk beter is
#opdracht 3
#de bestanden training.xlsx en predictions_training.xlsx zijn ingeladen
#een functie is geschreven om de intersection over union (iou) tussen twee bounding boxes te berekenen
#een tweede functie is gemaakt die de gemiddelde iou over alle bounding boxes berekent
#de gemiddelde iou is berekend en de resultaten zijn weergegeven


## opdracht 1

def generate_data(n_samples=200, seed=2):
    """
    Genereer synthetische lineaire regressiegegevens.

    Gemaakt door Laurence.

    Parameters
    ----------
    n_samples : int, default=200
        Aantal te genereren steekproeven.
    seed : int, default=2
        Zaad voor de random generator (np.random.seed) voor reproduceerbaarheid.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 1)
        Onafhankelijke variabele (features).
    y : np.ndarray, shape (n_samples,)
        Doelwaarde met lineaire relatie y = 2.5 * x + ruis (normale ruis, sigma ≈ 3).
    """
    np.random.seed(seed)
    x = np.random.uniform(0, 10, n_samples)
    y = 2.5 * x**2 - 5*x + 3 + np.random.normal(0, 10, n_samples)
    return x, y


def fit_regression_model(x, y):
    """
    Train een lineair regressiemodel op X en y, evalueer op een testset en return het getrainde model.

    Gemaakt door Laurence.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        Feature-matrix.
    y : array-like, shape (n_samples,)
        Doelvariabele.

    Returns
    -------
    model : sklearn.linear_model.LinearRegression
        Het getrainde LinearRegression-model.

    Notes
    -----
    - Er wordt een train/test-split gebruikt met test_size=0.2 en random_state=2.
    - De functie print R^2 en MSE voor de testset.
    """
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"R^2 Score: {r2:.3f}")
    print(f"MSE: {mse:.3f}")

    return model

##opdracht 2
x, y = generate_data(n_samples=200, seed=2)
model = fit_regression_model(x.reshape(-1, 1), y)

print (x,y, model)
# 1. Dataset inladen

df = pd.read_csv("winequality-red.csv")

# 2. Basisinformatie bekijken
print(df.info())
print(df.describe())

# 3. Missende waarden controleren
print(df.isnull().sum())

# 4. Basisverkenning
plt.figure(figsize=(6,4))
sns.countplot(x='quality', data=df)
plt.title('Verdeling van wijnkwaliteit')
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlatiematrix')
plt.show()

# 5. Enkele scatterplots
sns.pairplot(df, vars=['alcohol', 'volatile acidity', 'citric acid'], hue='quality')
plt.show()

# 6. Train-test split
X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Model 1 trainen (Linear Regression)
model1 = LinearRegression()
model1.fit(X_train, y_train)

# 8. Voorspellen en evalueren
y_pred = model1.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\nModel 1 (Linear Regression) resultaten:")
print(f"R^2 score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

# 9. Resultaten interpreteren
if r2 < 0.4:
    print("Het model verklaart relatief weinig variantie in de data.")
else:
    print("Het model presteert redelijk goed. Verdere tuning kan nodig zijn.")


## opdracht 3

# 1. Inladen van de data 
train_df = pd.read_excel("training.xlsx")
pred_df = pd.read_excel("predictions_training.xlsx")  

print(train_df.head())
print(pred_df.head())

# 2. IOU-functie
def calculate_iou(box1, box2):
    """
    Bereken de Intersection over Union (IoU) tussen twee bounding boxes.

    Gemaakt door Laurence.

    Parameters
    ----------
    box1 : sequence of 4 numbers
        Bounding box in de volgorde [min_r, min_c, max_r, max_c].
    box2 : sequence of 4 numbers
        Bounding box in dezelfde volgorde als box1.

    Returns
    -------
    float
        De IoU-waarde (tussen 0 en 1). Geeft 0 terug wanneer er geen overlap is
        of wanneer de unie nul is.
    """
    xA = max(box1[1], box2[1])
    yA = max(box1[0], box2[0])
    xB = min(box1[3], box2[3])
    yB = min(box1[2], box2[2])

    # Bereken overlappende breedte/hoogte
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    intersection = interWidth * interHeight

    # Bereken oppervlaktes
    box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
    box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])

    # IOU
    union = box1_area + box2_area - intersection
    if union == 0:
        return 0
    iou = intersection / union
    return iou

# 3. Gemiddelde IOU berekenen
def mean_iou(true_df, pred_df):
    """
    Bereken de gemiddelde Intersection over Union (IoU) voor gepaarde bounding boxes.

    Gemaakt door Laurence.

    Parameters
    ----------
    true_df : pandas.DataFrame
        DataFrame met de echte bounding boxes. Verwacht kolommen:
        ['min_r', 'min_c', 'max_r', 'max_c'] in deze volgorde.
    pred_df : pandas.DataFrame
        DataFrame met de voorspelde bounding boxes. Zelfde kolomnamen en -orde
        als true_df wordt verwacht.

    Returns
    -------
    tuple
        (mean_iou, ious)
        mean_iou : float
            Gemiddelde IoU over alle rijen.
        ious : list of float
            Lijst met IoU-waarden per rij.

    Notes
    -----
    - Beide DataFrames worden verondersteld gelijke lengte en bijbehorende rijen
      te hebben (same index/order).
    - Roept calculate_iou(box1, box2) aan voor de IoU-berekening per paar.
    """
    ious = []
    for i in range(len(true_df)):
        true_box = true_df.loc[i, ['min_r','min_c','max_r','max_c']].values
        pred_box = pred_df.loc[i, ['min_r','min_c','max_r','max_c']].values
        iou = calculate_iou(true_box, pred_box)
        ious.append(iou)
    return np.mean(ious), ious

mean_iou_value, ious = mean_iou(train_df, pred_df)

# 4. Resultaten tonen
print(f"\nGemiddelde IOU over alle bounding boxes: {mean_iou_value:.4f}")
print(f"Eerste 10 IOUs: {ious[:10]}")



