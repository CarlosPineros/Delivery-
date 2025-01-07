import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


Delivery_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(Delivery_dir, 'Data')
csv_path = os.path.join(data_dir, 'Food_Delivery_Times.csv')


class CustomerExperienceModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_and_preprocess_data(self):
        self.df = pd.read_csv(self.data_path)

        # Normalizar las variables
        self.df['Norm_Delivery_Time'] = ((self.df['Delivery_Time_min'] - self.df['Delivery_Time_min'].min()) /
                                         (self.df['Delivery_Time_min'].max() - self.df['Delivery_Time_min'].min()))
        self.df['Norm_Distance'] = ((self.df['Distance_km'] - self.df['Distance_km'].min()) /
                                    (self.df['Distance_km'].max() - self.df['Distance_km'].min()))
        self.df['Norm_Courier_Experience'] = ((self.df['Courier_Experience_yrs'] -
                                              self.df['Courier_Experience_yrs'].min()) /
                                              (self.df['Courier_Experience_yrs'].max() -
                                               self.df['Courier_Experience_yrs'].min()))

        # Ponderaciones y clima
        weather_score = {'Clear': 1.0, 'Windy': 0.8, 'Foggy': 0.6, 'Rainy': 0.4}
        self.df['Weather_Score'] = self.df['Weather'].map(weather_score)

        delivery_weight = -0.5
        distance_weight = -0.3
        experience_weight = 0.4

        # Calcular puntuación de experiencia
        self.df['Customer_Experience'] = (
            delivery_weight * self.df['Norm_Delivery_Time'] +
            distance_weight * self.df['Norm_Distance'] +
            experience_weight * self.df['Norm_Courier_Experience'] +
            self.df['Weather_Score']
        )

        # Escalar entre 1 y 10
        self.df['Customer_Experience'] = (
            (self.df['Customer_Experience'] - self.df['Customer_Experience'].min()) /
            (self.df['Customer_Experience'].max() - self.df['Customer_Experience'].min()) * 9 + 1
        ).round(2)

    def prepare_model_data(self):
        x = self.df[['Delivery_Time_min', 'Distance_km', 'Courier_Experience_yrs', 'Weather_Score']]
        y = self.df['Customer_Experience']

        # Limpiar valores faltantes
        x = x.fillna(x.median())
        data_cleaned = pd.concat([x, y], axis=1).dropna()
        x = data_cleaned.iloc[:, :-1]
        y = data_cleaned.iloc[:, -1]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    def train_and_evaluate_model(self):
        self.model = RandomForestRegressor(random_state=42, n_estimators=100)
        self.model.fit(self.x_train, self.y_train)

        y_pred = self.model.predict(self.x_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print(f"MSE: {mse:.4f}, R^2: {r2:.4f}")

    def feature_importance(self):
        importances = self.model.feature_importances_
        features = self.x_train.columns
        feature_importance_df = (pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values
                                 (by='Importance', ascending=False))

        # Visualizar la importancia de características
        plt.figure(figsize=(8, 5))
        sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis')
        plt.title('Importancia de Características')
        plt.xlabel('Importancia')
        plt.ylabel('Características')
        plt.show()

    def plot_predictions(self):
        y_pred = self.model.predict(self.x_test)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=self.y_test, y=y_pred, alpha=0.7, color="blue")
        plt.title("Predicciones vs Valores Reales")
        plt.xlabel("Valores Reales (Experiencia del Cliente)")
        plt.ylabel("Predicciones (Experiencia del Cliente)")
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()],
                 'r--', lw=2, label="Línea Ideal")
        plt.legend()
        plt.grid(True)
        plt.show()

    def predict_new_data(self, input_data):
        model_predictions = self.model.predict(input_data)
        input_data['Predicted_Customer_Experience'] = model_predictions.round(2)
        return input_data


if __name__ == "__main__":
    model = CustomerExperienceModel(data_path=csv_path)

    model.load_and_preprocess_data()
    model.prepare_model_data()

    model.train_and_evaluate_model()
    model.feature_importance()
    model.plot_predictions()

    # Fictional data to test the model
    new_data = pd.DataFrame({
        'Delivery_Time_min': [30, 45, 60],
        'Distance_km': [5, 15, 25],
        'Courier_Experience_yrs': [1, 3, 5],
        'Weather_Score': [1.0, 0.8, 0.4]
    })
    predictions = model.predict_new_data(input_data=new_data)
    print(predictions)
