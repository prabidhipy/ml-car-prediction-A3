import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import os

# --- NEW: Import the functions and variables from your model.py ---
# This assumes model.py is in the same directory (app/code/)
try:
    from model import predict_selling_price
except ImportError:
    print("Could not import from model.py. Make sure it's in the same directory.")

# --- Load and preprocess original dataset for dropdowns ---
# This part remains to populate the UI with options
DATA_PATH = 'data/Cars.csv'

try:
    df_original = pd.read_csv(DATA_PATH)
    # Simplify 'brand' column to just the brand name for the dropdown
    df_original.rename(columns={'name': 'brand'}, inplace=True)
    df_original['brand'] = df_original['brand'].str.split(' ').str[0]

    # Drop rows with missing critical information for clean dropdown options
    df_original.dropna(subset=['year', 'brand', 'transmission', 'owner'], inplace=True)

    # Prepare dropdown options
    year_options = sorted(df_original['year'].unique(), reverse=True)
    brand_options = sorted(df_original['brand'].unique())
    transmission_options = df_original['transmission'].unique()
    owner_options = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner']
except FileNotFoundError:
    print(f"Error: Could not find the data file at {DATA_PATH}. Dropdowns may be empty.")
    year_options, brand_options, transmission_options, owner_options = [], [], [], []


# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# App layout
app.layout = dbc.Container([
    # Title row
    dbc.Row(
        dbc.Col(html.H2("Car Price Category Prediction"), width={"size": 6, "offset": 3}),
        className="my-4"
    ),
    # Instructions row
    dbc.Row(
        dbc.Col(
            dbc.Alert(
                "Enter your car details below. The system will predict the price category "
                "(e.g., Cheap, Average, Expensive) based on the best model from our experiments.",
                color="info"
            ),
            width={"size": 6, "offset": 3}
        ),
        className="mb-4"
    ),
    # Input form row
    dbc.Row(
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    # --- REMOVED: Model selection dropdown ---

                    # Engine input
                    dbc.Label("Engine (cc)"),
                    dbc.Input(id='engine', type='number', placeholder="e.g., 1248", value=1248, className="mb-3"),

                    # Year dropdown
                    dbc.Label("Year"),
                    dcc.Dropdown(id='year',
                                 options=[{'label': i, 'value': i} for i in year_options],
                                 value=2014,
                                 className="mb-3"),
                    
                    # Owner dropdown
                    dbc.Label("Owner"),
                    dcc.Dropdown(id='owner',
                                 options=[{'label': i, 'value': i} for i in owner_options],
                                 value='First Owner',
                                 className="mb-3"),

                    # Transmission dropdown
                    dbc.Label("Transmission"),
                    dcc.Dropdown(id='transmission',
                                 options=[{'label': i, 'value': i} for i in transmission_options],
                                 value='Manual',
                                 className="mb-3"),

                    # Brand dropdown
                    dbc.Label("Brand"),
                    dcc.Dropdown(id='brand',
                                 options=[{'label': i, 'value': i} for i in brand_options],
                                 value='Maruti',
                                 className="mb-3"),

                    # Predict button
                    dbc.Button("Predict Price Category", id='predict-btn', color="primary", className="mt-2", n_clicks=0),

                    # Placeholder for prediction output
                    html.Div(id='prediction-output', className="mt-3")
                ])
            ), width={"size": 6, "offset": 3}
        )
    )
], fluid=True)

# Callback for prediction
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('engine', 'value'),
    State('owner', 'value'),
    State('year', 'value'),
    State('brand', 'value'),
    State('transmission', 'value')
)
def predict(n_clicks, engine, owner, year, brand, transmission):
    if n_clicks == 0:
        return ""

    # --- NEW: Simplified Prediction Logic ---
    # All the complex logic is now in the model.py file.
    # We just call the function with the user's inputs.
    try:
        # The imported function handles everything:
        # 1. Loading the production model from MLflow.
        # 2. Loading the correct scaler and encoders.
        # 3. Preprocessing the input data.
        # 4. Making a prediction.
        # 5. Returning the class and a user-friendly label.
        prediction, label = predict_selling_price(
            engine=engine,
            owner=owner,
            year=year,
            transmission=transmission,
            brand=brand
        )
        
        # Display the categorical prediction nicely
        return dbc.Alert(f"Predicted Price Category: {label}", color="success")

    except Exception as e:
        # If anything goes wrong in model.py (e.g., model not loaded), show an error.
        return dbc.Alert(f"An error occurred: {e}", color="danger")


# Run app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)