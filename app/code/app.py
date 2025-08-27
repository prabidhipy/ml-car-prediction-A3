import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import pickle
import os

# Load model, scaler, and column info
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'regression_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'scaler.pkl')
COLUMNS_PATH = os.path.join(os.path.dirname(__file__), '..', 'columns.pkl')

# Load pre-trained regression model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load scaler for numerical features
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Load columns used during training (important for matching dummy variables)
with open(COLUMNS_PATH, "rb") as f:
    columns = pickle.load(f)

# Load and preprocess original dataset
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'Cars.csv')
df_original = pd.read_csv(DATA_PATH)

# Simplify 'name' column to just the brand
df_original.rename(columns={'name': 'brand'}, inplace=True)
df_original['brand'] = df_original['brand'].str.split(' ').str[0]

# Drop rows with missing critical information
df_original.dropna(subset=['year', 'brand', 'transmission', 'owner'], inplace=True)

# Prepare dropdown options
year_options = sorted(df_original['year'].unique(), reverse=True)
brand_options = sorted(df_original['brand'].unique())
transmission_options = df_original['transmission'].unique()

# Owner dropdown mapping to numeric values used in model
owner_options = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner']
owner_map = {
    'First Owner': 1,
    'Second Owner': 2,
    'Third Owner': 3,
    'Fourth & Above Owner': 4,
}

# Separate numeric and categorical columns for preprocessing
num_cols = ['engine', 'owner', 'year']
cat_cols = ['brand', 'transmission']

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# App layout
app.layout = dbc.Container([
    # Title row
    dbc.Row(
        dbc.Col(html.H2("Car Price Prediction"), width={"size": 6, "offset": 3}),
        className="my-4"
    ),
    # Instructions row
    dbc.Row(
        dbc.Col(
            dbc.Alert(
                "Enter your car details below."
                "If you don’t know a value, just leave it blank and the system will use default values. "
                "Click 'Predict' to see the estimated price.",
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
                    # Engine input
                    dbc.Label("Engine (cc)"),
                    dbc.Input(id='engine', type='number', placeholder="e.g., 1500", value=1500, className="mb-3"),

                    # Year dropdown
                    dbc.Label("Year"),
                    dcc.Dropdown(id='year',
                                 options=[{'label': i, 'value': i} for i in year_options],
                                 value=2019,
                                 className="mb-3"),

                    # Brand dropdown
                    dbc.Label("Brand"),
                    dcc.Dropdown(id='brand',
                                 options=[{'label': i, 'value': i} for i in brand_options],
                                 value='Maruti',
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

                    # Predict button
                    dbc.Button("Predict", id='predict-btn', color="primary", className="mt-2", n_clicks=0),

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
    # Do nothing if button not clicked
    if n_clicks == 0:
        return ""
    
    # Convert owner string to numeric
    owner_numeric = owner_map[owner]

    # Create DataFrame from user input
    df = pd.DataFrame([[engine, owner_numeric, year, brand, transmission]],
                      columns=num_cols + cat_cols)
    
    # Scale numeric columns
    df[num_cols] = scaler.transform(df[num_cols])
    
    # Convert categorical columns to one-hot encoding
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    
    # Ensure all columns used during training are present
    df = df.reindex(columns=columns, fill_value=0)

    # Predict log(price) and convert to original scale
    y_pred_log = model.predict(df)
    y_pred = np.exp(y_pred_log)
    
    # Display prediction nicely
    return dbc.Alert(f"Predicted Price: ฿{y_pred[0]:,.2f}", color="success")

# Run app
if __name__ == "__main__":
    app.run(debug=True)
