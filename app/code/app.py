import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import pickle
import os
 
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'regression_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'scaler.pkl')
COLUMNS_PATH = os.path.join(os.path.dirname(__file__), '..', 'columns.pkl')

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
with open(COLUMNS_PATH, "rb") as f:
    columns = pickle.load(f)

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'Cars.csv')
df_original = pd.read_csv(DATA_PATH)

df_original.rename(columns={'name': 'brand'}, inplace=True)
df_original['brand'] = df_original['brand'].str.split(' ').str[0]

df_original.dropna(subset=['year', 'brand', 'transmission', 'owner'], inplace=True)

year_options = sorted(df_original['year'].unique(), reverse=True)
brand_options = sorted(df_original['brand'].unique())
transmission_options = df_original['transmission'].unique()

owner_options = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner']

owner_map = {
    'First Owner': 1,
    'Second Owner': 2,
    'Third Owner': 3,
    'Fourth & Above Owner': 4,
}

num_cols = ['engine', 'owner', 'year']
cat_cols = ['brand', 'transmission']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(html.H2("Car Price Prediction"), width={"size": 6, "offset": 3}),
        className="my-4"
    ),
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
    dbc.Row(
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    dbc.Label("Engine (cc)"),
                    dbc.Input(id='engine', type='number', placeholder="e.g., 1500", value=1500, className="mb-3"), #setting a default value for engine size

                    dbc.Label("Year"),
                    dcc.Dropdown(id='year',
                                 options=[{'label': i, 'value': i} for i in year_options],
                                 value=2019,  # Seting a default value for year
                                 className="mb-3"),

                    dbc.Label("Brand"),
                    dcc.Dropdown(id='brand',
                                 options=[{'label': i, 'value': i} for i in brand_options],
                                 value='Maruti',  # Setting a default value for brand
                                 className="mb-3"),
                                 
                    dbc.Label("Owner"),
                    dcc.Dropdown(id='owner',
                                 options=[{'label': i, 'value': i} for i in owner_options],
                                 value='First Owner',  # Setting a default value for owner type
                                 className="mb-3"),

                    dbc.Label("Transmission"),
                    dcc.Dropdown(id='transmission',
                                 options=[{'label': i, 'value': i} for i in transmission_options],
                                 value='Manual',  # Setting a default value for transmission type
                                 className="mb-3"),

                    dbc.Button("Predict", id='predict-btn', color="primary", className="mt-2", n_clicks=0),
                    html.Div(id='prediction-output', className="mt-3")
                ])
            ), width={"size": 6, "offset": 3}
        )
    )
], fluid=True)

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
    
    owner_numeric = owner_map[owner]

    df = pd.DataFrame([[engine, owner_numeric, year, brand, transmission]],
                      columns=num_cols + cat_cols)
    
    df[num_cols] = scaler.transform(df[num_cols])
    
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    
    df = df.reindex(columns=columns, fill_value=0)

    y_pred_log = model.predict(df)
    
    y_pred = np.exp(y_pred_log)
    
    return dbc.Alert(f"Predicted Price: ฿{y_pred[0]:,.2f}", color="success")

if __name__ == "__main__":
    app.run(debug=True)