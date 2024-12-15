from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import base64
import io
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


processed = None
trained_model = None
selected_features = None

app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Store(id='stored-data'),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px','backgroundColor':'gray'
        },

        multiple=False
    ),
    html.Div(id='output-data-upload'),
    
    html.H3("Group 13: Lakshita Madhavan, Neelaza Dahal, Jane Kalla, and Michelle Perez", style={'textAlign': 'center', 'marginTop': '20px'})


    html.Div([
        html.Label('Select Target Variable:   '),
        dcc.Dropdown(id='target-dropdown', options=[], value=None,style={'width': '150px','display': 'inline-block'} )
    ], style={
        'margin': '10px',
        'textAlign': 'center',        
        'backgroundColor': '#D3D3D3',   
        'padding': '20px',            
        'borderRadius': '8px'  
    }),

    html.Div([
        html.Label('Select Categorical Variable:'),


        dcc.RadioItems(id='categorical-radio', options=[], value=None)
    ], style={'margin': '10px'}),

    html.Div([
        dcc.Graph(id='category-average-chart', style={'flex': 1, 'margin-right': '10px'}),

        dcc.Graph(id='correlation-chart', style={'flex': 1, 'margin-left': '10px'})

    ], style={'display': 'flex', 'justify-content': 'space-between'}),

    html.Div([
        html.Div([
            html.Label('Select Feature Variables:'),
            dcc.Checklist(id='feature-checklist', options=[], value=[]),
            html.Button('Train Model', id='train-button', n_clicks=0),
            html.Div(id='model-output'),

            html.Label('Prediction Input (comma-separated):'),
            dcc.Input(id='predict-input', type='text', value=''),

            html.Button('Predict', id='predict-button', n_clicks=0),
            html.Div(id='prediction-output')
        ])
    ], style={'textAlign': 'center', 'marginTop': '20px'})
])


@app.callback(
    [
        Output('output-data-upload', 'children'),
        Output('stored-data', 'data'),
        Output('target-dropdown', 'options'),
        Output('target-dropdown', 'value'),
        Output('categorical-radio', 'options'),

        Output('categorical-radio', 'value'),
        Output('feature-checklist', 'options'),
        Output('feature-checklist', 'value')
    ],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
)
def update_output(content, name, date):
    if content is not None:
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

            df_filled = df.fillna(df.mean(numeric_only=True))
            for col in df.select_dtypes(include=['object', 'category']):

                if df[col].isnull().any():
                    mode_value = df[col].mode().iloc[0]
                    df_filled[col] = df[col].fillna(mode_value)

            categorical_columns = df.select_dtypes(include=['object', 'category']).columns
            label_encoder = LabelEncoder()

            if 'size' in df_filled.columns:
                df_filled['size'] = label_encoder.fit_transform(df_filled['size'])

            df_encoded = pd.get_dummies(df_filled, columns=[col for col in categorical_columns if col != 'size'], drop_first=True)


            global processed
            processed = df_filled
            global cats 
            cats = df_encoded
            numeric_columns = df.select_dtypes(include=[np.number]).columns

            target_options = [{'label': col, 'value': col} for col in numeric_columns]
            categorical_options = [{'label': col, 'value': col} for col in categorical_columns]

            feature_options = [{'label': col, 'value': col} for col in df_filled.columns if col != '']

            output_children = html.Div([
                
            ])

            return output_children, df_filled.to_dict('records'), target_options, None, categorical_options, None, feature_options, []

        except Exception as e:
            return html.Div(f"Error processing file: {e}"), None, [], None, [], None, [], []

    return html.Div('No file uploaded yet.'), None, [], None, [], None, [], []


@app.callback(
    Output('category-average-chart', 'figure'),
    [Input('target-dropdown', 'value'),
     Input('categorical-radio', 'value')]
)
def update_category_average_chart(target_variable, categorical_variable):
    if processed is None or target_variable is None or categorical_variable is None:
        return {}

    try:
        category_avg = processed.groupby(categorical_variable)[target_variable].mean().reset_index()
        fig = px.bar(
            category_avg,
            x=categorical_variable,
            y=target_variable,
            title=f'Average {target_variable} by {categorical_variable}',
            labels={categorical_variable: "Category", target_variable: "Average Value"}
        )
        return fig

    except Exception as e:
        return {}


@app.callback(
    Output('correlation-chart', 'figure'),
    Input('target-dropdown', 'value')
)
def update_correlation_chart(target_variable):
    if processed is None or target_variable is None:
        return {}

    try:
        numeric_df = cats.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        target_correlations = corr_matrix[target_variable].abs().sort_values(ascending=False)[1:]

        fig = px.bar(
            x=target_correlations.index,
            y=target_correlations.values,
            title=f'Absolute Correlation with {target_variable}',
            labels={'x': 'Features', 'y': 'Absolute Correlation'}
        )
        return fig

    except Exception as e:
        return {}

@app.callback(
    Output("model-output", "children"),
    [Input("train-button", "n_clicks")],
    [State("feature-checklist", "value"),
     State("target-dropdown", "value")]
)
def train_model(n_clicks, features, target):
    global processed, trained_model, selected_features

    if n_clicks == 0 or processed is None or not features or target is None:
        return "Enter inputs"

    try:
        selected_features = features
        X = processed[features]


        y = processed[target]

        numeric_features = X.select_dtypes(include=[np.number]).columns


        categorical_features = X.select_dtypes(exclude=[np.number]).columns

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(

            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ]
        )


        model = Pipeline(steps=[
            ("preprocessor", preprocessor),

            ("regressor", LinearRegression())
        ])

        model.fit(X, y)

        trained_model = model

        r2 = r2_score(y, model.predict(X))

        return f"R² Score: {r2:.4f}"

    except Exception as e:

        return f"training error {str(e)}"
    


@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),

    [State("predict-input", "value")]
)


def predict(submit, input_values):
    global trained_model, selected_features

    if submit == 0 or trained_model is None:
        return "Click Predict to submit"

    try:
        input_values = [value.strip() for value in input_values.split(",")]

        if len(input_values) != len(selected_features):
            return "Errors - inputs don't match the dataset"

        input_dict = {}
        for feature, value in zip(selected_features, input_values):
            try:
                input_dict[feature] = float(value)
            except ValueError:
                input_dict[feature] = value

        input_df = pd.DataFrame([input_dict])

        prediction = trained_model.predict(input_df)[0]
        return f"Prediction: {prediction:.4f}"

    except Exception as e:
        return "Prediction error"


if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8051)

