from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import base64
import io
import pandas as pd
import numpy as np
import plotly.express as px

processed_df = None
app = Dash(__name__, suppress_callback_exceptions=True)
app.layout = html.Div([
    dcc.Store(id='stored-data'),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-data-upload'),
    html.Div([
        html.Label('Select Target Variable:'),
        dcc.Dropdown(id='target-dropdown', options=[], value=None)
    ], style={'margin': '10px'}),
    html.Div([
        html.Label('Select Categorical Variable:'),
        dcc.RadioItems(id='categorical-radio', options=[], value=None)
    ], style={'margin': '10px'}),
    dcc.Graph(id='category-average-chart'), 
    dcc.Graph(id='correlation-chart')
])


@app.callback(
    [
        Output('output-data-upload', 'children'),
        Output('stored-data', 'data'),
        Output('target-dropdown', 'options'),
        Output('target-dropdown', 'value'),
        Output('categorical-radio', 'options'),
        Output('categorical-radio', 'value')
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
            # Load the uploaded file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

            # Data preprocessing
            df_filled_numeric = df.fillna(df.mean(numeric_only=True))
            for col in df.select_dtypes(include=['object', 'category']):
                if df[col].isnull().any():
                    mode_value = df[col].mode().iloc[0]
                    df_filled_numeric[col] = df[col].fillna(mode_value)
            df_filled = df_filled_numeric

            # One-hot encoding 
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns
            df_encoded = pd.get_dummies(df_filled, columns=categorical_columns, drop_first=True)
            
            global processed_df
            processed_df = df_filled
            
            numeric_columns = df_filled.select_dtypes(include=[np.number]).columns
            target_options = [{'label': col, 'value': col} for col in numeric_columns]
            categorical_options = [{'label': col, 'value': col} for col in categorical_columns]
            
            output_children = html.Div([
                html.P(f'Rows: {len(df)}'),
                html.P(f'Columns: {", ".join(df.columns)}'),
                html.P(f'Total Missing Values (Before): {df.isnull().sum().sum()}'),
                html.P(f'Total Missing Values (After): {df_filled.isnull().sum().sum()}'),
                html.P(f'Columns after One-Hot Encoding: {", ".join(df_encoded.columns)}'),
                html.H6('First 5 Rows of Processed Data (One-Hot Encoded):'),
                html.Pre(df_encoded.head().to_csv(index=False))
            ])
            
            return output_children, df_filled.to_dict('records'), target_options, None, categorical_options, None

        except Exception as e:
            return html.Div([
                html.H5(f"Error processing file: {name}"),
                html.P(str(e))
            ]), None, [], None, [], None

    return html.Div('No file uploaded yet.'), None, [], None, [], None

@app.callback(
    Output('category-average-chart', 'figure'), 
    [Input('target-dropdown', 'value'),    
    Input('categorical-radio', 'value')]       
)
def update_category_average_chart(target_variable, categorical_variable):
    global processed_df
    
    if processed_df is None or target_variable is None or categorical_variable is None:
        return {} 

    try:
        category_avg = processed_df.groupby(categorical_variable)[target_variable].mean().reset_index()
        fig = px.bar(
            category_avg,
            x=categorical_variable,
            y=target_variable,
            title=f'Average {target_variable} by {categorical_variable}',
            labels={categorical_variable: "Category", target_variable: "Average Value"}
        )
        
        return fig

    except Exception as e:
        print(f"Error generating category-average chart: {e}")
        return {}

@app.callback(
    Output('correlation-chart', 'figure'),
    [Input('target-dropdown', 'value')]
)
def update_correlation_chart(target_variable):
    global processed_df
    
    if processed_df is None or target_variable is None:
        return {
            'data': [],
            'layout': {
                'title': 'Please select a target variable',
                'xaxis': {'visible': False},
                'yaxis': {'visible': False}
            }
        }

    try:
        numeric_df = processed_df.select_dtypes(include=[np.number])
        
        corr_matrix = numeric_df.corr()
        target_correlations = corr_matrix[target_variable].abs()
        target_correlations = target_correlations.sort_values(ascending=False)[1:]

        if target_correlations.empty:
            return {
                'data': [],
                'layout': {
                    'title': 'No correlations to display',
                    'xaxis': {'visible': False},
                    'yaxis': {'visible': False}
                }
            }

        fig = px.bar(
            x=target_correlations.index,
            y=target_correlations.values,
            title=f'Absolute Correlation with {target_variable}',
            labels={'x': 'Features', 'y': 'Absolute Correlation'}
        )
        
        return fig

    except Exception as e:
        print(f"Error generating correlation chart: {e}")
        return {
            'data': [],
            'layout': {
                'title': f'Error: {str(e)}',
                'xaxis': {'visible': False},
                'yaxis': {'visible': False}
            }
        }

if __name__ == '__main__':
    app.run_server(debug=True)