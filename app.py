from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import base64
import io
import pandas as pd

app = Dash(__name__)

app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-data-upload')
])

@app.callback(
    Output('output-data-upload', 'children'),
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

            # Return HTML output
            return html.Div([
                html.H5(f'File: {name}'),
                html.P(f'Rows: {len(df)}'),
                html.P(f'Columns: {", ".join(df.columns)}'),
                html.P(f'Total Missing Values (Before): {df.isnull().sum().sum()}'),
                html.P(f'Total Missing Values (After): {df_filled.isnull().sum().sum()}'),
                html.P(f'Columns after One-Hot Encoding: {", ".join(df_encoded.columns)}'),
                html.H6('First 5 Rows of Processed Data (One-Hot Encoded):'),
                html.Pre(df_encoded.head().to_csv(index=False))
            ])
        except Exception as e:
            return html.Div([
                html.H5(f"Error processing file: {name}"),
                html.P(str(e))
            ])

    return html.Div('No file uploaded yet.')

if __name__ == '__main__':
    app.run_server(debug=True)
