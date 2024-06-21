import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, State
import pandas as pd
import geopandas as gpd
import dash_leaflet as dl
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import pearsonr
import pickle

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.BOOTSTRAP,
    ],
)

server = app.server

app.title = "Dream Land"

df = pd.read_csv("src/data.csv")
df_c = pd.read_csv("src/cleaned_data.csv")
df_a = df_c.dropna()
correlation_matrix = np.corrcoef(df_c.iloc[:, 3:].values.T)


model = pickle.load(open("src/model.pkl", "rb"))
scaler = pickle.load(open("src/scaler.pkl", "rb"))


navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("〈GitHub〉", href="https://github.com/MohamedAziz-Khezami/Dream_Land")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("LinkedIn", href="linkedin.com/in/mohamed-aziz-khezami-160523252"),
                dbc.DropdownMenuItem("Facebook", href="https://www.facebook.com/aziz.khezami.14"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
    ],
    brand="🏞 The Dream Land",
    brand_href="/",
    color="dark",
    dark=True,
)
collapse = dbc.Collapse(
    dbc.Container(
        [
            html.H5("This content will toggle based on the button"),
            html.P(
                "This section will be hidden when the navbar is collapsed and shown when expanded."
            ),
        ]
    ),
    id="navbar-collapse",
    is_open=False,
)

sidebar = html.Div(
    [
        dbc.Offcanvas(
            [
                html.P(
                    "Adjust the values to predict the GDP",
                    className="display-4 fw-bold",
                ),
                html.Hr(),
                html.Br(),
                html.P("FDI"),
                dcc.Slider(
                    id="fdi",
                    min=round(df_a["FDI"].min(), 2),
                    max=round(df_a["FDI"].max(), 2),
                    value=df_a["FDI"].mean(),  # Initial value
                    step=None,  # Increment by 0.5
                ),
                html.P("Inflation"),
                dcc.Slider(
                    id="inflation",
                    min=df_a["inflation"].min(),
                    max=df_a["inflation"].max(),
                    value=df_a["inflation"].mean(),  # Initial value
                    step=None,  # Increment by 0.5
                ),
                html.P("Literacy Rate"),
                dcc.Slider(
                    id="literacy",
                    min=df_a["literacy_rate"].min(),
                    max=df_a["literacy_rate"].max(),
                    value=df_a["literacy_rate"].mean(),  # Initial value
                    step=None,  # Increment by 0.5
                ),
                html.P("CO2 emissions"),
                dcc.Slider(
                    id="co2",
                    min=df_a["CO2"].min(),
                    max=df_a["CO2"].max(),
                    value=df_a["CO2"].mean(),  # Initial value
                    step=None,  # Increment by 0.5
                ),
                html.P("Internet users"),
                dcc.Slider(
                    id="internet",
                    min=df_a["Number of Internet users"].min(),
                    max=df_a["Number of Internet users"].max(),
                    value=df_a["Number of Internet users"].mean(),  # Initial value
                    step=None,  # Increment by 0.5
                ),
                html.P("Labor Force"),
                dcc.Slider(
                    id="labor",
                    min=df_a["Labor force"].min(),
                    max=df_a["Labor force"].max(),
                    value=df_a["Labor force"].mean(),  # Initial value
                    step=None,  # Increment by 0.5
                ),
                html.P("female of total population"),
                dcc.Slider(
                    id="ftp",
                    min=df_a["female_of_total_pop_mainDF"].min(),
                    max=df_a["female_of_total_pop_mainDF"].max(),
                    value=df_a["female_of_total_pop_mainDF"].mean(),  # Initial value
                    step=None,  # Increment by 0.5
                ),
                html.P("Ict development index"),
                dcc.Slider(
                    id="ict",
                    min=df_a["ict_development_index"].min(),
                    max=df_a["ict_development_index"].max(),
                    value=df_a["ict_development_index"].mean(),  # Initial value
                    step=None,  # Increment by 0.5
                ),
                html.P("Energy production"),
                dcc.Slider(
                    id="energy",
                    min=df_a["energy_production"].min(),
                    max=df_a["energy_production"].max(),
                    value=df_a["energy_production"].mean(),  # Initial value
                    step=None,  # Increment by 0.5
                ),
                html.P("Exchange rate"),
                dcc.Slider(
                    id="exchange",
                    min=df_a["exchange_rate"].min(),
                    max=df_a["exchange_rate"].max(),
                    value=df_a["exchange_rate"].mean(),  # Initial value
                    step=None,  # Increment by 0.5
                ),
            ],
            id="offcanvas",
            is_open=False,
            style={
                "border-radius": "0 15px 15px 0",
            },
        ),
        dbc.Button(
            html.I(className="bi bi-caret-right"),
            outline=True,
            color="primary",
            size="md",
            id="open-offcanvas",
            n_clicks=0,
            className="mt-2",
        ),
    ],
    style={"top": "0", "position": "sticky", "margin-right": "20px", "width": "27rem"},
)

figure = dcc.Graph(id="chloromap", config={"displayModeBar": False})

plot = dcc.Graph(id="scatter-plot", config={"displayModeBar": False})


app.layout = html.Div(
    [
        navbar,
        collapse,
        dbc.Container(
            [

                html.P(
                    "This project aims to predict the GDP of a fictional country with certain resources and values and see what are the most correlated variables with the wealth of a country. Enjoy!",
                    className="mt-2"
                    
                ),
                html.H2("Chloropleth Map"),
                dcc.Dropdown(
                    id="selection",
                    options=df.columns[3:],
                    value="GDP",  # Initially select all lines
                    multi=False,  # Allow multiple selections
                ),
                figure,
                html.H2("Correlation Scatter"),
                html.Div(
                    [   
                        
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="plot-select-1",
                                        options=df.columns[3:],
                                        value="GDP",  # Initially select all lines
                                        multi=False,  # Allow multiple selections
                                        style={"felx": "50%"},
                                    ),
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="plot-select-2",
                                        options=df.columns[3:],
                                        value="FDI",  # Initially select all lines
                                        multi=False,  # Allow multiple selections
                                    ),
                                ),
                            ],
                        ),
                        html.Div(
                            [
                               
                                plot,
                                html.H6(
                                    children=[
                                        html.Span("Correlation: "),
                                        html.Span(id="corr"),
                                    ]
                                ),
                                html.H6(
                                    children=[
                                        html.Span("P_Value: "),
                                        html.Span(id="pv"),
                                    ]
                                ),
                            ]
                        ),
                    ],
                ),
                html.H2("Heatmap of correlation"),
                dcc.Graph(
                    figure=px.imshow(
                        correlation_matrix,
                        x=list(df_c.columns[3:]),  # Set column names as x-axis labels
                        y=list(df_c.columns[3:]),  # Set column names as y-axis labels
                        color_continuous_scale="Reds",  # Set color scale
                    ),
                    config={"displayModeBar": False},
                ),
                html.Div(
                    [
                        html.H2("GDP predictor"),
                        sidebar,
                        dbc.Row(
                            dbc.Col(
                                html.H6(
                                    children=[
                                        html.Span("Prediction: $"),
                                        html.Span(id="prediction"),
                                    ],
                                    className="ps-6",
                                ),
                                style={"text-align":"center"}
                            ),

                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(
                                        id="redar", config={"displayModeBar": False}
                                    ),
                                ),
                            ],
                            align="center",
                        ),
                    ],
                ),
            ],
            fluid=True,
            className="ms-1 ",
            style={"height": "100%", "border-left": "1px solid"},
        ),
    ],
    className="m-0 p-0 w-100  me-0 ",
)


@app.callback(
    Output("redar", "figure"),
    Output("prediction", "children"),
    Input("fdi", "value"),
    Input("inflation", "value"),
    Input("literacy", "value"),
    Input("co2", "value"),
    Input("internet", "value"),
    Input("labor", "value"),
    Input("ftp", "value"),
    Input("ict", "value"),
    Input("energy", "value"),
    Input("exchange", "value"),
)
def correlation(a, b, c, d, e, f, g, h, i, j):

    a_ = (a - df_a["FDI"].min()) / (df_a["FDI"].max() - df_a["FDI"].min())
    b_ = (b - df_a["inflation"].min()) / (
        df_a["inflation"].max() - df_a["inflation"].min()
    )
    c_ = (c - df_a["literacy_rate"].min()) / (
        df_a["literacy_rate"].max() - df_a["literacy_rate"].min()
    )
    d_ = (d - df_a["CO2"].min()) / (df_a["CO2"].max() - df_a["CO2"].min())
    e_ = (e - df_a["Number of Internet users"].min()) / (
        df_a["Number of Internet users"].max() - df_a["Number of Internet users"].min()
    )
    f_ = (f - df_a["Labor force"].min()) / (
        df_a["Labor force"].max() - df_a["Labor force"].min()
    )
    g_ = (g - df_a["female_of_total_pop_mainDF"].min()) / (
        df_a["female_of_total_pop_mainDF"].max()
        - df_a["female_of_total_pop_mainDF"].min()
    )
    h_ = (h - df_a["ict_development_index"].min()) / (
        df_a["ict_development_index"].max() - df_a["ict_development_index"].min()
    )
    i_ = (i - df_a["energy_production"].min()) / (
        df_a["energy_production"].max() - df_a["energy_production"].min()
    )
    j_ = (j - df_a["exchange_rate"].min()) / (
        df_a["exchange_rate"].max() - df_a["exchange_rate"].min()
    )

    fig = go.Figure(
        data=go.Scatterpolar(
            r=[a_, b_, c_, d_, e_, f_, g_, h_, i_, j_],
            theta=[
                "Foreign Direct Investment",
                "Inflation Rate",
                "Literacy Rate",
                "CO2 Emissions",
                "Number of Internet Users",
                "Labour Force",
                "Female of Total Population",
                "ICT Development Index",
                "Energy Production",
                "Exchange Rate",
            ],
            fill="toself",
            name="1",
        ),
    )

    input_array = np.array([a, b, c, d, e, f, g, h, i, j]).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)

    # Print shapes of input data arrays

    # Make prediction
    pred = model.predict(input_array_scaled)

    return fig, pred


@app.callback(
    Output("corr", "children"),
    Output("pv", "children"),
    Input("plot-select-1", "value"),
    Input("plot-select-2", "value"),
)
def correlation(s1, s2):

    correlation, p_value = pearsonr(df_a[s1], df_a[s2])
    return round(correlation, 3), p_value


@app.callback(
    Output("scatter-plot", "figure"),
    Input("plot-select-1", "value"),
    Input("plot-select-2", "value"),
)
def plot_updater(s1, s2):

    fig = go.Figure(
        data=[
            go.Scatter(
                x=df_c[s2],
                y=df_c[s1],
                mode="markers",
            )
        ],  # Adjust mode (markers, lines, etc.)
    )
    fig.update_layout(
        xaxis_title=s2,
        yaxis_title=s1,
    )

    return fig


@app.callback(
    Output("chloromap", "figure"),
    Input("selection", "value"),
)
def map_updater(value):
    fige = px.choropleth(
        df,
        locations="Country Code",
        color=value,
        hover_name=value,
        projection="equirectangular",
        color_continuous_scale="Reds",
    )

    return fige


@app.callback(
    Output("offcanvas", "is_open"),
    Input("open-offcanvas", "n_clicks"),
    [State("offcanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open


@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
)
def toggle_collapse(n_clicks):
    if n_clicks:
        return not n_clicks  # Toggle on each click
    return False


if __name__ == "__main__":
    app.run_server(debug=True)
