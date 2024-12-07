import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input
from typing import Union, List, Dict
import dash_daq as daq


class Visualization:
    """
    A class to create and manage visualizations for model evaluation metrics.

    Attributes:
        mode (str): The mode of the visualization ('llm', 'safety', or 'rag').
        light_template (str): The light theme template for Plotly.
        dark_template (str): The dark theme template for Plotly.
        current_template (str): The current theme template for Plotly.
        models (list): A list of model data.
        plots (list): A list of plot types to be generated based on the mode and number of models.

    Methods:
        determine_plots(): Determines the types of plots to generate based on the mode and number of models.
        set_theme(theme: str): Sets the current theme template.
        create_radar_chart(): Creates a radar chart for the evaluation metrics.
        create_bar_chart(): Creates a bar chart for the evaluation metrics.
        create_gauge_chart(): Creates a gauge chart for the overall scores.
        create_scatter_plot(): Creates a scatter plot for the evaluation metrics.
        create_line_plot(): Creates a line plot for the evaluation metrics.
        create_heatmap(): Creates a heatmap for the evaluation metrics.
        create_violin_plot(): Creates a violin plot for the evaluation metrics.
        create_table(): Creates a table for the evaluation metrics.
        get_plot(plot_type: str): Returns the specified plot.
        create_layout(): Creates the layout for the Dash application.
        plot(): Runs the Dash application.
    """

    def __init__(self, data: Union[List[Dict], Dict], mode: str = 'llm', chart_interpretations=None):
        """
        Initializes the Visualization object with data and mode.

        Args:
            data (Union[List[Dict], Dict]): The data for the models.
            mode (str): The mode of the visualization ('llm', 'safety', or 'rag').
        """
        self.mode = mode
        self.light_template = "plotly_white"
        self.dark_template = "plotly_dark"
        self.current_template = self.light_template
        self.models = data if isinstance(data, list) else [data]
        self.plots = self.determine_plots()
        self.chart_interpretations = chart_interpretations

    def determine_plots(self):
        """
        Determines the types of plots to generate based on the mode and number of models.

        Returns:
            list: A list of plot types to be generated.
        """
        # if self.mode == 'llm':
        #     if len(self.models) == 1:
        #         return ['bar_chart', 'radar_chart', 'gauge_chart']
        #     else:
        #         return ['radar_chart', 'bar_chart', 'scatter_plot', 'line_plot', 'heatmap',
        #                 'gauge_chart', 'table']
        # elif self.mode == 'safety':
        #     if len(self.models) == 1:
        #         return ['bar_chart', 'gauge_chart']
        #     else:
        #         return ['radar_chart', 'bar_chart', 'gauge_chart']
        # elif self.mode == 'rag':
        #     return ['bar_chart', 'gauge_chart']
        if len(self.models) == 1:
            if self.mode == 'llm':
                return ['bar_chart', 'radar_chart', 'gauge_chart']
            elif self.mode == 'safety':
                return ['bar_chart', 'gauge_chart']
            elif self.mode == 'rag':
                return ['bar_chart', 'gauge_chart']

        else:
            return ['radar_chart', 'bar_chart', 'scatter_plot', 'line_plot', 'heatmap',
                    'gauge_chart', 'table']

    def set_theme(self, theme: str):
        """
        Sets the current theme template.

        Args:
            theme (str): The theme to set ('light' or 'dark').

        Raises:
            ValueError: If the theme is not 'light' or 'dark'.
        """
        if theme not in ('light', 'dark'):
            raise ValueError("Invalid theme. Must be 'light' or 'dark'.")

        self.current_template = self.dark_template if theme == 'dark' else self.light_template

    def create_radar_chart(self):
        """
        Creates a radar chart for the evaluation metrics.

        Returns:
            plotly.graph_objects.Figure: The radar chart figure.
        """
        fig = go.Figure()
        for model in self.models:
            name = model['name']
            metrics = model['metrics']
            labels = list(metrics.keys())
            values = list(metrics.values())
            values += values[:1]
            labels += labels[:1]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=labels,
                mode='lines+markers',
                line=dict(width=1),
                name=name
            ))

        fig.update_layout(
            title='Evaluation Metrics Radar Chart',
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            template=self.current_template
        )
        return fig

    def create_bar_chart(self):
        """
        Creates a bar chart for the evaluation metrics.

        Returns:
            plotly.express.Figure: The bar chart figure.
        """
        data = []
        for model in self.models:
            if len(self.models) == 1:
                metrics = model['metrics']
                for metric, value in metrics.items():
                    data.append({"Metric": metric, "Value": value})
                    df = pd.DataFrame(data)
                    fig = px.bar(df, x='Metric', y='Value', color='Metric', barmode='group', text='Value',
                                 template=self.current_template)
                fig.update_layout(
                    title='Evaluation Metrics Bar Chart',
                    xaxis_title='Metrics',
                    yaxis_title='Values',
                    xaxis_tickangle=-45,

                )
            else:
                name = model['name']
                metrics = model['metrics']
                for metric, value in metrics.items():
                    data.append({'Model': name, "Metric": metric, "Value": value})
                    df = pd.DataFrame(data)
                    fig = px.bar(df, x='Metric', y='Value', color='Model', barmode='group', text='Value',
                                 template=self.current_template)
                fig.update_layout(
                    title='Evaluation Metrics Bar Chart',
                    xaxis_title='Metrics',
                    yaxis_title='Values',
                    xaxis_tickangle=-45
                )
        return fig

    def create_gauge_chart(self):
        """
        Creates a gauge chart for the overall scores.

        Returns:
            plotly.subplots.make_subplots: The gauge chart figure.
        """
        fig = make_subplots(
            rows=1,
            cols=len(self.models),
            specs=[[{'type': 'indicator'}] * len(self.models)],
            subplot_titles=[model['name'] for model in self.models]
        )

        for i, model in enumerate(self.models, 1):
            name = model['name']
            score = model['score']

            if self.mode == 'llm' or 'rag':
                steps = [
                    {'range': [0, 0.2], 'color': "red"},
                    {'range': [0.2, 0.4], 'color': "orange"},
                    {'range': [0.4, 0.6], 'color': "yellow"},
                    {'range': [0.6, 0.8], 'color': "lightgreen"},
                    {'range': [0.8, 1], 'color': "green"}
                ]
            elif self.mode == 'safety':
                steps = [
                    {'range': [0, 0.2], 'color': "green"},
                    {'range': [0.2, 0.4], 'color': "lightgreen"},
                    {'range': [0.4, 0.6], 'color': "yellow"},
                    {'range': [0.6, 0.8], 'color': "orange"},
                    {'range': [0.8, 1], 'color': "red"}
                ]

            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=score,
                delta={'reference': 0.5, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [0, 1], 'tickwidth': 0.7, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': steps,
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': score
                    }
                }
            ), row=1, col=i)

        font = {'color': "white", 'family': "Arial"} if self.current_template == "plotly_dark" else {'color': "black",
                                                                                                     'family': "Arial"}
        paper_bgcolor = "black" if self.current_template == "plotly_dark" else "white"
        plot_bgcolor = "white" if self.current_template == "plotly_dark" else "black"

        fig.update_layout(
            title='Overall Evaluation Scores',
            font=font,
            paper_bgcolor=paper_bgcolor,
            plot_bgcolor=plot_bgcolor,
            template=self.current_template
        )
        return fig

    def create_scatter_plot(self):
        """
        Creates a scatter plot for the evaluation metrics.

        Returns:
            plotly.express.Figure: The scatter plot figure.
        """
        data = []
        for model in self.models:
            name = model['name']
            metrics = model['metrics']
            for metric, value in metrics.items():
                data.append({"Model": name, "Metric": metric, "Value": value})

        df = pd.DataFrame(data)

        fig = px.scatter(df, x='Value', y='Metric', color='Model', size='Value', hover_name='Model',
                         template=self.current_template)
        fig.update_layout(
            title='Evaluation Metrics Scatter Plot',
            xaxis_title='Value',
            yaxis_title='Metric'
        )
        return fig

    def create_line_plot(self):
        """
        Creates a line plot for the evaluation metrics.

        Returns:
            plotly.express.Figure: The line plot figure.
        """
        data = []
        for model in self.models:
            name = model['name']
            metrics = model['metrics']
            for metric, value in metrics.items():
                data.append({"Model": name, "Metric": metric, "Value": value})

        df = pd.DataFrame(data)

        fig = px.line(df, x='Metric', y='Value', color='Model', markers=True, template=self.current_template)
        fig.update_layout(
            title='Evaluation Metrics Line Plot',
            xaxis_title='Metrics',
            yaxis_title='Values'
        )
        return fig

    def create_heatmap(self):
        """
        Creates a heatmap for the evaluation metrics.

        Returns:
            plotly.express.Figure: The heatmap figure.
        """
        data = []
        for model in self.models:
            name = model['name']
            metrics = model['metrics']
            for metric, value in metrics.items():
                data.append({"Model": name, "Metric": metric, "Value": value})

        df = pd.DataFrame(data)

        heatmap_data = df.pivot(index="Model", columns="Metric", values="Value")
        fig = px.imshow(heatmap_data, aspect="auto", color_continuous_scale="Viridis",
                        template=self.current_template)

        fig.update_layout(
            title='Evaluation Metrics Heatmap',
            xaxis_title='Metrics',
            yaxis_title='Models'
        )
        return fig

    def create_violin_plot(self):
        """
        Creates a violin plot for the evaluation metrics.

        Returns:
            plotly.express.Figure: The violin plot figure.
        """
        data = []
        for model in self.models:
            name = model['name']
            metrics = model['metrics']
            for metric, value in metrics.items():
                data.append({"Model": name, "Metric": metric, "Value": value})

        df = pd.DataFrame(data)

        fig = px.violin(df, x='Metric', y='Value', color='Model', box=True, points="all",
                        template=self.current_template)

        fig.update_layout(
            title='Evaluation Metrics Violin Plot',
            xaxis_title='Metrics',
            yaxis_title='Values'
        )
        return fig

    def create_table(self):
        """
        Creates a table for the evaluation metrics.

        Returns:
            plotly.graph_objects.Figure: The table figure.
        """
        data = []
        for model in self.models:
            row = {'Model': model['name']}
            row.update(model['metrics'])
            data.append(row)

        df = pd.DataFrame(data)
        text_color = 'white' if self.current_template == self.dark_template else 'black'
        cells_color = 'black' if self.current_template == self.dark_template else 'white'
        fig = go.Figure(data=[go.Table(
            columnwidth=[150] * len(df.columns),
            header=dict(values=list(df.columns),
                        fill_color='paleturquoise',
                        align='left',
                        font=dict(color="black", size=12)),
            cells=dict(values=[df[col].tolist() for col in df.columns],
                       fill_color='lavender',
                       align='left',
                       font=dict(size=10, color="black"))
        )])

        fig.update_layout(
            title='Evaluation Metrics Table',
            autosize=True,
            margin=dict(l=0, r=0, t=30, b=0),
            template=self.current_template
        )
        return fig

    def get_plot(self, plot_type):
        plot_methods = {
            'radar_chart': self.create_radar_chart,
            'bar_chart': self.create_bar_chart,
            'scatter_plot': self.create_scatter_plot,
            'line_plot': self.create_line_plot,
            'heatmap': self.create_heatmap,
            'violin_plot': self.create_violin_plot,
            'gauge_chart': self.create_gauge_chart,
            'table': self.create_table
        }
        if plot_type in plot_methods:
            return plot_methods[plot_type]()
        else:
            raise ValueError(f"Plot type {plot_type} not recognized.")

    def create_layout(self):
        app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, dbc.themes.DARKLY, dbc.icons.BOOTSTRAP])

        nav_items = [
            dbc.NavItem(dbc.NavLink(plot.replace('_', ' ').title(), href=f"#{plot}", className="nav-link",
                                    external_link=True, id=f"nav-{plot}"))
            for plot in self.plots
        ]
        if not self.chart_interpretations:
            cards = [
                dbc.Card([
                    dbc.CardHeader(html.H4(plot.replace('_', ' ').title(), id=plot, className="display-4 cart-title")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(dcc.Graph(id=f"graph-{plot}"), width=8),
                            dbc.Col(html.P(
                                f"This {plot.replace('_', ' ')} displays data for the model{'s' if len(self.models) > 1 else ''}.",
                                className="card-text p-3"), width=4)
                        ])
                    ])
                ], className="mb-4", id=f"card-{plot}")
                for plot in self.plots
            ]
        else:
            cards = [
                dbc.Card([
                    dbc.CardHeader(html.H4(plot.replace('_', ' ').title(), id=plot, className="display-4 cart-title")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(dcc.Graph(id=f"graph-{plot}"), width=8),
                            dbc.Col(html.P(
                                self.chart_interpretations.get(plot.replace('_', ' ').title(),
                                                               f"This {plot.replace('_', ' ')} displays data for the model{'s' if len(self.models) > 1 else ''}."),
                                className="card-text p-3"), width=4)
                        ])
                    ])
                ], className="mb-4", id=f"card-{plot}")
                for plot in self.plots
            ]

        app.layout = html.Div([
            html.Link(href='/assets/style.css', rel='stylesheet'),
            dcc.Location(id='url', refresh=False),
            dbc.Container([
                dbc.Row([
                    dbc.Col(html.H2("IndoxJudge", className="text-center display-4", id="title-text"), width=10),
                    dbc.Col([
                        html.I(className="bi bi-brightness-high-fill"),
                        daq.ToggleSwitch(id="dark-mode-switch", value=False, className="my-4 out modebtn"),
                        html.I(className="bi bi-moon-fill"),
                    ], className='change-bg', width=2),
                ], className='header', align="center"),
                dbc.Row([
                    dbc.Col([
                        dbc.Nav(nav_items, pills=True, className="bg-light-custom stylish-nav justify-content-center",
                                id="nav-container"),
                    ], width=12),
                ], className="mb-4"),
                dbc.Row([dbc.Col(cards, width=12, className="mb-4")]),
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Go to Top", className="btn-primary btn-gtt position-fixed bottom-0 end-0 m-4",
                                   href="#url")
                    ], width=12, className="d-flex justify-content-end btn-gtt-container")
                ])
            ], fluid=True, id='main-container', className='bg-dark-custom')
        ])

        return app

    def plot(self, mode="external"):
        app = self.create_layout()

        @app.callback(
            [Output('main-container', 'className'),
             Output('title-text', 'className'),
             Output('nav-container', 'className')] +
            [Output(f'graph-{plot}', 'figure') for plot in self.plots],
            [Input('dark-mode-switch', 'value')]
        )
        def update_theme_and_graphs(dark_mode):
            if dark_mode:
                self.set_theme('dark')
                container_class = 'bg-dark-custom'
                title_class = 'text-custom-primary-dark'
                nav_class = 'navbar-custom-dark'
            else:
                self.set_theme('light')
                container_class = 'bg-light-custom'
                title_class = 'text-custom-primary'
                nav_class = 'navbar-custom'

            plots = [self.get_plot(plot) for plot in self.plots]

            return [container_class, title_class, nav_class] + plots

        app.run(jupyter_mode=mode)
