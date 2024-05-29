def visualize_contexts_(query, contexts, scores):
    """
    Visualizes the contexts retrieved for a given query.

    Args:
        query (str): The query for which contexts were retrieved.
        contexts (list): The list of contexts retrieved.
        scores (list): The list of scores for each retrieved context.

    Returns:
        None. Displays a bar chart using Plotly.
    """
    import plotly.express as px
    import pandas as pd

    # Define a function to truncate long text
    def truncate(text, max_length=100):
        """
        Truncates a given text to a specified maximum length.

        Args:
            text (str): The text to be truncated.
            max_length (int, optional): The maximum length of the truncated text. Defaults to 100.

        Returns:
            str: The truncated text.
        """
        return text if len(text) <= max_length else text[:max_length] + '...'

    # Truncate the contexts for display
    truncated_contexts = [truncate(context) for context in contexts]

    # Create a DataFrame from the input data
    data = {
        'Query': [query] * len(contexts),
        'Document': [f'Doc{i}' for i in range(1, len(contexts) + 1)],
        'Score': scores,
        'Truncated Context': truncated_contexts,
        'Full Context': contexts,
        'Index': list(range(1, len(contexts) + 1))
    }
    df = pd.DataFrame(data)
    df['Score'] = df['Score'].astype(float)  # Ensure 'Score' is of type float for sorting

    # Sort the DataFrame by 'Score' in ascending order
    df = df.sort_values(by='Score', ascending=True).reset_index(drop=True)

    title_text = f"Top-{len(contexts)} Retrieved Contexts<br><span style='font-size:14px'>Query: {query}</span>"

    fig = px.bar(df, x='Index', y='Score', color='Document', title=title_text,
                 hover_data={'Truncated Context': True, 'Full Context': False, 'Score': ':.2f', 'Document': False})

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            # tickvals=df['Index'],
            ticktext=df['Document']
        ),
        title={
            'text': title_text,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=False ,
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Rockwell"
        ),
        autosize=True,
        paper_bgcolor="LightSteelBlue"

    )

    return fig.show()