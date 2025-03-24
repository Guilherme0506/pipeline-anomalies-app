import plotly.express as px
import pandas as pd
import plotly.io as pio

# Set renderer for VS Code
pio.renderers.default = 'browser'

df = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [10, 11, 12, 13, 14]
})

fig = px.line(df, x='x', y='y', title='Basic')
fig.show()

