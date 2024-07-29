from bokeh.io import curdoc
from bokeh.layouts import row, column, gridplot, Spacer
from bokeh.models import ColumnDataSource, Select, Legend, Div, ColorBar, LinearColorMapper
from bokeh.plotting import figure
from bokeh.transform import factor_cmap, transform
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = '../titanic3.xls'
df = pd.read_excel(file_path)

# Data cleaning
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df = df.drop(columns=['cabin'])
df = df.drop_duplicates()

# Feature engineering
df['FamilySize'] = df['sibsp'] + df['parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Encoding categorical variables
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['embarked'], drop_first=True)

# Scaling numerical features
scaler = StandardScaler()
df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

# Convert DataFrame to ColumnDataSource
source = ColumnDataSource(data=dict(age=df['age'], fare=df['fare'], sex=df['sex'], FamilySize=df['FamilySize']))

# Create a scatter plot using scatter() method
plot = figure(title="Age vs Fare", x_axis_label='Age', y_axis_label='Fare', tools="pan,wheel_zoom,box_zoom,reset")
scatter = plot.scatter('age', 'fare', source=source, size=10, color="navy", alpha=0.5)

# Create dropdown menus for selecting x and y axes
x_select = Select(title="X Axis", value="age", options=list(df.columns))
y_select = Select(title="Y Axis", value="fare", options=list(df.columns))

# Update function to change the axis
def update_axis(attr, old, new):
    source.data = {col: df[col] for col in df.columns if col in source.data.keys()}
    plot.xaxis.axis_label = x_select.value
    plot.yaxis.axis_label = y_select.value
    plot.title.text = f"{x_select.value} vs {y_select.value}"

x_select.on_change('value', update_axis)
y_select.on_change('value', update_axis)

# Add explanatory text for encoded and scaled features
explanation = Div(text="""
<h2 class="centered-header">Feature Explanation</h2>
<ul class="centered-list">
<li><b>Sex:</b> Encoded as 0 (male) and 1 (female)</li>
<li><b>Embarked:</b> One-hot encoded for C (Cherbourg) and Q (Queenstown)</li>
<li><b>Age and Fare:</b> Scaled using StandardScaler</li>
</ul>
""", width=800, height=100, css_classes=["centered-header"])

# Create additional plots using Bokeh

# Plot the distribution of 'age'
hist, edges = np.histogram(df['age'], bins=30)
p1 = figure(title="Distribution of Age", x_axis_label='Age', y_axis_label='Frequency')
p1.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="navy", line_color="white", alpha=0.7)

# Plot the survival rate by 'sex'
sex_grouped = df.groupby('sex')['survived'].mean().reset_index()
sex_source = ColumnDataSource(sex_grouped)
p2 = figure(x_range=['0', '1'], title="Survival Rate by Sex", x_axis_label='Sex', y_axis_label='Survival Rate')
p2.vbar(x='sex', top='survived', width=0.9, source=sex_source, legend_label="Sex", line_color='white', fill_color=factor_cmap('sex', palette=["navy", "firebrick"], factors=['0', '1']))

# Plot the survival rate by 'pclass'
pclass_grouped = df.groupby('pclass')['survived'].mean().reset_index()
pclass_source = ColumnDataSource(pclass_grouped)
p3 = figure(x_range=['1', '2', '3'], title="Survival Rate by Passenger Class", x_axis_label='Passenger Class', y_axis_label='Survival Rate')
p3.vbar(x='pclass', top='survived', width=0.9, source=pclass_source, legend_label="Pclass", line_color='white', fill_color="navy")

# Plot the correlation heatmap
numeric_cols = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_cols.corr()

correlation_data = correlation_matrix.stack().reset_index(name='value')
correlation_data.columns = ['x', 'y', 'value']
heatmap_source = ColumnDataSource(correlation_data)

colors = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"]
mapper = LinearColorMapper(palette=colors, low=-1, high=1)

p4 = figure(title="Correlation Heatmap", x_axis_label='', y_axis_label='', x_range=list(correlation_matrix.columns), y_range=list(correlation_matrix.index), width=800, height=400)
p4.rect(x="x", y="y", width=1, height=1, source=heatmap_source, line_color=None, fill_color=transform('value', mapper))

# Add text labels to the heatmap
text_source = ColumnDataSource(data=dict(x=correlation_data['x'], y=correlation_data['y'], value=[f'{v:.2f}' for v in correlation_data['value']]))
p4.text(x='x', y='y', text='value', source=text_source, text_align="center", text_baseline="middle", text_font_size="10pt")

color_bar = ColorBar(color_mapper=mapper, location=(0, 0))
p4.add_layout(color_bar, 'right')

# Arrange plots and widgets in layouts
header = row(Spacer(width=350), explanation, Spacer(width=350))
plots = [
    [plot, p1],
    [p2, p3],
    [p4, Spacer(width=350)]
]

grid = gridplot(plots, toolbar_location=None)
layout = column(header, row(x_select, y_select), grid)
curdoc().add_root(layout)
