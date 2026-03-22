import altair as alt
import pandas as pd
import streamlit as st
from vega_datasets import data

alt.data_transformers.disable_max_rows()
st.set_page_config(page_title="Altair Visualization Lab Dashboard", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    cars = data.cars().copy()
    cars["Year"] = pd.to_datetime(cars["Year"])
    cars["Model_Year"] = cars["Year"].dt.year.astype(str)
    return cars


def filter_cars(cars: pd.DataFrame, origins: list[str], year_range: tuple[int, int]) -> pd.DataFrame:
    years = cars["Year"].dt.year
    mask = cars["Origin"].isin(origins) & years.between(year_range[0], year_range[1])
    return cars.loc[mask].copy()


def metric_card(label: str, value: str):
    st.metric(label, value)


def chart_count_by_origin(df: pd.DataFrame):
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("Origin:N", title="Origin"),
        y=alt.Y("count():Q", title="Number of cars"),
        tooltip=["Origin:N", alt.Tooltip("count():Q", title="Count")],
    ).properties(title="Number of Cars by Origin", height=320)



def chart_count_by_origin_sorted(df: pd.DataFrame):
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("Origin:N", sort="-y", title="Origin"),
        y=alt.Y("count():Q", title="Number of cars"),
        color=alt.Color("Origin:N", legend=None),
        tooltip=["Origin:N", alt.Tooltip("count():Q", title="Count")],
    ).properties(title="Number of Cars by Origin (Sorted)", height=320)



def chart_hist_mpg(df: pd.DataFrame, bins: int):
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("Miles_per_Gallon:Q", bin=alt.Bin(maxbins=bins), title="Miles per Gallon"),
        y=alt.Y("count():Q", title="Count"),
        tooltip=[alt.Tooltip("count():Q", title="Count")],
    ).properties(title="Distribution of Miles per Gallon", height=320)



def chart_hist_hp(df: pd.DataFrame, bins: int):
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("Horsepower:Q", bin=alt.Bin(maxbins=bins), title="Horsepower"),
        y=alt.Y("count():Q", title="Count"),
        tooltip=[alt.Tooltip("count():Q", title="Count")],
    ).properties(title="Distribution of Horsepower", height=320)



def chart_scatter_hp_mpg(df: pd.DataFrame, point_size: int, opacity: float):
    return alt.Chart(df).mark_circle(size=point_size, opacity=opacity).encode(
        x=alt.X("Horsepower:Q", title="Horsepower"),
        y=alt.Y("Miles_per_Gallon:Q", title="Miles per Gallon"),
        color=alt.Color("Origin:N", title="Origin"),
        tooltip=["Name:N", "Origin:N", "Horsepower:Q", "Miles_per_Gallon:Q"],
    ).properties(title="Horsepower vs Miles per Gallon", height=360)



def chart_scatter_weight_acc(df: pd.DataFrame, point_size: int, opacity: float):
    return alt.Chart(df).mark_circle(size=point_size, opacity=opacity).encode(
        x=alt.X("Weight_in_lbs:Q", title="Weight (lbs)"),
        y=alt.Y("Acceleration:Q", title="Acceleration"),
        color=alt.Color("Origin:N", title="Origin"),
        tooltip=["Name:N", "Origin:N", "Weight_in_lbs:Q", "Acceleration:Q"],
    ).properties(title="Weight vs Acceleration", height=360)



def chart_avg_mpg_origin(df: pd.DataFrame):
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("Origin:N", title="Origin"),
        y=alt.Y("mean(Miles_per_Gallon):Q", title="Average MPG"),
        color=alt.Color("Origin:N", legend=None),
        tooltip=["Origin:N", alt.Tooltip("mean(Miles_per_Gallon):Q", title="Average MPG", format=".2f")],
    ).properties(title="Average Miles per Gallon by Origin", height=320)



def chart_boxplot_hp_origin(df: pd.DataFrame):
    return alt.Chart(df).mark_boxplot().encode(
        x=alt.X("Origin:N", title="Origin"),
        y=alt.Y("Horsepower:Q", title="Horsepower"),
        color=alt.Color("Origin:N", legend=None),
        tooltip=["Origin:N"],
    ).properties(title="Horsepower Distribution by Origin", height=340)



def chart_line_mpg_time(df: pd.DataFrame):
    return alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("Model_Year:O", title="Model Year"),
        y=alt.Y("mean(Miles_per_Gallon):Q", title="Average MPG"),
        tooltip=[
            alt.Tooltip("Model_Year:O", title="Year"),
            alt.Tooltip("mean(Miles_per_Gallon):Q", title="Average MPG", format=".2f"),
        ],
    ).properties(title="Average Miles per Gallon Over Time", height=340)



def chart_facet_hp_mpg(df: pd.DataFrame, point_size: int, opacity: float):
    return alt.Chart(df).mark_circle(size=point_size, opacity=opacity).encode(
        x=alt.X("Horsepower:Q", title="Horsepower"),
        y=alt.Y("Miles_per_Gallon:Q", title="Miles per Gallon"),
        tooltip=["Name:N", "Horsepower:Q", "Miles_per_Gallon:Q"],
    ).properties(width=220, height=220).facet(
        column=alt.Column("Origin:N", title="Origin")
    ).properties(title="Horsepower vs MPG, Faceted by Origin")



def chart_interactive_legend(df: pd.DataFrame, point_size: int):
    selection = alt.selection_point(fields=["Origin"], bind="legend")
    return alt.Chart(df).mark_circle(size=point_size).encode(
        x=alt.X("Horsepower:Q", title="Horsepower"),
        y=alt.Y("Miles_per_Gallon:Q", title="Miles per Gallon"),
        color=alt.condition(selection, "Origin:N", alt.value("lightgray")),
        tooltip=["Name:N", "Origin:N", "Horsepower:Q", "Miles_per_Gallon:Q"],
    ).add_params(selection).properties(title="Interactive Scatter Plot by Origin", height=360)



def chart_brush_link(df: pd.DataFrame, point_size: int, opacity: float):
    brush = alt.selection_interval()

    points = alt.Chart(df).mark_circle(size=point_size, opacity=opacity).encode(
        x=alt.X("Horsepower:Q", title="Horsepower"),
        y=alt.Y("Miles_per_Gallon:Q", title="Miles per Gallon"),
        color=alt.condition(brush, "Origin:N", alt.value("lightgray")),
        tooltip=["Name:N", "Origin:N", "Horsepower:Q", "Miles_per_Gallon:Q"],
    ).add_params(brush).properties(width=420, height=320, title="Brush points")

    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X("Origin:N", title="Origin"),
        y=alt.Y("count():Q", title="Selected cars"),
        color=alt.Color("Origin:N", legend=None),
    ).transform_filter(brush).properties(width=260, height=320, title="Counts in selection")

    return points | bars



def exercise_avg_hp_origin(df: pd.DataFrame):
    return alt.Chart(df).mark_bar().encode(
        x="Origin:N",
        y="mean(Horsepower):Q",
        color=alt.Color("Origin:N", legend=None),
        tooltip=["Origin:N", alt.Tooltip("mean(Horsepower):Q", format=".2f")],
    ).properties(title="Exercise: Average Horsepower by Origin", height=320)



def exercise_weight_hist(df: pd.DataFrame):
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("Weight_in_lbs:Q", bin=alt.Bin(maxbins=20), title="Weight (lbs)"),
        y=alt.Y("count():Q", title="Count"),
    ).properties(title="Exercise: Distribution of Weight", height=320)



def exercise_weight_mpg(df: pd.DataFrame, point_size: int, opacity: float):
    return alt.Chart(df).mark_circle(size=point_size, opacity=opacity).encode(
        x=alt.X("Weight_in_lbs:Q", title="Weight (lbs)"),
        y=alt.Y("Miles_per_Gallon:Q", title="Miles per Gallon"),
        color="Origin:N",
        tooltip=["Name:N", "Origin:N", "Weight_in_lbs:Q", "Miles_per_Gallon:Q"],
    ).properties(title="Exercise: Weight vs Miles per Gallon", height=360)



def exercise_line_hp_year(df: pd.DataFrame):
    return alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("Model_Year:O", title="Model Year"),
        y=alt.Y("mean(Horsepower):Q", title="Average Horsepower"),
        tooltip=[
            alt.Tooltip("Model_Year:O", title="Year"),
            alt.Tooltip("mean(Horsepower):Q", title="Average Horsepower", format=".2f"),
        ],
    ).properties(title="Exercise: Average Horsepower by Model Year", height=340)



def exercise_facet_mpg_origin(df: pd.DataFrame):
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("Miles_per_Gallon:Q", bin=alt.Bin(maxbins=15), title="Miles per Gallon"),
        y=alt.Y("count():Q", title="Count"),
        color=alt.Color("Origin:N", legend=None),
    ).properties(width=220, height=220).facet(column="Origin:N").properties(title="Exercise: MPG Distribution by Origin")



def exercise_interactive_weight_hp(df: pd.DataFrame, point_size: int):
    sel = alt.selection_point(fields=["Origin"], bind="legend")
    return alt.Chart(df).mark_circle(size=point_size).encode(
        x="Weight_in_lbs:Q",
        y="Horsepower:Q",
        color=alt.condition(sel, "Origin:N", alt.value("lightgray")),
        tooltip=["Name:N", "Origin:N", "Weight_in_lbs:Q", "Horsepower:Q"],
    ).add_params(sel).properties(title="Exercise: Interactive Weight vs Horsepower", height=360)


cars = load_data()
all_origins = sorted(cars["Origin"].dropna().unique().tolist())
year_min = int(cars["Year"].dt.year.min())
year_max = int(cars["Year"].dt.year.max())

st.title("Altair Visualization Techniques — Streamlit Dashboard")
st.caption("Based on the notebook: lab01_visualization_preparation.ipynb")

with st.sidebar:
    st.header("Controls")
    selected_origins = st.multiselect("Origins", all_origins, default=all_origins)
    selected_years = st.slider("Model year range", min_value=year_min, max_value=year_max, value=(year_min, year_max))
    point_size = st.slider("Point size", min_value=30, max_value=140, value=60, step=10)
    point_opacity = st.slider("Point opacity", min_value=0.2, max_value=1.0, value=0.65, step=0.05)
    histogram_bins = st.slider("Histogram max bins", min_value=8, max_value=40, value=20)
    st.markdown("---")
    st.markdown("Use the filters to discuss how the same visual encoding changes under different subsets of the data.")

filtered = filter_cars(cars, selected_origins, selected_years)

if filtered.empty:
    st.warning("No rows match the current filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Rows", f"{len(filtered):,}")
with c2:
    metric_card("Origins", str(filtered['Origin'].nunique()))
with c3:
    metric_card("Year range", f"{filtered['Year'].dt.year.min()}–{filtered['Year'].dt.year.max()}")
with c4:
    metric_card("Avg MPG", f"{filtered['Miles_per_Gallon'].mean():.2f}")

with st.expander("Show filtered data preview"):
    st.dataframe(filtered, use_container_width=True)

intro, distributions, relationships, summaries, time_and_facet, interactive, exercises = st.tabs([
    "Overview",
    "Distributions",
    "Relationships",
    "Aggregations",
    "Time & Faceting",
    "Interactivity",
    "Exercises",
])

with intro:
    st.subheader("What this dashboard demonstrates")
    st.write(
        "This app mirrors the charts from the notebook and groups them by visualization goal: comparison, distribution, relationship, time series, faceting, and interaction."
    )
    left, right = st.columns(2)
    with left:
        st.altair_chart(chart_count_by_origin(filtered), use_container_width=True)
    with right:
        st.altair_chart(chart_count_by_origin_sorted(filtered), use_container_width=True)

with distributions:
    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(chart_hist_mpg(filtered, histogram_bins), use_container_width=True)
    with col2:
        st.altair_chart(chart_hist_hp(filtered, histogram_bins), use_container_width=True)

with relationships:
    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(chart_scatter_hp_mpg(filtered, point_size, point_opacity), use_container_width=True)
    with col2:
        st.altair_chart(chart_scatter_weight_acc(filtered, point_size, point_opacity), use_container_width=True)

with summaries:
    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(chart_avg_mpg_origin(filtered), use_container_width=True)
    with col2:
        st.altair_chart(chart_boxplot_hp_origin(filtered), use_container_width=True)

with time_and_facet:
    st.altair_chart(chart_line_mpg_time(filtered), use_container_width=True)
    st.altair_chart(chart_facet_hp_mpg(filtered, point_size, point_opacity), use_container_width=True)

with interactive:
    st.altair_chart(chart_interactive_legend(filtered, point_size + 10), use_container_width=True)
    st.altair_chart(chart_brush_link(filtered, point_size, point_opacity), use_container_width=True)

with exercises:
    st.write("These reproduce the exercise-style plots from the end of the notebook.")
    e1, e2 = st.columns(2)
    with e1:
        st.altair_chart(exercise_avg_hp_origin(filtered), use_container_width=True)
    with e2:
        st.altair_chart(exercise_weight_hist(filtered), use_container_width=True)

    e3, e4 = st.columns(2)
    with e3:
        st.altair_chart(exercise_weight_mpg(filtered, point_size, point_opacity), use_container_width=True)
    with e4:
        st.altair_chart(exercise_line_hp_year(filtered), use_container_width=True)

    e5, e6 = st.columns(2)
    with e5:
        st.altair_chart(exercise_facet_mpg_origin(filtered), use_container_width=True)
    with e6:
        st.altair_chart(exercise_interactive_weight_hp(filtered, point_size + 10), use_container_width=True)

st.markdown("---")
st.markdown(
    "**Run locally:** `streamlit run streamlit_altair_dashboard.py`  \\\n    **Dependencies:** `pip install streamlit altair vega_datasets pandas`"
)
