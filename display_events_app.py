import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
from streamlit.file_util import streamlit_write
from cmcrameri import cm
from matplotlib.colors import Normalize
import base64
import os

DATA_URL = (
    os.path.join(os.getcwd(), 'all_events.csv')
)

st.sidebar.title("Mesetas Earthquake Sequence with AI_picker")

st.sidebar.markdown("This application shows all earthquakes generated automatically by "
            "AI_picker 😎 over the stations: ARAMC, DORAC, CLEJA, "
            "URMC, CLBC, TAPM, CHI, PRA, ORTC, PIRM, MACC, VIL")


@st.cache(persist=True)
def load_data():
    # Loading data as a pandas DataFrame retriving only the columns of our interest
    df = pd.read_csv(DATA_URL, index_col=0, parse_dates=[6])
    df.sort_values(by='mag', ascending=False, inplace=True)
    df['mag_scale'] = 800 * (2 ** df.mag)
    df.rename(columns={'z':'depth [km]', 'mag':'magnitude', 'lat':'latitude', 'lon':'longitude',
            'phasecount':'phase count', 'stationcount':'station count',
            'z_e':'depth error [km]', 'lat_e':'latitude error [km]',
            'lon_e':'longitude error [km]', 't_e':'rms [s]',
            'min_dis':'minimum station distance [km]'}, inplace=True)
    return df


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download csv file</a>'
    return href



data = load_data()

st.title("Mesetas Earthquake Sequence with AI_picker")
st.markdown("Earthquakes localized with LOCSAT algorith with "
            "iaspei91 model")

show_table = st.sidebar.checkbox("Show raw data", True)

st.sidebar.subheader("Magnitud filter:")
mag_min = st.sidebar.number_input('Minimun magnitude', min_value=-1.0,
                                  max_value=10.0, step=0.1)

st.sidebar.subheader("Choose a Region:")
region = st.sidebar.radio('Region', ('Mesetas', 'all'))
if region == 'Mesetas':
    data = data.query('latitude >= 2.6 and latitude <= 4.0 \
                                and longitude >= -75.0 and longitude <= -73.5\
                                and magnitude >= @mag_min')
else:
    data = data.query('magnitude >= @mag_min')

st.markdown(get_table_download_link(data), unsafe_allow_html=True)

if show_table:
    st.write(data)

st.markdown(f"\nTotal earthquakes: **{len(data)}**")

st.sidebar.markdown("### Number of EQ by ...")
select = st.sidebar.selectbox('EQ property', ['magnitude', 'station count',
                                              'phase count', 'rms [s]',
                                              'depth [km]'], key='1')
st.markdown(f"## Number of EQ by {select}")

if select in ['magnitude', 'rms [s]', 'depth [km]']:
    counts, bins = np.histogram(data[select], bins=range(int(data[select].max()+1)))
    bins = 0.5 * (bins[:-1] + bins[1:])
    fig = px.bar(x=bins, y=counts, labels={'x':select, 'y':'Number of EQ'}, height=500)
else:
    count = data[select].value_counts()
    count_df = pd.DataFrame({select:count.index, 'Number of EQ':count.values})
    fig = px.bar(count_df, x=select, y='Number of EQ',  height=500)
st.plotly_chart(fig)

# map all points
#st.map(data)


#lat = st.sidebar.number_input('Latitude', -4, 15)
#lat_min = st.sidebar.slider('Mimimun Latitude', -4, 15)
#lat_max = st.sidebar.slider('Maximun Latitude', lat_min, 15)
#lon_min = st.sidebar.slider('Minimun Longitude', -85, -65)
#lon_max = st.sidebar.slider('Maximun Longitude', lon_min, -65)

#mag_min = st.sidebar.number_input('Minimun Magnitude', min_value=0.0, max_value=10.0)

#filtered_data = data.query('latitude >= @lat_min and latitude <= @lat_max \
#                            and longitude >= @lon_min and longitude <= @lon_max\
#                            and magnitude >= @mag_min')"""

st.sidebar.subheader("Map")
style = st.sidebar.selectbox('Style', ['outdoors-v11', 'streets-v11',
                                              'light-v9', 'light-v10',
                                              'dark-v10', 'satellite-v9',
                                              'satellite-streets-v11'],
                              key='2')

opacity = st.sidebar.slider('Color opacity', 0.0, 1.0, value=0.6)

st.header("Map")

# custom map
cmap = cm.batlow_r

# we need to normalize the depth in order to use
# matplotlib cmap properly
norm = Normalize(vmin=0, vmax=300)
data['norm'] = data['depth [km]'].apply(norm)

def get_color(depth_norm):
    # cmap returns a tuple of normalized rgba elements
    # we desnormalize and keep with the first three (rgb)
    c = cmap(depth_norm)
    return [x*255 for x in c][:-1]

data['color'] = data['norm'].apply(get_color)

data['dates_str'] = data['orig_time'].apply(lambda x: x.strftime('%Y-%m-%d %H-%M-%S'))
data['magnitude'] = data['magnitude'].round(1)
data['depth [km]'] = data['depth [km]'].round(1)

layers = [
    pdk.Layer(
        'ScatterplotLayer',
        data=data,
        pickable=True,
        stroked=True,
        opacity=opacity,
        get_position='[longitude, latitude]',
        get_radius='mag_scale',
        get_fill_color='color',
        get_line_color=[0, 0, 0],
        line_width_min_pixels=0.3
    )
]

r = pdk.Deck(
    tooltip={"text": "{dates_str}\n Magnitude: {magnitude}\n Depth: {depth [km]} km"},
    map_style=f'mapbox://styles/mapbox/{style}',
    initial_view_state=pdk.ViewState(
        latitude=3.0,
        longitude=-74.0,
        zoom=7,
        pitch=0,
    ),
    layers=layers
)
st.pydeck_chart(r)
