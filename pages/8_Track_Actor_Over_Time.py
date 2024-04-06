import streamlit as st

from streamlit_shared import *


# *********************************************************************************************************************

add_logo("./logos/peacerep_text.png")

st.header("Track an Actor Over Time")

# *********************************************************************************************************************

st.video('./movie.mp4', format="video/mp4", start_time=0, subtitles=None, end_time=None, loop=False)