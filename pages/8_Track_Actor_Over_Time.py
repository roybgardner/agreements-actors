import streamlit as st

from streamlit_shared import *


# *********************************************************************************************************************

add_logo("./logos/peacerep_text.png")

st.header("Track an Actor Over Time")

# *********************************************************************************************************************

st.write("The video below shows UK activity in the Bosnia Peace Process. The\
         UK (CON_19) is the actor in the fixed position on the right of the screen")
st.write("Each frame shows the UK signatory network on the date shown in the frame's title.")
st.write("All agreements signed on a date are shown even if the UK was not a signatory.\
         However, there are only two dates in the sequence that contain agreements not signed by\
         the UK: 1992-08-27 AGT_1170, and 1995-11-21 AGT_1289.")

st.video('./movie.mp4', format="video/mp4", start_time=0, subtitles=None, end_time=None, loop=False)