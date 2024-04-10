import streamlit as st

from streamlit_shared import *

# *********************************************************************************************************************

add_logo("./logos/peacerep_text.png")

st.header("Track an Actor Over Time")

# *********************************************************************************************************************

st.write("The video below shows UK engagement in the Bosnia Peace Process. The\
         UK (CON_19) is the actor in the fixed position on the right of the frame")
st.write("Each frame shows the UK's signatory network on the date shown in the frame's title.")
st.write("All agreements signed on a date are shown even if the UK was not a signatory.\
         However, there are only two dates in the sequence that contain agreements not signed by\
         the UK: 1992-08-27 AGT_1170, and 1995-11-21 AGT_1289.")
st.write("The video provides a glimpse into a wealth of possibilities: changing networks of co-signatories,\
         metadata analysis etc. The computation is fast enough that network diagrams can be generated in real-time.")

st.video('./movie.mp4', format="video/mp4", start_time=0, subtitles=None, end_time=None, loop=False)


st.write("The video below shows Bosnia and Herzegovina (CON_0) and UK (CON_19) co-signatory engagements in the Bosnia Peace Process.")
st.write("Bosnia and Herzegovina (CON_0) is a fixed position of the left of the frame, the UK (CON_19) on the right.")
st.write("This analysis could be extended to any number of co-signatories.")
st.video('./movie_uk_bosnia.mp4', format="video/mp4", start_time=0, subtitles=None, end_time=None, loop=False)


st.divider()
st.write(':violet[POTENTIAL FUNCTIONS]')
st.write(':violet[Interface for real-time generation and display of a time series of network diagrams.]')
st.write(':violet[Interactive network diagrams with reveal of actor and agreement data.]')
st.write(':violet[Side-by-side comparison of network digrams with different dates.]')
