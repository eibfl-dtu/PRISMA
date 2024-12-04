import streamlit as st
from PIL import Image
import subprocess
import time


col1, col2 = st.columns([1, 3])

col1.image(Image.open("./gui2/assets/logo.png"))
col2.title("PRISMA")
col2.markdown("An app for high-throughput processing of spectra.")
col2.markdown(
    "Eibar Flores | [The BIG-MAP Project](https://www.big-map.eu/)\
               | [Cite](https://doi.org/10.1002/cmtd.202100094)."
)
st.markdown("## Select a procesing pipeline")

pipeline_col1, pipeline_col2, _, _ = st.columns([1, 1, 1, 1])

with pipeline_col1:
    st.image(
        Image.open("./gui2/assets/icon_preprocessing.png"),
        use_container_width=True
    )
    pipeline_preprocessing_selected = st.button(
        "Pre-processing", use_container_width=True
    )
    st.markdown(
        "Trimming, outlier rejection, downsampling,\
                baseline correction."
    )


with pipeline_col2:
    st.image(Image.open("./gui2/assets/icon_fit.png"),
             use_container_width=True)
    pipeline_fit_selected = st.button("Peak Fitting", use_container_width=True)
    st.markdown("Fitting with Lorentz, Gaussian and pseud-Voight peaks.")


if pipeline_preprocessing_selected:
    with st.spinner("The preprocessing app will open in a new tab."):
        time.sleep(3)
        subprocess.Popen(["streamlit", "run", "./gui2/app_preprocessing.py"])
    st.success("App opened")


if pipeline_fit_selected:
    with st.spinner("The peak fitting app will open in a new tab."):
        time.sleep(3)
        subprocess.Popen(["streamlit", "run", "./gui2/app_peak_fitting.py"])
    st.success("App opened")
