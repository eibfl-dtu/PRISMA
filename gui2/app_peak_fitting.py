import streamlit as st
import os
import sys
import pandas as pd
from PIL import Image
import plotly.graph_objects as go


@st.cache_data
def set_wdir():
    # Add the root directory of your repository to the Python path
    current_directory = os.path.dirname(os.path.abspath(__file__))
    repository_root = os.path.join(current_directory, '..')
    sys.path.append(repository_root)

set_wdir()

from prisma import parsers1, fitpeaks
from prisma.spectrum1 import Spectrum







######################################### FUNCTIONS #########################################



@st.cache_data
def payload_to_spectra(payload, parser:str):
    """ Load spectra from the FileUpload streamlit widget, using prisma parsers"""

    if parser == 'Single .csv':
        spectra, spectra_metadata = parsers1.single_csv(payload.getvalue())
    elif parser == 'Single .txt (Bruker)':
        spectra, spectra_metadata = parsers1.single_txt_bruker(payload.getvalue())
    elif parser == 'Multiple .txt':
        spectra, spectra_metadata = parsers1.multiple_txt({spectrum_bits.name : spectrum_bits.getvalue() for spectrum_bits in payload})
    else:
        raise KeyError('The parser is not defined')
    
    return spectra, list(spectra.keys()), spectra_metadata



@st.cache_data
def update_preprocessing_parameters(spectra_metadata:dict = None):
    """ Keeps track of pre-processing parameters that are updated from metadata values form the spectra"""

    if not spectra_metadata: #use default preprocessing params
        preprocs_params = {
            "peak_limits": {
                "min_value":400,
                "max_value": 4000,
                "min_width":4,
                "max_widht":100
            }}
    else:
        min_index, max_index = spectra_metadata["energy_limits"]
        preprocs_params = {
                "peak_limits": {
                    "min_value":int(min_index),
                    "max_value": int(max_index),
                    "min_width": 3*spectra_metadata["min_resolvable_width"],
                    "max_width": int(0.1*(max_index-min_index))
                }}
        
    return preprocs_params


def peakfit_spectrum(spectrum:Spectrum, peak_lineshape:str, peak_data:pd.DataFrame):

    peak_bounds=list(zip(peak_data["Peak lower bound location"], peak_data["Peak upper bound location"]))
    guess_widths = peak_data["Approximate Width"].to_list()
    lineshape = peak_lineshape

    return fitpeaks(spectrum,
                    peak_bounds=peak_bounds,
                    guess_widths=guess_widths,
                    lineshape = lineshape)


######################################### MAIN APP LOOP  ###############################################


############## BOILERPLATE 

st.sidebar.image(Image.open("./gui2/assets/logo.png"))

st.header("Peak fitting")



################### FILE UPLOAD #####################


with st.sidebar.expander("Upload files"):

        parser = st.radio("Format of spectra", 
                                options=["Single .csv", "Single .txt (Bruker)", "Multiple .txt"], 
                                help="See documentation: what formats are accepted")
        
        multiple_files = True if parser == "Multiple .txt" else False

        uploaded_files = st.file_uploader("Choose CSV files", accept_multiple_files=multiple_files)



if uploaded_files:

    spectra, spectra_names, spectra_metadata = payload_to_spectra(uploaded_files, parser = parser)

    with st.sidebar.expander("Spectra metadata"):
        st.json(spectra_metadata)

    selected_spectrum_name:str = st.sidebar.selectbox("Select spectrum", spectra_names)

    current_spectrum = spectra[selected_spectrum_name]

    preprocs_params = update_preprocessing_parameters(spectra_metadata) 

else:
    current_spectrum = None
    spectra_metadata = None


################### PEAK FITTING SPECS #####################

peak_lineshape = st.selectbox(label="Peak lineshape", options=["Lorentzian", "Gaussian", r"Pseudo-Voight 50% Lorentzian"])


peak_data = st.data_editor(data = pd.DataFrame({"Peak lower bound location":[], "Peak upper bound location":[], "Approximate Width":[]}),
                hide_index=True,
                num_rows="dynamic",
                use_container_width=True)




############# PLOTS ###################

spectrum_container = st.container()
fig = go.Figure()
fig.update_xaxes(title_text="Index")
fig.update_yaxes(title_text="Counts")
fig.update_layout(template="simple_white",
                      legend=dict(
                        orientation="h",
                        entrywidth=70,
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1),
                    modebar_remove=["lasso2d", "displaylogo"])

if current_spectrum: 

    fig.add_trace(go.Scatter(x=current_spectrum.indexes,
                                 y=current_spectrum.counts, 
                                 name="Spectrum",
                                 mode="markers",
                                 marker={"color":"#455A64"}))
    
    
    if peak_data.empty or (peak_data.isnull().all(axis=1)).all():
        pass

    else:
        fit_spectrum = peakfit_spectrum(spectrum=current_spectrum, 
                                        peak_lineshape=peak_lineshape,
                                        peak_data=peak_data)
    
    
        fig.add_trace(go.Scatter(x=fit_spectrum.indexes,
                                    y=fit_spectrum.counts, 
                                    name="Peak fit", 
                                    mode="lines",
                                    marker={"color":"#FF1744"}))
    
        for peak_id, profile in fit_spectrum.profiles.items():

            fig.add_trace(go.Scatter(x=fit_spectrum.indexes,
                                    y=profile, 
                                    name=peak_id, 
                                    mode="lines",
                                    marker={"color":"#FF1744"},
                                    fill='tozeroy'))


    

spectrum_container.plotly_chart(fig, use_container_width=True, config={'displaylogo': False})
