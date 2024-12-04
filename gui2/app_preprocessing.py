import streamlit as st
import sys
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import os


@st.cache_data
def set_wdir():
    # Add the root directory of your repository to the Python path
    current_directory = os.path.dirname(os.path.abspath(__file__))
    repository_root = os.path.join(current_directory, "..")
    sys.path.append(repository_root)


set_wdir()

from prisma.util import parsers
from prisma.spectrum import Spectrum


################## FUNCTIONS #####################


@st.cache_data
def payload_to_spectra(payload, parser: str):
    """Load spectra from the FileUpload streamlit widget, using prisma parsers"""

    if parser == "Single .csv":
        spectra, spectra_metadata = parsers.single_csv(payload.getvalue())
    elif parser == "Single .txt (Bruker)":
        spectra, spectra_metadata = parsers.single_txt_bruker(
            payload.getvalue()
        )
    elif parser == "Multiple .txt":
        spectra, spectra_metadata = parsers.multiple_txt(
            {
                spectrum_bits.name: spectrum_bits.getvalue()
                for spectrum_bits in payload
            }
        )
    else:
        raise KeyError("The parser is not defined")

    return spectra, list(spectra.keys()), spectra_metadata


@st.cache_data
def update_preprocessing_parameters(spectra_metadata: dict = dict()):
    """Keeps track of pre-processing parameters that are updated from metadata values form the spectra"""

    if not spectra_metadata:  # use default preprocessing params
        preprocs_params = {
            "trimming": {
                "min_value": 400,
                "max_value": 4000,
                "value": [500, 3500],
            }
        }
    else:
        min_index, max_index = spectra_metadata["energy_limits"]
        preprocs_params = {
            "trimming": {
                "min_value": int(min_index),
                "max_value": int(max_index),
                "value": [
                    int(0.25 * (max_index - min_index) + min_index),
                    int(0.75 * (max_index - min_index) + min_index),
                ],
            }
        }

    return preprocs_params


def process_spectrum(
    spectrum: Spectrum,
    trim_range: tuple[int],
    donwsampling_factor: int,
    outliers_threshold: float,
    baseline_p: tuple[int],
    baseline_lambda: tuple[int],
):
    """Takes preprocessing parameters from the widgets to apply all pre-processing functions"""

    processed_spectrum = spectrum.trimming(trim_range)
    processed_spectrum = processed_spectrum.downsample(donwsampling_factor)
    processed_spectrum = processed_spectrum.reject_outliers(outliers_threshold)
    processed_spectrum = processed_spectrum.asymmetric_least_squares(
        log_p=baseline_p, log_lambda=baseline_lambda
    )

    return processed_spectrum


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode("utf-8")


############################ MAIN APP LOOP  ##################################


############## BOILERPLATE

st.sidebar.image(Image.open("./gui2/assets/logo.png"), width=150)
st.header("Preprocessing")


############## RAW SPECTRUM
raw_spectrum_container = st.container()


########## FILE UPLOAD

with st.sidebar.expander("Upload files"):

    parser = st.radio(
        "Format of spectra",
        options=["Single .csv", "Single .txt (Bruker)", "Multiple .txt"],
        help="See documentation: what formats are accepted",
    )

    multiple_files = True if parser == "Multiple .txt" else False

    uploaded_files = st.file_uploader(
        "Choose CSV files", accept_multiple_files=multiple_files
    )


if uploaded_files:

    spectra, spectra_names, spectra_metadata = payload_to_spectra(
        uploaded_files, parser=parser
    )

    total_number_spectra = spectra_metadata["number_of_spectra"]

    with st.sidebar.expander("Log"):
        st.markdown(f"{total_number_spectra} files successfuly uploaded.")
        st.markdown("Metadata:")
        st.json(spectra_metadata)

    selected_spectrum_name: str = st.sidebar.selectbox(
        "Select spectrum", spectra_names
    )

    current_spectrum = spectra[selected_spectrum_name]

    preprocs_params = update_preprocessing_parameters(spectra_metadata)

else:
    total_number_spectra = 0
    spectra = None
    spectra_names = None
    current_spectrum = None
    spectra_metadata = None


########## PROCESSING PARAMETERS


preprocs_params = update_preprocessing_parameters(spectra_metadata)


tab_trim, tab_downsample, tab_outliers, tab_baseline = st.tabs(
    ["Trimming", "Downsampling", "Outliers", " Baseline correction"]
)

with tab_trim:
    st.markdown("Keeps only a subset of a spectrum for further processing")
    trimm_range: tuple[int] = st.slider(
        "Trimming range",
        min_value=preprocs_params["trimming"]["min_value"],
        max_value=preprocs_params["trimming"]["max_value"],
        value=preprocs_params["trimming"]["value"],
    )

with tab_downsample:
    st.markdown(
        "Reduce datapoints from the original signal. Ex. 2 means singal reduced by half."
    )
    donwsampling_factor: int = st.slider("Downsampling factor", 1, 10, 1)

with tab_outliers:
    st.markdown(
        "Removes points outside X * IQR of the spectral noise. X is the outlier removal threshold."
    )
    outliers_threshold: float = st.slider(
        "Outlier removal threshold",
        min_value=0.0,
        max_value=3.0,
        value=0.0,
        step=0.5,
    )

with tab_baseline:
    st.markdown("Applies Eiler's ALS algorithm to fit a baseline to spectra")
    baseline_p: float = st.slider(
        "log10 (P-parameter)",
        min_value=-6.5,
        max_value=-0.5,
        value=-1.5,
        step=0.5,
    )
    baseline_lambda: float = st.slider(
        "log10 (Lambda-parameter)",
        min_value=-7.0,
        max_value=14.0,
        value=7.0,
        step=0.5,
    )


########## PROCESS SPECTRA AND PLOT
processed_spectrum_container = st.container()
fig_raw = go.Figure()
fig_processed = go.Figure()


if current_spectrum:

    processed_spectrum = process_spectrum(
        spectrum=current_spectrum,
        trim_range=trimm_range,
        donwsampling_factor=donwsampling_factor,
        outliers_threshold=outliers_threshold,
        baseline_p=baseline_p,
        baseline_lambda=baseline_lambda,
    )

    # Plots

    fig_raw.add_trace(
        go.Scatter(
            x=current_spectrum.indexes,
            y=current_spectrum.counts,
            name="Raw spectrum",
            mode="markers",
            marker={"color": "#455A64"},
        )
    )

    fig_raw.add_trace(
        go.Scatter(
            x=processed_spectrum.indexes,
            y=processed_spectrum.baseline,
            name="Baseline",
            mode="lines",
            marker={"color": "#FF1744"},
        )
    )

    fig_raw.add_vrect(
        x0=min(current_spectrum.indexes),
        x1=trimm_range[0],
        fillcolor="white",
        opacity=0.7,
        line_width=10,
        line_color="white",
    )

    fig_raw.add_vrect(
        x0=trimm_range[1],
        x1=max(current_spectrum.indexes),
        fillcolor="white",
        opacity=0.7,
        line_width=10,
        line_color="white",
    )

    fig_raw.update_xaxes(title_text="Index")
    fig_raw.update_yaxes(title_text="Counts")
    fig_raw.update_layout(
        template="simple_white",
        legend=dict(
            orientation="h",
            entrywidth=70,
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    fig_processed.add_trace(
        go.Scatter(
            x=processed_spectrum.indexes,
            y=processed_spectrum.counts,
            name="Processed spectrum",
            mode="markers",
            marker={"color": "#455A64"},
        )
    )

    fig_processed.update_xaxes(title_text="Index")
    fig_processed.update_yaxes(title_text="Counts")
    fig_processed.update_layout(
        title="Processed spectrum",
        title_x=0.05,
        title_y=0.82,
        template="simple_white",
    )


else:
    fig_raw.add_trace(
        go.Scatter(x=[0], y=[0], mode="markers", marker={"color": "#455A64"})
    )
    fig_raw.update_layout(
        title="Raw spectrum",
        title_x=0.05,
        title_y=0.82,
        template="simple_white",
    )

    fig_processed.add_trace(
        go.Scatter(x=[0], y=[0], mode="markers", marker={"color": "#455A64"})
    )
    fig_processed.update_layout(
        title="Processed spectrum",
        title_x=0.05,
        title_y=0.82,
        template="simple_white",
    )


raw_spectrum_container.plotly_chart(
    fig_raw, use_container_width=True, config={"displaylogo": False}
)
processed_spectrum_container.plotly_chart(
    fig_processed, use_container_width=True, config={"displaylogo": False}
)


####################### BATCH PROCESSING AND DOWNLOAD ######################

st.markdown("### Batch processing")


disable_download = True
disable_batchrun = True if not spectra else False

batchcol1, batchcol2 = st.columns(2)

run_batch_processing = batchcol1.button(
    "Run batch processing", disabled=disable_batchrun
)
batch_progress_container = st.container()
batch_log_container = st.container()


if run_batch_processing:

    progress_bar = batch_progress_container.progress(
        0.0, text="Processing spectra"
    )

    total_number_spectra = len(spectra.keys())
    current_spectrum_number = 0
    processed_spectra_dict = {}

    for spectrum_label, spectrum in spectra.items():

        processed_spectrum = process_spectrum(
            spectrum=spectrum,
            trim_range=trimm_range,
            donwsampling_factor=donwsampling_factor,
            outliers_threshold=outliers_threshold,
            baseline_p=baseline_p,
            baseline_lambda=baseline_lambda,
        )

        if current_spectrum_number == 0:
            processed_spectra_dict["index"] = processed_spectrum.indexes

        processed_spectra_dict[spectrum_label] = processed_spectrum.counts

        current_spectrum_number = current_spectrum_number + 1
        progress_bar.progress(
            int(100 * current_spectrum_number / total_number_spectra),
            text=f"Processing spectra: {spectrum_label}",
        )

    processing_df = pd.DataFrame(
        processed_spectra_dict
    )  # replace by processed dataframe
    disable_download = False

else:
    processing_df = pd.DataFrame()
    disable_download = True


csv = convert_df(processing_df)

donwload_processed_data = batchcol2.download_button(
    label="Download results",
    data=csv,
    file_name="Processed_spectra.csv",
    mime="text/csv",
    disabled=disable_download,
)
