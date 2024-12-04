# © Copyright 2021, PRISMA’s Authors

import ipywidgets as wdg
import pandas as pd
import functools 

from .views.baselines import ViewBaselineAls
from .views.load import ViewLoad
from .views.plots import PlotBaseline, PlotIntensities


import prisma.preprocessing
import prisma.baselines
import prisma.util.parsers


# TKINTER OPTION IMPORTS
# from tkinter.filedialog import askopenfilename #GUI box to open files
# import tkinter


AVAILABLE_PARSERS = {'Single .csv':          {'multiple files': False,'file format':'.csv'},
                    'Single .txt (Bruker)': {'multiple files': False,'file format':'.txt'},
                    'Multiple .txt':        {'multiple files': True, 'file format':'*'}}


class BaselineCorrection:

    def __init__(self):
        self.interface = None
        self.spectra = {}
        self.spectra_metadata = {}

        self.view_load_subapps()
        self.view_assemble_interface()
        self.control_events()



    def view_load_subapps(self):
        self.subapps = {'Load':ViewLoad(AVAILABLE_PARSERS),
                        'Baseline':ViewBaselineAls(),
                        'Plot Baseline': PlotBaseline(x_label='Raman shift [cm-1]',y_label='Counts [a.u.]',title='Original'),
                        'Plot Intensities':PlotIntensities(x_label='Raman shift [cm-1]',y_label='Counts [a.u.]',title='Baseline corrected')}



    def view_assemble_interface(self):
        plot_box = wdg.HBox([self.subapps['Plot Baseline'].interface, 
                            self.subapps['Plot Intensities'].interface])

        load_baseline_box = wdg.HBox([self.subapps['Load'].interface, 
                            self.subapps['Baseline'].interface])

        self.interface = wdg.VBox([load_baseline_box, plot_box])
  

    
    @property
    def inputs(self):
        user_inputs = {}

        user_inputs['Files'] = self.subapps['Load'].inputs
        user_inputs['Baseline'] = self.subapps['Baseline'].inputs

        return user_inputs



    # ------------------- Auxiliary functions -------------

    #TKINTER OPTION: ANONYMOUS ARGUMENT + FILENAME WINDOW + OPEN EVERY FILE IN FILENAME + PARSE INTO NAME:BINARY STRING DICTIONARY
    # root = tkinter.Tk()
    # files_full_paths = askopenfilename(title = "Select spectra", filetypes = (("text files","*.txt"),("all files","*.*")), multiple = True) #window to select files
    # root.destroy()

    def aux_get_payload_from_file_upload(self, upload):
        parser_name = self.subapps['Load'].current_parser_name

        if AVAILABLE_PARSERS[parser_name]['multiple files']:
            payload =  {key:value['content'] for key, value in upload.items()}
        else: 
            filename = list(upload.keys())[0]
            payload = upload[filename]['content'] 

        return payload, parser_name



    def aux_run_available_parser(self, payload, parser):
        if parser == 'Single .csv':
            return prisma.util.parsers.single_csv(payload)
        elif parser == 'Single .txt (Bruker)':
            return prisma.util.parsers.single_txt_bruker(payload)
        elif parser == 'Multiple .txt':
            return prisma.util.parsers.multiple_txt(payload)
        else:
            raise KeyError('The parser is not defined')


    def aux_process_spectrum(self,label):
        baseline_parameters = self.subapps['Baseline'].inputs

        trimmed_spectrum = prisma.preprocessing.trimming(self.spectra[label]['root'], 
                                                    within=baseline_parameters['Trim'])
        self.spectra[label]['processed'] = prisma.baselines.asymmetric_least_squares(trimmed_spectrum, 
                                                                                log_p=baseline_parameters['Log p'], 
                                                                                log_lambda=baseline_parameters['Log lambda'])


    
    def aux_refresh_plots(self,label):
        self.subapps['Plot Baseline'].update_marks(self.spectra[label]['root'],
                                                   self.spectra[label]['processed'])
        self.subapps['Plot Intensities'].update_marks(self.spectra[label]['processed'])



    def aux_update_download_payload(self, label):
        individual_dataframes = [pd.DataFrame(self.spectra[label]['root'].counts, columns=['Original'], index=self.spectra[label]['root'].indexes),
                                 pd.DataFrame(self.spectra[label]['processed'].counts, columns=['Processed'], index=self.spectra[label]['processed'].indexes)]
        
        spectrum_dataframe = functools.reduce(lambda  left,right: pd.merge(left,right, left_index=True, right_index=True, how='outer'), individual_dataframes)
        spectrum_csv = spectrum_dataframe.to_csv(sep=',')

        self.subapps['Load'].update_download_button(filename=label,string=spectrum_csv)



    # ------------------ CALLBACKS -------------------------
    def callback_upload_spectra(self, button_upload):
        #TKINTER OPTION: NO ARGUMENT
        payload, parser_name = self.aux_get_payload_from_file_upload(button_upload['new'])   

        self.spectra, self.spectra_metadata = self.aux_run_available_parser(payload, parser_name)

        if not self.spectra:
            self.subapps['Load'].display_loading_error(self.spectra_metadata['error'])

        else: 
            self.subapps['Load'].display_loading_status(status_msg = 'Uploaded {} spectra'.format(len(self.spectra)))

            list_of_spectra = list(self.spectra.keys())
            list_of_spectra.sort()

            self.subapps['Load'].display_spectra_names(list_of_spectra)
            self.subapps['Baseline'].set_trim_limits(trim_limits = self.spectra_metadata['energy_limits'], 
                                                    min_resolvable_width = self.spectra_metadata['min_resolvable_width'])


    def callback_update_processing(self,_):
        current_spectrum_label = self.subapps['Load'].current_spectrum
 
        if current_spectrum_label == None:
            pass
        else:
            self.aux_process_spectrum(current_spectrum_label)
            self.aux_refresh_plots(current_spectrum_label)
            self.aux_update_download_payload(current_spectrum_label)


            
    # --------------------EVENTS-------------------

    def control_events(self):
        self.subapps['Load'].widget['Fld_spectra'].observe(self.callback_upload_spectra, names='value') # TKINTER CHANGE: ONCLICK
        self.subapps['Load'].widget['Sct_spectra'].observe(self.callback_update_processing, names='value')
        self.subapps['Baseline'].widget['Range_trim'].observe(self.callback_update_processing, names='value')
        self.subapps['Baseline'].widget['Slider_baseline_p'].observe(self.callback_update_processing, names='value')
        self.subapps['Baseline'].widget['Slider_baseline_l'].observe(self.callback_update_processing, names='value')


    # --------------------API------------------- 

    def batch_processing(self):        
        baseline_parameters = self.subapps['Baseline'].inputs
        unsuccessful_fits = []

        if not self.spectra:
            pass

        elif self.spectra_metadata['common_energy_axis'] == False:
            unsuccessful_fits = 'multiple energy axes'

        else:
            for label in self.spectra.keys():
                trimmed_spectrum = prisma.preprocessing.trimming(self.spectra[label]['root'], 
                                                            within=baseline_parameters['Trim'])
                self.spectra[label]['processed'] = prisma.baselines.asymmetric_least_squares(trimmed_spectrum, 
                                                                                        log_p=baseline_parameters['Log p'], 
                                                                                        log_lambda=baseline_parameters['Log lambda'])
        return unsuccessful_fits


    def export_batch_processing_payloads(self):
        label_first_spectrum = list(self.spectra.keys())[0] #all spectra share same energy vector. Enough to take the first one to not iterate over all
        csv_payload = {}

        csv_payload['Processed_spectra']  = pd.DataFrame(data = {label: value['processed'].counts for label, value in self.spectra.items()}, 
                                                        columns = list(self.spectra.keys()), 
                                                        index = self.spectra[label_first_spectrum]['processed'].indexes).to_csv(sep=',')

        metadata_dict = self.subapps['Load'].inputs
        metadata_dict.update(self.subapps['Baseline'].inputs)
        csv_payload['Metadata'] = pd.DataFrame({'Parameter':list(metadata_dict.keys()),'Value':list(metadata_dict.values())}).to_csv(sep=',')

        return csv_payload