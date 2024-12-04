# © Copyright 2021, PRISMA’s Authors

import ipywidgets as wdg
import pandas as pd
import functools 

from .views.baselines import ViewBaselineAls
from .views.fitpeaks import ViewFitPeaks
from .views.load import ViewLoad
from .views.plots import PlotBaseline, PlotPeaks

import prisma.preprocessing
import prisma.baselines
import prisma.util.parsers
import prisma.fitpeaks



AVAILABLE_PARSERS = {'Single .csv':          {'multiple files': False,'file format':'.csv'},
                    'Single .txt (Bruker)': {'multiple files': False,'file format':'.txt'},
                    'Multiple .txt':        {'multiple files': True, 'file format':'*'}}



class BaselinePeakFitting:

    def __init__(self):
        self.interface = None
        self.spectra = {}
        self.spectra_metadata = {}

        self.__load_subapps()
        self.__assemble_interface()        
        self.control_events()


    # ---------------------- GUI APPEARANCE ---------------------

    def __load_subapps(self):
        self.subapps = {'Load':ViewLoad(AVAILABLE_PARSERS),
                        'Baseline':ViewBaselineAls(),
                        'FitPeaks':ViewFitPeaks(),
                        'Plot Baseline': PlotBaseline(x_label='Index',y_label='Counts [a.u.]',title='Original'),
                        'Plot Peaks': PlotPeaks(x_label='Index',y_label='Counts [a.u.]',title='Processed')}



    def __assemble_interface(self):
        plot_box = wdg.HBox([self.subapps['Plot Baseline'].interface, 
                            self.subapps['Plot Peaks'].interface])

        load_baseline_box = wdg.HBox([self.subapps['Load'].interface, 
                            self.subapps['Baseline'].interface])

        self.interface = wdg.VBox([load_baseline_box, plot_box, self.subapps['FitPeaks'].interface])

    

    # -----------------   AUXILIARY FUNCTIONS  ---------------

    def aux_get_payload_from_file_upload(self, upload):
        parser_name = self.subapps['Load'].current_parser_name

        if AVAILABLE_PARSERS[parser_name]['multiple files']: #if upload consist of multiple files
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



    def aux_refresh_plots(self,label, also_peaks = False):
        self.subapps['Plot Baseline'].update_marks(self.spectra[label]['root'],
                                                   self.spectra[label]['processed'])

        #if plot peaks is requested AND the fitting was successful
        if (also_peaks) and (self.spectra[label]['fitted'].metadata['Fitting success']):
            try:
                self.subapps['Plot Peaks'].update_marks(spectrum_original = self.spectra[label]['processed'],
                                                        spectrum_fit = self.spectra[label]['fitted'])
            except KeyError:
                self.subapps['Plot Peaks'].update_marks(spectrum_original =self.spectra[label]['processed'], 
                                                        spectrum_fit = None)
        else:
            self.subapps['Plot Peaks'].update_marks(spectrum_original =self.spectra[label]['processed'], 
                                                    spectrum_fit = None)
 

    def aux_process_spectrum(self,label):
        baseline_parameters = self.subapps['Baseline'].inputs

        trimmed_spectrum = prisma.preprocessing.trimming(self.spectra[label]['root'], 
                                                    within=baseline_parameters['Trim'])
        self.spectra[label]['processed'] = prisma.baselines.asymmetric_least_squares(trimmed_spectrum, 
                                                                                log_p=baseline_parameters['Log p'], 
                                                                                log_lambda=baseline_parameters['Log lambda'])


    def aux_fit_spectrum(self,label):
        peakfit_parameters = self.subapps['FitPeaks'].inputs

        self.spectra[label]['fitted'] = prisma.fitpeaks.fit_peaks(self.spectra[label]['processed'], 
                                                            peak_bounds = peakfit_parameters['Bounds'], 
                                                            guess_widths = peakfit_parameters['Widths'],
                                                            lineshape = peakfit_parameters['Lineshape'])

        self.subapps['FitPeaks'].render_parameter_table(self.spectra[label]['fitted'])



    def aux_update_download_payload(self, label):
        individual_dataframes = [pd.DataFrame(self.spectra[label]['root'].counts, columns=['Original'], index=self.spectra[label]['root'].indexes),
                                pd.DataFrame(self.spectra[label]['processed'].counts, columns=['Processed'], index=self.spectra[label]['processed'].indexes)]

        try:
            individual_dataframes.append(pd.DataFrame(self.spectra[label]['fitted'].counts, columns=['Peak sum'], index=self.spectra[label]['fitted'].indexes))
            individual_dataframes += [pd.DataFrame(array, columns=[peak], index=self.spectra[label]['fitted'].indexes) for peak, array in self.spectra[label]['fitted'].profiles.items()]
        except KeyError:
            pass 
        
        spectrum_dataframe = functools.reduce(lambda  left,right: pd.merge(left,right, left_index=True, right_index=True, how='outer'), individual_dataframes)
        spectrum_csv = spectrum_dataframe.to_csv(sep=',')

        self.subapps['Load'].update_download_button(filename=label,string=spectrum_csv)



   # ------------------ CALLBACKS -------------------------

    def callback_upload_spectra(self, button_upload):

        self.subapps['Load'].display_loading_status(status_msg = 'Loading Files...')

        payload, parser_name = self.aux_get_payload_from_file_upload(button_upload['new'])   

        self.spectra, self.spectra_metadata = self.aux_run_available_parser(payload, parser_name)

        if not self.spectra:
            self.subapps['Load'].display_loading_error(error_msg = self.spectra_metadata['error'])

        else:
            list_of_spectra = list(self.spectra.keys())
            list_of_spectra.sort()

            self.subapps['Load'].display_spectra_names(list_of_spectra)
            self.subapps['Baseline'].set_trim_limits(trim_limits = self.spectra_metadata['energy_limits'], 
                                                    min_resolvable_width = self.spectra_metadata['min_resolvable_width'])
            self.subapps['FitPeaks'].set_bound_limits(bound_limits = self.spectra_metadata['energy_limits'],
                                                    min_resolvable_width = self.spectra_metadata['min_resolvable_width'])
            self.subapps['FitPeaks'].set_width_limits(bound_limits = self.spectra_metadata['energy_limits'],
                                                    min_resolvable_width = self.spectra_metadata['min_resolvable_width'])

            self.subapps['Load'].display_loading_status(status_msg = 'Uploaded {} spectra'.format(len(self.spectra)))


    
    def callback_select_spectrum(self,_):
        current_spectrum_label = self.subapps['Load'].current_spectrum
 
        if current_spectrum_label != None:
            self.aux_process_spectrum(current_spectrum_label)
            self.aux_refresh_plots(current_spectrum_label, also_peaks = False)
            self.aux_update_download_payload(current_spectrum_label)



    def callback_update_peakfit(self,_):
        current_spectrum_label = self.subapps['Load'].current_spectrum
 
        if current_spectrum_label != None:
            self.aux_fit_spectrum(current_spectrum_label)
            self.aux_refresh_plots(current_spectrum_label, also_peaks = True)
            self.aux_update_download_payload(current_spectrum_label)    



    # --------------------- EVENTS ------------------------------                   

    def control_events(self):
        self.subapps['Load'].widget['Fld_spectra'].observe(self.callback_upload_spectra, names='value')
        self.subapps['Load'].widget['Sct_spectra'].observe(self.callback_select_spectrum, names='value')
        self.subapps['Baseline'].widget['Range_trim'].observe(self.callback_select_spectrum, names='value')
        self.subapps['Baseline'].widget['Slider_baseline_p'].observe(self.callback_select_spectrum, names='value')
        self.subapps['Baseline'].widget['Slider_baseline_l'].observe(self.callback_select_spectrum, names='value')
        self.subapps['FitPeaks'].widget['Button_peakfit'].on_click(self.callback_update_peakfit)



    # --------------------- API ------------------------------

    @property
    def inputs(self):

        user_inputs = {'Files': self.subapps['Load'].inputs,
                       'Baseline': self.subapps['Baseline'].inputs,
                       'FitPeaks': self.subapps['FitPeaks'].inputs}

        return user_inputs



    def batch_processing(self):        
        baseline_parameters = self.subapps['Baseline'].inputs
        peakfit_parameters = self.subapps['FitPeaks'].inputs
        unsuccessful_fits = []

        if (not peakfit_parameters['Bounds']) or (not self.spectra):
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

                self.spectra[label]['fitted'] = prisma.fitpeaks.fit_peaks(self.spectra[label]['processed'], 
                                                                    peak_bounds = peakfit_parameters['Bounds'], 
                                                                    guess_widths = peakfit_parameters['Widths'],
                                                                    lineshape = peakfit_parameters['Lineshape'])

                if not self.spectra[label]['fitted'].metadata['Fitting success']:
                    unsuccessful_fits.append(label)

        return unsuccessful_fits


    def export_batch_processing_payloads(self):
        label_first_spectrum = list(self.spectra.keys())[0] #all spectra share same energy vector. Enough to take the first one to not iterate over all
        csv_payload = {}

        csv_payload['Processed_spectra']  = pd.DataFrame(data = {label: value['processed'].counts for label, value in self.spectra.items()}, 
                                                        columns = list(self.spectra.keys()), 
                                                        index = self.spectra[label_first_spectrum]['processed'].indexes).to_csv(sep=',')

            
        csv_payload['Peak_parameters'] = pd.DataFrame(data = {label: list(spectrum['fitted'].metadata['Fitted parameters'].values())  for label, spectrum in self.spectra.items()},
                                                      columns = list(self.spectra.keys()),
                                                      index = list(self.spectra[label_first_spectrum]['fitted'].metadata['Fitted parameters'].keys())).transpose().to_csv(sep=',')

        metadata_dict = self.subapps['Load'].inputs
        metadata_dict.update(self.subapps['Baseline'].inputs)
        metadata_dict.update(self.subapps['FitPeaks'].inputs)
        csv_payload['Metadata'] = pd.DataFrame({'Parameter':list(metadata_dict.keys()),'Value':list(metadata_dict.values())}).to_csv(sep=',')

        return csv_payload