EEG Data Preprocessing Pipeline

**Data Loading and Channel Setup:**
The preprocessing pipeline begins by loading the raw EEG data. Rather than relying on standard .vhdr files, the data was loaded directly from Emotiv’s self-contained format. The initial raw recording contained 127 channels, which included non-EEG auxiliary streams like head positioning and movement tracking. The data was filtered down to the 32 core EEG channels. To ensure compatibility with MNE’s standard 10-20 electrode montage, Emotiv’s default naming for the central frontal electrode was corrected from "Afz" to "AFz".

**Signal Filtering:**
Emotiv hardware typically applies an automatic notch filter to handle regional power grid interference (e.g., 50Hz or 60Hz). To further clean the signal and isolate the cognitive activity, a bandpass filter of 0.1 Hz to 40 Hz was applied. This effectively suppresses remaining high-frequency noise and focuses the data on the 1–30 Hz frequency range, which is the most relevant for observing standard cognitive brainwave activity.

**Artifact Removal via ICA:**
Independent Component Analysis (ICA) was utilized to identify and remove ocular artifacts, such as eye blinks and horizontal eye movements. Because ICA performs optimally when low-frequency drift is minimized, a separate copy of the data was temporarily created and passed through a strict 1.0 Hz high-pass filter. An automatic rejection threshold was set to ignore extreme, non-biological voltage spikes, and the ICA model was fitted to this high-passed data copy.

**Event Mapping and Epoching:**
To connect the recorded brainwaves to the specific visual stimuli, the hardware triggers embedded in the EEG recording were cross-referenced with the experiment's metadata file. A filtered mapping dictionary was generated to ensure that only the event IDs actually present in the current recording block were retained. Using this mapped dictionary, the continuous EEG data was sliced into 1-second epochs locked to the onset of each image.

**Final Cleaning and Export:**
The previously fitted ICA model was applied directly to these final epochs, effectively scrubbing the ocular noise from the trial data. In the final step, the cleaned MNE Epochs object was split into standard machine learning variables: an X array containing the 3D brainwave features and a y array containing the corresponding image labels. These were exported as NumPy files, fully preparing the data for the machine learning training phase.
