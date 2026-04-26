## Testing the EBSL using the Sybil DDoS Dataset

This repository contains the code and pre-trained models to reproduce results in the paper titled: "Trust-based attack detection model for connected cars using a Subjective Logic based framework". The core implementation is in the [ebsl-ids](https://github.com/a-h-ismail/ebsl-ids) repository.

### Running Locally

1. Create a Python 3.12+ environment.
1. Install required packages listed in `requirements.txt`.
1. Copy the datasets (will be published [here](https://github.com/A-Wehby/Connected-Cars-DDoS-Dataset)) to the `datasets` directory.
1. Run the `test_ebsl_with_scenario.py` file and check the results. Redirecting output to a file is recommended.
1. Repeat step 4 with the `test_other_dynamic_ensembles.py` file to obtain results from other dynamic ensembles.

Existing results from a local run of the EBSL are in `results.txt` file, while other dynamic ensemble results are in the `others.txt` file.

Note: All dynamic ensemble classifiers used here are "fitted" using the same validation dataset.
