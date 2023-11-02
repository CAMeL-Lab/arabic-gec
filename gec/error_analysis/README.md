# Error Analysis:

Specific error type performance of AraBART and our best system (AraBART+Morph+GED<sup>13</sup>) on average on the dev sets of QALB-2014, QALB-2015, and ZAEBUC.
We do so by first aligning the erroneous input sentences with the models' outputs and passing the alignments to ARETA to obtain the specific error types.
We then project the error types on the tokens and evaluate those error tags against the gold error tags. Running the [error_analysis.sh](error_analysis.sh) script does all of these steps automatically.

