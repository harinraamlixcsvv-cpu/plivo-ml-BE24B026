\# NOTES



\## Approach



This submission treats end-of-turn detection as a causal binary classification problem.



For every candidate pause, the model estimates:



\- `eot`: the speaker has finished the turn

\- `hold`: the speaker is temporarily pausing and may continue



The system uses only information available before `pause\_start`.



No future audio is used during feature extraction.



\## Features



The final feature vector contains 16 causal features covering:



\### Energy

\- Final energy

\- Mean energy

\- Energy standard deviation

\- Energy drop relative to recent context

\- Recent energy slope



\### Pitch

\- Final F0

\- Mean F0

\- F0 standard deviation

\- Recent F0 slope

\- Final F0 relative to mean F0



\### Voicing

\- Overall voiced fraction

\- Final voiced fraction

\- Length of final voiced run



\### Turn Context

\- Elapsed time at the candidate pause

\- Pause index within the turn

\- Available recent context duration



A 2-second speech window immediately preceding the candidate pause is used for local prosodic analysis.



\## Model



The final classifier is Logistic Regression with:



\- StandardScaler

\- Balanced class weights

\- C = 0.5

\- Maximum 2000 iterations



Three models are trained and saved:



\- Hindi specialist

\- English specialist

\- Global fallback



The specialist models allow language-specific prosodic patterns to be learned independently.



The global model provides a fallback when the language cannot be reliably inferred.



\## Inference



`predict.py`:



1\. Reads the provided labels metadata.

2\. Determines the appropriate specialist when language information is available from the dataset structure or identifiers.

3\. Falls back to the global model otherwise.

4\. Loads the corresponding saved model.

5\. Extracts exactly the same causal features used during training.

6\. Outputs `p\_eot` for every candidate pause.



The inference pipeline never refits the model on evaluation data.



\## Validation



Model selection used 5-fold GroupKFold out-of-fold evaluation.



Grouping was performed by `turn\_id`, ensuring that pauses belonging to the same conversation turn never appeared simultaneously in training and validation.



Primary Hindi OOF result:



\- AUC: 0.654

\- Mean response delay: 850 ms

\- Interrupted turns: 5%



Primary English OOF result:



\- AUC: 0.596

\- Mean response delay: 1301 ms

\- Interrupted turns: 5%



\## Design Decisions



Logistic Regression was selected over Histogram Gradient Boosting because it achieved a better Hindi out-of-fold result on the primary latency metric.



A Logistic/Boosting probability ensemble was also evaluated but did not improve response delay.



A pause-position probability heuristic slightly improved AUC but did not improve the primary response-delay metric and was therefore rejected.



\## Limitations



The dataset is relatively small, so estimates may have high variance across turns.



Language routing depends on available dataset metadata or identifiers. When these do not provide a reliable language signal, the system intentionally uses the global fallback model.



The current model relies on acoustic/prosodic information and does not use lexical or semantic information from speech recognition.



Future improvements could include stronger speaker-normalized prosody, learned acoustic embeddings, streaming ASR-derived semantic completion signals, and validation on a larger multilingual corpus.



\## Reproduction



Train the saved models using:



python train.py --english\_dir <english\_data\_dir> --hindi\_dir <hindi\_data\_dir> --model\_dir models



Run inference using:



python predict.py --data\_dir <evaluation\_data\_dir> --model\_dir models --out predictions.csv

