### CSE-6250 Project - Experiments with Hierarchical Ensembles for ICD-9 Coding

### /data

- ICD9CM.csv is the ICD-9 ontology from bioportal.bioontology.org
- node_desc.csv is the processed node-description pair list from ICD9CM.csv
- node_parent.csv is the processed node-parent pair list from ICD9CM.csv
- *_hadm_ids.csv are the MIMIC III splits from Jame's Mullenbach's CAML repo

To run the notebooks, you must also create a directory named
"restricted_mimic_iii" containing the following MIMIC III files:

- DIAGNOSES_ICD.csv
- NOTEEVENTS.csv
- PROCEDURES_ICD.csv

### /notebooks

- prep-icd9-hierarchy.ipynb parses the source ontology into the node-description
and node-parent lists
- prep-mimic-iii.ipynb joins the ICD-9 codes to the corresponding discharge
summaries and labels splits based on James Mullenbach's split files
- svm-tests.ipynb provides a demo of the ICD9Tree class which can be used to
store the ontology and fit and predict with a hierarchical SVM model. This class
can be extended to perform evaluation or use other models.
- svm-level-tests-MIMIC-III.ipynb compares the performance of flat and hierarchical SVMs at varying depths
- svm-level-tests-MIMIC-III-top50.ipynb compares the performance of flat and hierarchical SVMs at varying depths 

### icd9.py

Provides classes for constructing an ICD-9 graph and using it for fitting and
predicting with a hierarchical model. 
