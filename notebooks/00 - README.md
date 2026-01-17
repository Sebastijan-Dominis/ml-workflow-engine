# Notebooks

## Purpose:

- This folder only exists to provide full transparency on how the EDA, data preparation, and hyperparameter searches were initially done. The actual pipeline for everything, including training, evaluation, and explainability, exists in separate folders. The pipelines and explainability data saved via the notebooks found within this folder only existed for experimental purposes. 

## Structure:

- The file names include numbers at the beginning, which helps identify the chronological order in which they were created. That, in turn, helps understand some of the decisions done in this stage of the project.

### Note:

> Some of the notebooks were accessed on multiple occasions during the experimental phase of the project, which means that the chronological order is not exact/perfect. This is deemed acceptable for this experimental folder, but noteworthy nonetheless.

## Feature Savings

- The feature savings notebook is the main source of truth for the creation of the parquet files which are then used for the official training. The data splits use the same random seed used in the relevant notebooks found within this folder.
- The feature savings notebook may be updated with new versions of features, if and when appropriate.

**IMPORTANT**
- In updating the feature savings notebook, the old feature creation should not deleted. Instead, the existing structure should be complied with when creating the new cells for the savings of new features. The model for which the splits are done should be marked with "##" in a markdown cell, while the feature version should be marked with "###" under the relevant part of the notebook. This helps follow the versions chronologically, and clearly separates the feature creation by model. All of that makes the project more scalable, readable, and maintainable.

## New Models

- If creating new models, a separate folder should be created for each of them, and follow the same naming convention (chronological number + name of the model). Those folders should also follow the existing naming convention within them (chronological number + name of the algorithm/method tried within the notebook).
- New files should not save anything outside of the notebooks folder. This was previously done, but only in the initial stages of the project, since no harm could be done at that stage. If anything needs to be saved in the future (model/pipeline/etc.), save it within this folder. The obvious exception is the Feature Saving notebook.