# Glossary

Definitions of domain-specific terms and abbreviations.

> Add new entries to this glossary whenever introducing domain-specific terminology, abbreviations, or project-specific conventions.

## General

- **CLI**: Command Line Interface
- **Configs**: Configurations
- **Repo**: Git repository
- **E2E**: End-to-End
- **Logging**: The process of recording events, messages, and errors that occur during a program’s execution.

## Hospitality Domain

- **ADR**: Average Daily Rate -> commonly used metric in the hospitality industry. [Click here](https://www.investopedia.com/terms/a/average-daily-rate.asp) to read an article from Investopedia for more information.
- **Lead Time**: Number of days between booking date and arrival date.
- **TA**: Travel agents.
- **TO**: Tour operators.

### hotel_bookings Dataset - Specific terms
- **Room upgrade**: Assigned room type differs from the reserved room type (in the context of this repo).
- **Arrival Date (Year/Month/Week/Day)**: Components of the date the guest is scheduled to arrive.
- **Stays in Weekend Nights**: Number of weekend nights (Sat/Sun) included in the stay.
- **Stays in Week Nights**: Number of week nights (Mon–Fri) included in the stay.
- **Customer Type (Contract)**: Booking linked to a corporate or blockage contract.
- **Customer Type (Group)**: Booking made for a group of guests together.
- **Customer Type (Transient)**: A booking not part of a group or contract, and not associated with other transient bookings.
- **Customer Type (Transient‑party)**: A transient booking that is associated with at least one other transient booking (e.g., linked reservations for related guests).
- **Reservation Status**: Final status of the booking, typically one of:
    - **Canceled** – booking cancelled by the customer
    - **Check‑Out** – guest checked in and completed their stay
    - **No‑Show** – guest did not check in and did not notify the hotel beforehand
- **Reservation Status Date**: Date when the reservation status was last set (e.g., cancellation date or check‑out date).
- **Is Canceled**: Binary indicator (1/0) for whether the booking was canceled.
- **Is Repeated Guest**: Whether the guest has stayed at the hotel previously (1 = yes, 0 = no).
- **Previous Cancellations**: Number of prior cancellations by the same guest.
- **Previous Bookings Not Canceled**: Number of prior bookings not canceled for the same guest.
- **Booking Changes**: Number of amendments made to the booking after it was created.
- **Deposit Type**: Category of deposit:
    - **No Deposit** – no deposit made
    - **Refundable** – partial refund possible
    - **Non Refund** – deposit equals or exceeds total stay cost
- **Days in Waiting List**: Number of days the booking sat on a waiting list before confirmation.
- **Total of Special Requests**: Count of special requests made by the customer (e.g., high floor, twin beds).
- **Agent / Company**: IDs representing booking travel agents or corporate entities (usually anonymized in this dataset).

## Machine Learning

- **Model**: Multiple meanings:
    1. A combination of problem type + segment + *optionally* model version -> these three define folder nesting for configs that define all model-specific parts of the workflow.
    2. A trained model - e.g. a trained .cbm (CatBoost model) that can be used to make predictions.
    3. A Pydantic model (used for validation)
- **Pipeline**: Multiple meanings:
    1. sklearn Pipeline (wrapper around a trained model that does some steps with incoming data - e.g. preprocessing and feature engineering - prior to that data reaching the model).
    2. A python script that executes a part of the ml workflow, such as building an interim dataset, freezing a feature set, or evaluating a model.
    3. Code that is assembled together and executed in named, logical steps to make it more readable.
    4. An entire ml workflow covered by repo's code (rarely used in this context within this repo).
- **Snapshot**: Multiple meanings:
    1. An immutable set of artifacts (e.g. a feature set snapshot includes a frozen feature set and metadata that are never meant to be changed)
    2. A moment in time during model training (e.g. catboost training snapshot) - used to continue training from the last saved "checkpoint" (iteration) in case of an unexpected failure (like a power shutdown).
- **Run**: An execution of a pipeline (python script). Usually each run generates a new snapshot, but search and training runs are allowed to continue in case they fail mid-way.
- **Interim Dataset**: A dataset that is the same as the raw data, but has been optimized to be more memory-efficient.
- **Processed Dataset**: A dataset that is ready for production (feature freezing, creating the target variable). Within this repo, that means row_id is present and the leaky columns are dropped (unless there is a specific reason to keep them).
- **Feature Set**: A logically connected group of features that may come from one or more datasets.
- **Feature Freezing**: A process of grouping features into a feature set based on given snapshot(s) of one or more datasets, then performing some operations on them, and then storing them. Frozen feature sets are not meant to ever be altered.
- **Feature Engineering**: A process of creating new features, usually (and within this repo exclusively) based on the existing ones.
- **Feature Registry**: In the context of this repo, configs that define all of the available feature sets - meaning the information used to freeze them.
- **Search**: Hyperparameter search (terms used interchangeably throughout this repo).
- **Training**: Code that trains a given model with using predefined configs and best parameters from search.
- **Evaluation**: Code that evaluates model's performance on some metrics (e.g. roc_auc and f1 for classification, or mae and rmse for regression, among others).
- **Explainability**: Code that generates artifacts that can help a human understand how/why the model is making some predictions (e.g. feature importances, SHAP importances).
- **Experiment**: A logical grouping of artifacts generated by the modeling steps (search, training, evaluation, explainability); defined by search within this repo (one search = one experiment); includes trained models and pipelines that can be used if a model gets promoted.
- **Artifact**: Stored output of a pipeline run (model, metrics, datasets).
- **Modeling**: The process that includes hyperparameter searching, training, evaluation, and explainability.
- **Metadata**: Information describing a specific run.
- **Runtime info**: Information that can include git commit, python version, conda environment information, OS and hardware-related information connected to a specific run. Sometimes it may also include the duration of a given run.
- **Reproducibility**: The ability to execute a pipeline again and get the same output, primarily in terms of the produced artifacts.
- **Model Promotion**: Process of promoting a trained model to a higher environment (e.g., staging or production).
- **Model Registry**: Multiple meanings within this repo:
    1. A file containg the information regarding all of the models that are currently in staging or production
    2. A folder that contains the model registry, as well as the model archive, and all of the promotion-run-specific information (metadata and logs)
- **Model Archive**: A file containing the information regarding all of the models that used to be in production.
- **Orchestration**: The process of running multiple pipelines in sequence.
- **Notebook**: A jupyter notebook file - may contain markdown and python code.
- **Persistence**: The process of saving relevant information/artifacts.
- **Components**: Within this repo, usually refers to the pre-model steps within an sklearn Pipeline.
- **Imputation**: Within this repo, a process of replacing missing (null or N/A) rows with some values (e.g. the string "missing").
- **Target**: A variable the value of which we are trying to predict with a given model.