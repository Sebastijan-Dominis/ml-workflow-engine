_Last updated: 2026-03-24_
# Roadmap

- This document outlines the planned development roadmap for the project.
- Time ranges are relative to the last update of this document.

## Structure
- Each planned feature is nested within short term, mid term, or long term segment, which signals when it can be expected
- Each planned feature contains two descriptive elements:
    - status (planned/in progress/done)
    - importance (low/medium/high/crucial)

## Short Term (less than 1 month)
- [x] Add a document describing each of the configs
    - Status: done (2026-03-13)
    - Importance: high
- [x] Create an ml service that wraps pipelines and allows config writing using Dash and FastAPI
    - Status: done (2026-03-19)
    - Importance: medium
- [x] Add pipeline config validation to the training pipeline
    - Status: done (2026-03-13)
    - Importance: medium
- [x] Create dataset + feature set snapshot binding logic and configs
    - Status: done (2026-03-23)
    - Importance: medium
- [x] Improve dataset and feature set merging logic
    - Status: done (2026-03-24)
    - Importance: medium
- [ ] Create infer.py
    - Status: planned
    - Importance: medium
- [ ] Create monitor.py
    - Status: planned
    - Importance: medium
- [ ] Create tools for an LLM that leverage model_registry + explainability artifacts to provide users with useful feedback
    - Status: planned
    - Importance: low
- [ ] Create a Dash + FastAPI app that implements that LLM, where users can interact with it
    - Status: planned
    - Importance: low

## Mid Term (1-6 months)
- [ ] Add time-series logic, starting with Prophet algorithm
    - Status: planned
    - Importance: low
- [ ] Add support for xgboost and lgbm algorithms
    - Status: planned
    - Importance: low

## Long Term (6+ months)
_No items currently planned._