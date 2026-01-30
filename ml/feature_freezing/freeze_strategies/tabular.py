import yaml
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)
import hashlib
import pyarrow as pa

from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split

from ml.feature_freezing.utils import generate_operator_hash
from ml.registry import FEATURE_OPERATORS

from ml.logging_config import setup_logging

class FreezeTabular:
    def load_and_hash_data(self, path: Path) -> tuple[pd.DataFrame, str]:
        with open(path, "rb") as f:
            data_hash = hashlib.md5(f.read()).hexdigest()

        data = pd.read_parquet(path)
        return data, data_hash
    
    def validate_operators(self, operators: list, operator_hash: str):
        for name in operators:
            if name not in FEATURE_OPERATORS:
                raise ValueError(f"Unknown operator: {name}")
            
        generated_hash = generate_operator_hash(operators)
        if generated_hash != operator_hash:
            msg = f"Operator hash mismatch: expected {operator_hash}, got {generated_hash}"
            logger.error(msg)
            raise ValueError(msg)

    def validate_include_exclude_columns(self, config: dict):
        include = set(config["columns"]["include"])
        exclude = set(config["columns"]["exclude"]["leaky"] + config["columns"]["exclude"]["useless"] + [config["target"]["name"]])
        intersection = include.intersection(exclude)
        if intersection:
            msg = f"Columns {intersection} are present in both include and exclude lists."
            logger.error(msg)
            raise ValueError(msg)
        
    def normalize_dtype(self, dtype) -> str:
        """
        Normalize any pandas dtype (including extension dtypes) to a string.
        """
        # Handle categorical
        if hasattr(dtype, "categories") and hasattr(dtype, "ordered"):
            return "category"

        # Handle nullable string dtype
        if str(dtype) == "string[python]" or str(dtype) == "string":
            return "object"

        # Handle nullable integers (Int64, Int32, Int16, Int8)
        if str(dtype).startswith("Int") or str(dtype).startswith("UInt"):
            return "int64"

        if np.issubdtype(dtype, np.integer):
            return "int64"
        if np.issubdtype(dtype, np.floating):
            return "float64"
        if np.issubdtype(dtype, np.bool_):
            return "bool"
        if np.issubdtype(dtype, np.object_):
            return "object"
        if np.issubdtype(dtype, np.datetime64):
            return "datetime64[ns]"
        return str(dtype)

    def validate_target(self, y: pd.Series, config: dict):
        if y.isnull().any():
            msg = "Target variable contains null values."
            logger.error(msg)
            raise ValueError(msg)
        
        actual_dtype = self.normalize_dtype(y.dtype)
        allowed = config["target"]["allowed_dtypes"]
        if actual_dtype not in allowed:
            msg = f"Target variable has dtype {y.dtype}, expected one of {allowed}."
            logger.error(msg)
            raise ValueError(msg)
        
        positive_class = config["target"].get("positive_class", None)
        if positive_class is not None and positive_class not in y.unique():
            msg = f"Positive class {positive_class} not found in target variable."
            logger.error(msg)
            raise ValueError(msg)
            
    def validate_input_no_nulls(self, X: pd.DataFrame, config: dict):
        forbidden_nulls = config["constraints"].get("forbid_nulls", [])
        if forbidden_nulls:
            for col in forbidden_nulls:
                if col in X.columns and X[col].isnull().any():
                    msg = f"Feature {col} contains null values, which is forbidden by constraints."
                    logger.error(msg)
                    raise ValueError(msg)
            
    def validate_max_cardinality(self, X: pd.DataFrame, config: dict):
        categorical_features = config["feature_roles"].get("categorical", [])
        max_cardinality = config["constraints"].get("max_cardinality", {})

        if max_cardinality:
            for col in categorical_features:
                if col in X.columns:
                    cardinality = X[col].nunique()
                    if cardinality > max_cardinality.get(col, float('inf')):
                        msg = f"Categorical feature {col} exceeds max cardinality of {max_cardinality.get(col, float('inf'))} with {cardinality} unique values."
                        logger.error(msg)
                        raise ValueError(msg)

    def validate_constraints(self, X: pd.DataFrame, config: dict):
        self.validate_input_no_nulls(X, config)
        self.validate_max_cardinality(X, config)
        
    def prepare_features(self, data: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.Series]:
        TARGET = config["target"]["name"]
        INCLUDE = config["columns"]["include"]
        
        # NOTE: `include` is the source of truth.
        # `exclude` exists only for validation and safety.
        X = data[INCLUDE].copy()
        y = data[TARGET].copy()

        return X, y

    def validate_data_types(self, X: pd.DataFrame, y: pd.Series, config: dict):
        categorical_features = config["feature_roles"].get("categorical", [])
        numerical_features = config["feature_roles"].get("numerical", [])
        datetime_features = config["feature_roles"].get("datetime", [])
        allowed_categorical_types = ["object", "category", "bool", "string"]
        allowed_numerical_types = ["int64", "float64", "int32", "float32", "int16", "int8"]
        allowed_datetime_types = ["datetime64[ns]", "datetime64[ns, UTC]"]

        for col in categorical_features:
            if col in X.columns:
                actual_dtype = self.normalize_dtype(X[col].dtype)
                if actual_dtype not in allowed_categorical_types:
                    msg = f"Categorical feature {col} has invalid dtype {X[col].dtype}"
                    logger.error(msg)
                    raise ValueError(msg)
            
        for col in numerical_features:
            if col in X.columns:
                actual_dtype = self.normalize_dtype(X[col].dtype)
                if actual_dtype not in allowed_numerical_types:
                    msg = f"Numerical feature {col} has invalid dtype {X[col].dtype}"
                    logger.error(msg)
                    raise ValueError(msg)
            
        for col in datetime_features:
            if col in X.columns:
                actual_dtype = self.normalize_dtype(X[col].dtype)
                if actual_dtype not in allowed_datetime_types:
                    msg = f"Datetime feature {col} has invalid dtype {X[col].dtype}"
                    logger.error(msg)
                    raise ValueError(msg)

    def random_split(self, X, y, test_size, random_state, stratify):
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

    def split_data(self, X: pd.DataFrame, y: pd.Series, config: dict, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        # Expandable for future split strategies
        SPLIT_REGISTRY = {
            "random": self.random_split,
        }

        split_func = SPLIT_REGISTRY[config["split"]["strategy"]]

        X1, X2, y1, y2 = split_func(
            X, y,
            test_size=test_size,
            random_state=config["split"]["random_state"],
            stratify=y if config["split"].get("stratify_by", False) else None
        )

        return X1, X2, y1, y2

    def freeze_parquet(self, path, X_train, X_val, X_test, y_train, y_val, y_test, compression=None):
        X_train.to_parquet(path / "X_train.parquet", index=False, compression=compression)
        X_val.to_parquet(path / "X_val.parquet", index=False, compression=compression)
        X_test.to_parquet(path / "X_test.parquet", index=False, compression=compression)

        y_train.to_frame().to_parquet(path / "y_train.parquet", index=False, compression=compression)
        y_val.to_frame().to_parquet(path / "y_val.parquet", index=False, compression=compression)
        y_test.to_frame().to_parquet(path / "y_test.parquet", index=False, compression=compression)
        logger.info(f"Tabular features saved to {path}")

    def freeze_features(self, context, config, X_train, X_val, X_test, y_train, y_val, y_test, now):
        path = Path(f"data/feature_store/{context.problem}/{context.segment}/{context.feature_set}/{context.version}/{now}")
        path.mkdir(parents=True, exist_ok=True)

        # Expandable for future storage formats
        FREEZE_FORMAT_REGISTRY = {
            "parquet": self.freeze_parquet,
        }

        freeze_func = FREEZE_FORMAT_REGISTRY[config["storage"]["format"]]
        freeze_func(path, X_train, X_val, X_test, y_train, y_val, y_test, config["storage"].get("compression", None))

        return path

    def save_input_schema(self, path: Path, X_train: pd.DataFrame, config: dict):
        # Stop if raw schema already exists
        schema_path = path / "input_schema.csv"
        if schema_path.exists():
            logger.info(f"Input schema already exists at {schema_path}, skipping.")
            return

        schema = pd.DataFrame({
            "feature": X_train.columns,
            "dtype": X_train.dtypes.astype(str),
            "role": "input",
        })

        schema.to_csv(schema_path, index=False)
        logger.info(f"Input schema saved to {schema_path}")
    
    def save_derived_schema(self, path: Path, X_train: pd.DataFrame, operator_names: list[str]):
        # Stop if derived schema already exists
        schema_path = path / "derived_schema.csv"
        if schema_path.exists():
            logger.info(f"Derived schema already exists at {schema_path}, skipping save.")
            return

        operators = [FEATURE_OPERATORS[name]() for name in operator_names]

        X_sample = X_train.head(100)  # small sample to detect dtypes
        derived_features = []
        for op in operators:
            X_sample = op.transform(X_sample)
            for f in op.output_features:
                derived_features.append({
                    "feature": f,
                    "dtype": str(X_sample[f].dtype),
                    "role": "derived",
                    "source_operator": op.__class__.__name__,
                    "materialized": f in X_train.columns,
                })

        derived_schema = pd.DataFrame(derived_features)
        derived_schema.to_csv(schema_path, index=False)
        logger.info(f"Derived schema saved to {schema_path}")

    def hash_config(self, config: dict) -> str:
        config_str = yaml.dump(config, sort_keys=True)
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()
    
    def hash_schema(self, X: pd.DataFrame) -> str:
        arr = pd.util.hash_pandas_object(X, index=True).to_numpy()
        return hashlib.md5(arr.tobytes()).hexdigest()

    def create_metadata(self, snapshot_path: Path, schema_path: Path, data_hash: str, schema_hash: str, operators_hash: str, config_hash: str, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> dict:
        return {
            "created_by": "freeze.py",
            "created_at": datetime.now().isoformat(),
            "feature_type": "tabular",

            "snapshot_path": str(snapshot_path),
            "schema_path": str(schema_path),

            "data_hash": data_hash,
            "operators_hash": operators_hash,
            "config_hash": config_hash,

            "row_counts": {
                "train": len(X_train),
                "val": len(X_val),
                "test": len(X_test),
            },
            "column_count": X_train.shape[1],
        }

    def freeze(self, context, config):
        setup_logging()
        
        data, data_hash = self.load_and_hash_data(Path(config["data_path"]))
        
        self.validate_operators(config["operators"], config["operator_hash"])
        self.validate_include_exclude_columns(config)

        X, y = self.prepare_features(data, config)
        self.validate_data_types(X, y, config)
        self.validate_target(y, config)
        self.validate_constraints(X, config)

        X_train_val, X_test, y_train_val, y_test = self.split_data(X, y, config, test_size=config["split"]["test_size"])

        relative_val_size = config["split"]["val_size"] / (1.0 - config["split"]["test_size"])

        X_train, X_val, y_train, y_val = self.split_data(X_train_val, y_train_val, config, test_size=relative_val_size)
        
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        snapshot_path = self.freeze_features(context, config, X_train, X_val, X_test, y_train, y_val, y_test, now)

        schema_path = Path(f"data/feature_store/{context.problem}/{context.segment}/{context.feature_set}/{context.version}")

        self.save_input_schema(schema_path, X_train, config)

        self.save_derived_schema(schema_path, X_train, config["operators"])

        config_hash = self.hash_config(config)
        schema_hash = self.hash_schema(X_train)

        metadata = self.create_metadata(snapshot_path, schema_path, data_hash, schema_hash, config["operator_hash"], config_hash, X_train, X_val, X_test)

        return snapshot_path, metadata