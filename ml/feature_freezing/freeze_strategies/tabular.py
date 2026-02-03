# TODO: Modularize this file to separate concerns and make it more readable and maintainable.
import yaml
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)
import hashlib
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split

from ml.feature_freezing.freeze_strategies.base import FreezeStrategy
from ml.feature_freezing.utils import generate_operator_hash
from ml.utils.utils import get_git_commit
from ml.registry.feature_operators import FEATURE_OPERATORS

class FreezeTabular(FreezeStrategy):
    def _safe(self, val) -> str:
        return "None" if val is None else str(val)

    def hash_parquet_metadata(self, path: Path) -> str:
        pf = pq.ParquetFile(path)
        meta = pf.metadata

        h = hashlib.sha256()

        for i in range(meta.num_columns):
            col = meta.schema.column(i)
            h.update(col.name.encode())
            h.update(self._safe(col.physical_type).encode())
            h.update(self._safe(col.logical_type).encode())

        h.update(self._safe(meta.num_rows).encode())
        h.update(self._safe(meta.created_by).encode())

        for i in range(meta.num_row_groups):
            rg = meta.row_group(i)
            for j in range(rg.num_columns):
                col = rg.column(j)
                stats = col.statistics
                if stats:
                    h.update(self._safe(stats.min).encode())
                    h.update(self._safe(stats.max).encode())
                    h.update(self._safe(stats.null_count).encode())
                    h.update(self._safe(stats.distinct_count).encode())

        return h.hexdigest()

    def hash_arrow_metadata(self, path: Path) -> str:
        with pa.memory_map(path, 'r') as source:
            reader = pa.ipc.open_file(source)
            schema = reader.schema

            h = hashlib.sha256()
            for field in schema:
                h.update(field.name.encode())
                h.update(self._safe(field.type).encode())
                h.update(self._safe(field.nullable).encode())

            h.update(self._safe(reader.num_record_batches).encode())
            return h.hexdigest()

    def hash_streaming(self, path: Path, chunk_size=1024 * 1024) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    
    def load_and_hash_data(self, path: Path, format: str) -> tuple[pd.DataFrame, str]:
        HASH_REGISTRY = {
            "parquet": self.hash_parquet_metadata,
            "arrow": self.hash_arrow_metadata,
            "csv": self.hash_streaming,
            "json": self.hash_streaming,
        }

        if format not in HASH_REGISTRY:
            msg = f"Unsupported data format for loading and hashing: {format}"
            logger.error(msg)
            raise ValueError(msg)

        data_hash = HASH_REGISTRY[format](path)

        FORMAT_REGISTRY = {
            "parquet": pd.read_parquet,
            "csv": pd.read_csv,
            "json": pd.read_json,
            "arrow": lambda p: pd.read_feather(p),
        }

        if format not in FORMAT_REGISTRY:
            msg = f"Unsupported data format for loading: {format}"
            logger.error(msg)
            raise ValueError(msg)
        
        data = FORMAT_REGISTRY[format](path)
        return data, data_hash
    
    def apply_segmentation(self, data: pd.DataFrame, config: dict) -> pd.DataFrame:
        seg_cfg = config.get("segmentation", {})
        
        if not seg_cfg.get("enabled", False):
            return data

        df = data.copy()

        OP_MAP = {
            "eq": lambda s, v: s == v,
            "neq": lambda s, v: s != v,
            "in": lambda s, v: s.isin(v),
            "not_in": lambda s, v: ~s.isin(v),
            "gt": lambda s, v: s > v,
            "gte": lambda s, v: s >= v,
            "lt": lambda s, v: s < v,
            "lte": lambda s, v: s <= v,
        }

        for f in seg_cfg.get("filters", []):
            col = f["column"]
            op = f["op"]
            val = f["value"]

            if op not in OP_MAP:
                msg = f"Unsupported segmentation op: {op}"
                logger.error(msg)
                raise ValueError(msg)

            if col not in df.columns:
                msg = f"Segmentation column {col} not found in data."
                logger.error(msg)
                raise ValueError(msg)

            values = val if isinstance(val, (list, tuple, set)) else [val]
            for v in values:
                if v not in df[col].unique():
                    msg = f"Segmentation value {v} for column {col} not found in data."
                    logger.error(msg)
                    raise ValueError(msg)

            df = df[OP_MAP[op](df[col], val)]

        return df
    
    def validate_min_rows(self, data: pd.DataFrame, min_rows: int):
        if not min_rows:
            logger.warning("Minimum rows constraint not set.")
        
        logger.info(f"Validating minimum rows: data has {len(data)} rows, minimum required is {min_rows}.")
        
        if len(data) < min_rows:
            msg = f"Data has {len(data)} rows, which is less than the minimum required {min_rows} rows."
            logger.error(msg)
            raise ValueError(msg)
    
    def validate_min_class_count(self, y: pd.Series, min_class_count: int):
        if y.nunique() < 2:
            msg = "Target variable must have at least two classes for classification."
            logger.error(msg)
            raise ValueError(msg)
        
        if not min_class_count:
            logger.warning("Minimum class count constraint not set.")
        
        logger.info(f"Validating minimum class count: minimum required is {min_class_count}.")
        
        class_counts = y.value_counts()
        for cls, count in class_counts.items():
            if count < min_class_count:
                msg = f"Class {cls} has {count} instances, which is less than the minimum required {min_class_count}."
                logger.error(msg)
                raise ValueError(msg)
            else:
                logger.info(f"Class {cls} has {count} instances, which meets the minimum required {min_class_count}.")

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
        
        positive_class = config["target"].get("classes", {}).get("positive_class", None)
        if positive_class is not None and positive_class not in y.unique():
            msg = f"Positive class {positive_class} not found in target variable."
            logger.error(msg)
            raise ValueError(msg)
        
        if config["target"]["problem_type"] == "classification":
            return  # No further checks for classification

        target_constraints = config.get("target", {}).get("constraints", {})
        min_val = target_constraints.get("min_value", None)
        max_val = target_constraints.get("max_value", None)
        if min_val is not None and y.min() < min_val:
            msg = f"Target min {y.min()} < allowed min {min_val}"
            logger.error(msg)
            raise ValueError(msg)
        if max_val is not None and y.max() > max_val:
            msg = f"Target max {y.max()} > allowed max {max_val}"
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

    def add_arrival_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        if not all(col in df.columns for col in ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']):
            logger.warning("Arrival date columns not found, skipping arrival_datetime creation.")
            return df
        
        df['arrival_datetime'] = pd.to_datetime(
            df['arrival_date_year'].astype(str) + '-' +
            df['arrival_date_month'].astype(str) + '-' +
            df['arrival_date_day_of_month'].astype(str)
        )
        return df

    def apply_operators(self, X: pd.DataFrame, operator_names: list[str]) -> pd.DataFrame:
        operators = [FEATURE_OPERATORS[name]() for name in operator_names]
        for op in operators:
            X = op.transform(X)
        return X

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

    def save_input_schema(self, path: Path, X_train: pd.DataFrame):
        # Stop if raw schema already exists
        schema_path = path / "input_schema.csv"
        if schema_path.exists():
            logger.info(f"Input schema already exists at {schema_path}, skipping save.")
            return

        schema = pd.DataFrame({
            "feature": X_train.columns,
            "dtype": X_train.dtypes.astype(str),
            "role": "input",
        })

        schema.to_csv(schema_path, index=False)
        logger.info(f"Input schema saved to {schema_path}")
    
    def save_derived_schema(self, path: Path, X_train: pd.DataFrame, operator_names: list[str], mode: str):
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
                    "materialized": mode == "materialized",
                })

        derived_schema = pd.DataFrame(derived_features)
        derived_schema.to_csv(schema_path, index=False)
        logger.info(f"Derived schema saved to {schema_path}")

    def hash_config(self, config: dict) -> str:
        config_str = yaml.dump(config, sort_keys=True)
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()
    
    def hash_data_schema(self, X: pd.DataFrame) -> str:
        arr = pd.util.hash_pandas_object(X, index=True).to_numpy()
        return hashlib.md5(arr.tobytes()).hexdigest()

    def hash_feature_set(self, X: pd.DataFrame) -> str:
        h = hashlib.sha256()
        for col in X.columns:
            h.update(col.encode())
            h.update(str(X[col].dtype).encode())
        return h.hexdigest()
    
    def validate_feature_set_hashes_match(self, X: pd.DataFrame, expected_hash: str):
        actual_hash = self.hash_feature_set(X)
        if actual_hash != expected_hash:
            msg = f"Feature set hash mismatch: expected {expected_hash}, got {actual_hash}"
            logger.error(msg)
            raise ValueError(msg)

    def create_metadata(self, snapshot_path: Path, schema_path: Path, data_hash: str, train_schema_hash: str, val_schema_hash: str, test_schema_hash: str, operators_hash: str, config_hash: str, feature_set_hash: str, git_commit: str | None, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series, task: str) -> dict:
        metadata = {
            "created_by": "freeze.py",
            "created_at": datetime.now().isoformat(),
            "feature_type": "tabular",

            "snapshot_path": str(snapshot_path),
            "schema_path": str(schema_path),

            "data_hash": data_hash,
            "schema_hashes": {
                "train": train_schema_hash,
                "val": val_schema_hash,
                "test": test_schema_hash,
            },
            "operators_hash": operators_hash,
            "config_hash": config_hash,
            "feature_set_hash": feature_set_hash,
            "git_commit": git_commit,

            "row_counts": {
                "train": len(X_train),
                "val": len(X_val),
                "test": len(X_test),
            },
            "column_count": X_train.shape[1],
        }

        if task == "classification":
            metadata["class_counts"] = {
                "train": y_train.value_counts().to_dict(),
                "val": y_val.value_counts().to_dict(),
                "test": y_test.value_counts().to_dict(),
            }

        return metadata

    def freeze(self, context, config):
        data, data_hash = self.load_and_hash_data(Path(config["data"]["path"]), config["data"]["format"])
        
        data = self.apply_segmentation(data, config)
        self.validate_min_rows(data, config.get("min_rows", 0))
        if config.get("task", "classification") == "classification":
            self.validate_min_class_count(data[config["target"]["name"]], config.get("min_class_count", 0))

        self.validate_operators(config["operators"]["list"], config["operators"]["hash"])
        self.validate_include_exclude_columns(config)

        X, y = self.prepare_features(data, config)
        X = self.add_arrival_datetime(X)
        self.validate_data_types(X, y, config)
        self.validate_target(y, config)
        self.validate_constraints(X, config)

        if config["operators"]["mode"] == "materialized":
            X = self.apply_operators(X, config["operators"]["list"])

        X_train_val, X_test, y_train_val, y_test = self.split_data(X, y, config, test_size=config["split"]["test_size"])

        relative_val_size = config["split"]["val_size"] / (1.0 - config["split"]["test_size"])

        X_train, X_val, y_train, y_val = self.split_data(X_train_val, y_train_val, config, test_size=relative_val_size)
        
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        snapshot_path = self.freeze_features(context, config, X_train, X_val, X_test, y_train, y_val, y_test, now)

        schema_path = Path(f"data/feature_store/{context.problem}/{context.segment}/{context.feature_set}/{context.version}")

        self.save_input_schema(schema_path, X_train)

        self.save_derived_schema(schema_path, X_train, config["operators"]["list"], config["operators"]["mode"])

        config_hash = self.hash_config(config)
        train_schema_hash = self.hash_data_schema(X_train)
        val_schema_hash = self.hash_data_schema(X_val)
        test_schema_hash = self.hash_data_schema(X_test)

        feature_set_hash = self.hash_feature_set(X_train)
        self.validate_feature_set_hashes_match(X_val, feature_set_hash)
        self.validate_feature_set_hashes_match(X_test, feature_set_hash)
        
        git_commit = get_git_commit(Path("."))

        metadata = self.create_metadata(snapshot_path, schema_path, data_hash, train_schema_hash, val_schema_hash, test_schema_hash, config["operators"]["hash"], config_hash, feature_set_hash, git_commit, X_train, X_val, X_test, y_train, y_val, y_test, config["target"]["problem_type"])

        return snapshot_path, metadata