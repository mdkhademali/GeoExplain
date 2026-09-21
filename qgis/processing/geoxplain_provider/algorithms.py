"""GeoExplain Processing algorithms (thin wrappers around the ``geoxplain`` package)."""

from pathlib import Path

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFile,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
)

from .helpers import import_geoxplain, layer_to_dataframe

ALGORITHMS = ["random_forest", "xgboost", "lightgbm", "mlp", "logistic_regression", "svm"]
TASKS = ["classification", "regression"]
VALIDATIONS = ["spatial_split", "random_split", "none"]


class _Base(QgsProcessingAlgorithm):
    """Common metadata for all GeoExplain algorithms."""

    def group(self):
        return "Explainable GeoAI"

    def groupId(self):  # noqa: N802
        return "explainable_geoai"

    def createInstance(self):  # noqa: N802
        return type(self)()

    def tr(self, text):
        return text

    # -- parameter helpers ---------------------------------------------------------------
    def _add_table_params(self, with_features=True):
        self.addParameter(QgsProcessingParameterFeatureSource(
            "INPUT", "Input observations (points or polygons; centroids are used)"))
        self.addParameter(QgsProcessingParameterField(
            "TARGET", "Target field", parentLayerParameterName="INPUT", type=QgsProcessingParameterField.Any))
        if with_features:
            self.addParameter(QgsProcessingParameterField(
                "FEATURES", "Feature fields", parentLayerParameterName="INPUT",
                type=QgsProcessingParameterField.Numeric, allowMultiple=True))
        self.addParameter(QgsProcessingParameterEnum("TASK", "Task", options=TASKS, defaultValue=0))

    def _dataset(self, parameters, context, feedback, features=None):
        gx = import_geoxplain()
        source = self.parameterAsSource(parameters, "INPUT", context)
        if source is None:
            raise QgsProcessingException("Invalid input layer.")
        target = self.parameterAsString(parameters, "TARGET", context)
        feats = features or self.parameterAsFields(parameters, "FEATURES", context)
        task = TASKS[self.parameterAsEnum(parameters, "TASK", context)]
        df = layer_to_dataframe(source, [target, *feats], feedback)
        return gx.io.dataframe_to_dataset(df, target=target, features=feats, task=task,
                                          crs=source.sourceCrs().authid() or None)


class TrainModelAlgorithm(_Base):
    """Train a model and save it as a ``.joblib`` file."""

    def name(self):
        return "train_model"

    def displayName(self):  # noqa: N802
        return "Train Geospatial ML Model"

    def shortHelpString(self):  # noqa: N802
        return ("Trains a Random Forest, XGBoost, LightGBM, MLP, logistic regression or SVM model on the attribute "
                "table and saves it. A spatially blocked hold-out estimate is reported in the log. "
                "Only open model files from sources you trust (joblib files can execute code).")

    def initAlgorithm(self, config=None):  # noqa: N802
        self._add_table_params()
        self.addParameter(QgsProcessingParameterEnum("ALGORITHM", "Algorithm", options=ALGORITHMS, defaultValue=0))
        self.addParameter(QgsProcessingParameterEnum("VALIDATION", "Validation report", options=VALIDATIONS, defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber("SEED", "Random seed", type=QgsProcessingParameterNumber.Integer,
                                                       defaultValue=42))
        self.addParameter(QgsProcessingParameterFileDestination("OUTPUT", "Output model", fileFilter="Model (*.joblib)"))

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        gx = import_geoxplain()
        ds = self._dataset(parameters, context, feedback)
        algo = ALGORITHMS[self.parameterAsEnum(parameters, "ALGORITHM", context)]
        seed = self.parameterAsInt(parameters, "SEED", context)
        validation = VALIDATIONS[self.parameterAsEnum(parameters, "VALIDATION", context)]
        if validation != "none":
            if validation == "spatial_split":
                tr, te = gx.validation.spatial_split(ds.require_coords(), 0.25, None, seed)
            else:
                tr, te = gx.validation.random_split(len(ds.X), 0.25, seed)
            probe = gx.GeoExplainModel(algo, ds.task, random_state=seed).fit(ds.X.iloc[tr], ds.y.iloc[tr])
            for k, v in probe.evaluate(ds.X.iloc[te], ds.y.iloc[te]).items():
                feedback.pushInfo(f"{validation} hold-out {k}: {v:.4f}")
        model = gx.GeoExplainModel(algo, ds.task, random_state=seed).fit(ds.X, ds.y)
        out = self.parameterAsFileOutput(parameters, "OUTPUT", context)
        model.save(out)
        feedback.pushInfo("Saved model (refit on all observations).")
        return {"OUTPUT": out}


class ExplainAlgorithm(_Base):
    """Global and local SHAP explanations for point data."""

    def name(self):
        return "generate_shap_explanation"

    def displayName(self):  # noqa: N802
        return "Generate SHAP Explanation"

    def shortHelpString(self):  # noqa: N802
        return ("Computes SHAP values for the observations and writes global (summary, bar), permutation-importance "
                "and partial-dependence figures plus CSV tables to the output folder. SHAP describes model behaviour, "
                "not causal effects.")

    def initAlgorithm(self, config=None):  # noqa: N802
        self.addParameter(QgsProcessingParameterFile("MODEL", "Trained model (.joblib)", extension="joblib"))
        self.addParameter(QgsProcessingParameterFeatureSource("INPUT", "Observations"))
        self.addParameter(QgsProcessingParameterField("TARGET", "Target field (for permutation importance)",
                                                      parentLayerParameterName="INPUT", type=QgsProcessingParameterField.Any))
        self.addParameter(QgsProcessingParameterNumber("MAX_SAMPLES", "Maximum observations to explain",
                                                       type=QgsProcessingParameterNumber.Integer, defaultValue=400, minValue=10))
        self.addParameter(QgsProcessingParameterFolderDestination("OUTPUT_FOLDER", "Output folder"))

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        gx = import_geoxplain()
        from geoxplain import visualization as viz

        model = gx.GeoExplainModel.load(self.parameterAsFile(parameters, "MODEL", context))
        source = self.parameterAsSource(parameters, "INPUT", context)
        target = self.parameterAsString(parameters, "TARGET", context)
        df = layer_to_dataframe(source, [target, *model.feature_names_], feedback)
        ds = gx.io.dataframe_to_dataset(df, target=target, features=model.feature_names_, task=model.task,
                                        crs=source.sourceCrs().authid() or None)
        out = Path(self.parameterAsString(parameters, "OUTPUT_FOLDER", context))
        (out / "figures").mkdir(parents=True, exist_ok=True)
        n = min(self.parameterAsInt(parameters, "MAX_SAMPLES", context), len(ds.X))
        sample = ds.X.sample(n, random_state=42)
        sv = gx.GeoShapExplainer(model, background=ds.X.sample(min(100, len(ds.X)), random_state=42)).explain(sample)
        sv.mean_abs().to_csv(out / "shap_global.csv", index=False)
        sv.to_frame(include_data=True).to_csv(out / "shap_values.csv", index=False)
        viz.plot_shap_summary(sv, path=out / "figures" / "shap_summary")
        viz.plot_shap_bar(sv, path=out / "figures" / "shap_bar")
        perm = gx.permutation_importance(model, ds.X, ds.y, n_repeats=5)
        perm.to_csv(out / "permutation_importance.csv", index=False)
        viz.plot_importance(perm, path=out / "figures" / "permutation_importance")
        return {"OUTPUT_FOLDER": str(out)}


class SpatialShapAlgorithm(_Base):
    """Per-cell SHAP GeoTIFFs plus prediction, uncertainty and dominant-driver rasters."""

    def name(self):
        return "generate_spatial_shap"

    def displayName(self):  # noqa: N802
        return "Generate Spatial SHAP"

    def shortHelpString(self):  # noqa: N802
        return ("Applies the model to every valid cell of a multi-band predictor raster and writes one SHAP GeoTIFF per "
                "feature (e.g. NDVI_SHAP.tif) plus prediction.tif, uncertainty.tif (when supported) and dominant_driver.tif. "
                "CRS, transform, size and nodata of the input grid are preserved. Band descriptions must be the feature "
                "names, or list them in band order below.")

    def initAlgorithm(self, config=None):  # noqa: N802
        self.addParameter(QgsProcessingParameterFile("MODEL", "Trained model (.joblib)", extension="joblib"))
        self.addParameter(QgsProcessingParameterRasterLayer("RASTER", "Multi-band predictor raster"))
        self.addParameter(QgsProcessingParameterString("BAND_NAMES", "Feature names in band order (comma separated; optional)",
                                                       optional=True))
        self.addParameter(QgsProcessingParameterFolderDestination("OUTPUT_FOLDER", "Output folder"))

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        gx = import_geoxplain()
        model = gx.GeoExplainModel.load(self.parameterAsFile(parameters, "MODEL", context))
        raster = self.parameterAsRasterLayer(parameters, "RASTER", context)
        names_txt = self.parameterAsString(parameters, "BAND_NAMES", context)
        names = [n.strip() for n in names_txt.split(",") if n.strip()] or None
        stack = gx.io.read_raster_stack(raster.source(), names=names)
        bg = stack.to_dataframe()[model.feature_names_]
        bg = bg.sample(min(100, len(bg)), random_state=42)
        feedback.pushInfo(f"Explaining {int(stack.valid_mask.sum())} valid cells ...")
        res = gx.compute_spatial_shap(model, stack, background=bg, feature_labels=gx.datasets.FEATURE_LABELS)
        out = Path(self.parameterAsString(parameters, "OUTPUT_FOLDER", context))
        written = res.write(out)
        for note in res.notes:
            feedback.pushInfo(note)
        return {"OUTPUT_FOLDER": str(out), **{k: str(v) for k, v in written.items()}}


class CompareModelsAlgorithm(_Base):
    """Benchmark models with random and spatial validation and write an HTML report."""

    def name(self):
        return "compare_models"

    def displayName(self):  # noqa: N802
        return "Compare Models"

    def shortHelpString(self):  # noqa: N802
        return ("Runs the full GeoExplain experiment (random vs spatial validation for every selected model, feature "
                "importance, SHAP, optional spatial SHAP) and writes tables, figures and report.html.")

    def initAlgorithm(self, config=None):  # noqa: N802
        self._add_table_params()
        self.addParameter(QgsProcessingParameterEnum("MODELS", "Models to compare", options=ALGORITHMS, allowMultiple=True,
                                                     defaultValue=[0, 1, 2, 3]))
        self.addParameter(QgsProcessingParameterRasterLayer("RASTER", "Multi-band predictor raster (optional)", optional=True))
        self.addParameter(QgsProcessingParameterNumber("FOLDS", "Cross-validation folds", type=QgsProcessingParameterNumber.Integer,
                                                       defaultValue=5, minValue=2))
        self.addParameter(QgsProcessingParameterFolderDestination("OUTPUT_FOLDER", "Output folder"))

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        gx = import_geoxplain()
        ds = self._dataset(parameters, context, feedback)
        out = Path(self.parameterAsString(parameters, "OUTPUT_FOLDER", context))
        out.mkdir(parents=True, exist_ok=True)
        csv = out / "input_observations.csv"
        table = ds.X.copy()
        table.insert(0, "y", ds.coords["y"].to_numpy())
        table.insert(0, "x", ds.coords["x"].to_numpy())
        table.insert(0, "id", ds.ids.to_numpy())
        table[ds.target_name] = ds.y.to_numpy()
        table.to_csv(csv, index=False)
        raster = self.parameterAsRasterLayer(parameters, "RASTER", context)
        models = [ALGORITHMS[i] for i in self.parameterAsEnums(parameters, "MODELS", context)]
        cfg = gx.ExperimentConfig(data=str(csv), target=ds.target_name, features=ds.feature_names, task=ds.task,
                                  crs=ds.crs, raster=raster.source() if raster else None, models=models,
                                  primary_model=models[0], out_dir=str(out / "experiment"),
                                  n_splits=self.parameterAsInt(parameters, "FOLDS", context),
                                  dataset_label="User-supplied data")
        gx.run_experiment(cfg)
        report = gx.build_report(cfg.out_dir)
        feedback.pushInfo(f"Report: {report}")
        return {"OUTPUT_FOLDER": str(out), "REPORT": str(report)}


class PredictRasterAlgorithm(_Base):
    """Apply a trained model to a predictor raster."""

    def name(self):
        return "generate_prediction_raster"

    def displayName(self):  # noqa: N802
        return "Generate Prediction Raster"

    def shortHelpString(self):  # noqa: N802
        return ("Applies a trained model to a multi-band predictor raster and writes the prediction "
                "(class probability or regression value) as a GeoTIFF on the same grid.")

    def initAlgorithm(self, config=None):  # noqa: N802
        self.addParameter(QgsProcessingParameterFile("MODEL", "Trained model (.joblib)", extension="joblib"))
        self.addParameter(QgsProcessingParameterRasterLayer("RASTER", "Multi-band predictor raster"))
        self.addParameter(QgsProcessingParameterString("BAND_NAMES", "Feature names in band order (optional)", optional=True))
        self.addParameter(QgsProcessingParameterRasterDestination("OUTPUT", "Prediction raster"))

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        gx = import_geoxplain()
        model = gx.GeoExplainModel.load(self.parameterAsFile(parameters, "MODEL", context))
        raster = self.parameterAsRasterLayer(parameters, "RASTER", context)
        names = [n.strip() for n in self.parameterAsString(parameters, "BAND_NAMES", context).split(",") if n.strip()] or None
        stack = gx.io.read_raster_stack(raster.source(), names=names)
        pred = gx.predict_raster(model, stack)
        out = self.parameterAsOutputLayer(parameters, "OUTPUT", context)
        gx.write_raster(out, pred.prediction, stack)
        return {"OUTPUT": out}
