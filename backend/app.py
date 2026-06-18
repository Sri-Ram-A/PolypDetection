from __future__ import annotations
import architectures
import sys

# Make architectures classes visible as if they were defined in __main__
# Must happen before any torch.load call
_main = sys.modules["__main__"]
for _name in dir(architectures):
    _obj = getattr(architectures, _name)
    if isinstance(_obj, type):
        setattr(_main, _name, _obj)
# 1. Imports

import torch
import streamlit as st
from PIL import Image
import pynvml  # type: ignore[import]
import artifacts_reader
import config
import inference
import visualization as viz


# 2. Page Configuration
st.set_page_config(
    page_title="Polyp Detection Analysis ",
    layout="wide",
    initial_sidebar_state="expanded",
)


# 4. Session State Initialisation
if "run_log" not in st.session_state:
    st.session_state.run_log = []  # accumulates profiler metrics across the session


# 5. Header
st.title("Polyp Detection Analysis ")
st.caption(
    "Classification, segmentation, and localisation models for colonoscopy "
    "image analysis, with live compute and sustainability profiling."
)
st.divider()


# 6. Sidebar - Image Upload, Model Selection, System Info
with st.sidebar:
    st.header("Input")
    uploaded_file = st.file_uploader(
        "Upload a colonoscopy frame", type=["png", "jpg", "jpeg"]
    )

    st.header("Model Selection")
    classification_choices = st.multiselect(
        "Classification models",
        options=[model.key for model in config.CLASSIFICATION_MODELS],
        format_func=lambda key: config.MODELS_BY_KEY[key].display_name,
    )
    segmentation_choices = st.multiselect(
        "Segmentation models",
        options=[model.key for model in config.SEGMENTATION_MODELS],
        format_func=lambda key: config.MODELS_BY_KEY[key].display_name,
    )
    detection_choices = st.multiselect(
        "Localisation models",
        options=[model.key for model in config.DETECTION_MODELS],
        format_func=lambda key: config.MODELS_BY_KEY[key].display_name,
    )

    st.divider()
    with st.expander("System Information"):
        device = inference.get_device()
        st.write(f"Compute device: {device}")
        if torch.cuda.is_available():
            st.write(f"GPU: {torch.cuda.get_device_name(0)}")
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            st.write(f"GPU memory: {total_gb:.1f} GB")
        else:
            st.write("GPU: not available, running on CPU")
    st.caption(f"Models directory: {config.MODELS_DIR}")


# 7. Tab Layout
tab_pipeline, tab_classification, tab_segmentation, tab_detection, tab_compute = (
    st.tabs(
        [
            "Image Analysis",
            "Classification Explorer",
            "Segmentation Explorer",
            "Localisation Explorer",
            "Compute and Sustainability",
        ]
    )
)


# 8. Image Analysis Tab - end to end triage pipeline
with tab_pipeline:
    if uploaded_file is None:
        st.info("Upload a colonoscopy frame from the sidebar to begin analysis.")
    else:
        image = Image.open(uploaded_file)
        column_image, column_verdict = st.columns([1, 1.4])

        with column_image:
            st.image(image, caption="Input frame", width="stretch")

        # 8.1 Run classification models first - they decide the downstream path
        if not classification_choices:
            st.warning("Select at least one classification model from the sidebar.")
        else:
            classification_results = []
            with st.spinner("Running classification models..."):
                for key in classification_choices:
                    spec = config.MODELS_BY_KEY[key]
                    try:
                        result = inference.run_classification(spec, image)
                        classification_results.append(result)
                        st.session_state.run_log.append(result)
                    except Exception as exc:
                        st.error(f"{spec.display_name} failed to run: {exc}")

            if classification_results:
                decision = inference.aggregate_polyp_decision(classification_results)

                with column_verdict:
                    verdict_css = (
                        "verdict-polyp" if decision["is_polyp"] else "verdict-clear"
                    )
                    verdict_text = (
                        "Polyp Detected"
                        if decision["is_polyp"]
                        else "No Polyp Detected"
                    )
                    st.markdown(
                        f"""
                        <div class="{verdict_css}">
                        <h3>{verdict_text}</h3>
                        <p>Mean polyp probability: {decision["mean_polyp_probability"]:.3f}</p>
                        <p>Models voting polyp: {decision["votes_for_polyp"]} / {decision["total_models"]}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.subheader("Classification Report")
                report_columns = st.columns(len(classification_results))
                for column, result in zip(report_columns, classification_results):
                    with column:
                        st.markdown(f"**{result['display_name']}**")
                        st.metric("Prediction", result["predicted_label"])
                        st.metric("Confidence", f"{result['confidence']:.3f}")
                        st.metric(
                            "Latency", f"{result['metrics']['latency_ms']:.1f} ms"
                        )
                        st.plotly_chart(
                            viz.plot_class_probabilities(result, config.CLASS_NAMES),
                            width="stretch",
                        )

                # 8.2 Route to segmentation and localisation only if polyp detected
                if decision["is_polyp"]:
                    st.divider()
                    st.subheader("Segmentation Results")
                    if not segmentation_choices:
                        st.warning(
                            "Select at least one segmentation model from the sidebar."
                        )
                    else:
                        segmentation_results = []
                        with st.spinner("Running segmentation models..."):
                            for key in segmentation_choices:
                                spec = config.MODELS_BY_KEY[key]
                                try:
                                    result = inference.run_segmentation(spec, image)
                                    segmentation_results.append(result)
                                    st.session_state.run_log.append(result)
                                except Exception as exc:
                                    st.error(
                                        f"{spec.display_name} failed to run: {exc}"
                                    )

                        if segmentation_results:
                            mask_columns = st.columns(min(3, len(segmentation_results)))
                            for index, result in enumerate(segmentation_results):
                                with mask_columns[index % len(mask_columns)]:
                                    overlay = viz.overlay_mask(
                                        image, result["binary_mask"]
                                    )
                                    st.image(
                                        overlay,
                                        caption=result["display_name"],
                                        width="stretch",
                                    )
                                    st.caption(
                                        f"Polyp area: {result['polyp_pixel_ratio'] * 100:.1f}% | "
                                        f"Latency: {result['metrics']['latency_ms']:.1f} ms"
                                    )
                            st.plotly_chart(
                                viz.plot_pixel_ratio_comparison(segmentation_results),
                                width="stretch",
                            )

                    st.divider()
                    st.subheader("Localisation Results")
                    if not detection_choices:
                        st.warning("Select a localisation model from the sidebar.")
                    else:
                        for key in detection_choices:
                            spec = config.MODELS_BY_KEY[key]
                            try:
                                with st.spinner(f"Running {spec.display_name}..."):
                                    detection_result = inference.run_detection(
                                        spec, image
                                    )
                                st.session_state.run_log.append(detection_result)
                                if detection_result["boxes"]:
                                    boxed_image = viz.draw_boxes(
                                        image,
                                        detection_result["boxes"],
                                        detection_result["scores"],
                                    )
                                    st.image(
                                        boxed_image,
                                        caption=(
                                            f"{spec.display_name} - "
                                            f"{len(detection_result['boxes'])} detection(s)"
                                        ),
                                        width="stretch",
                                    )
                                else:
                                    st.info(
                                        f"{spec.display_name} found no detections above "
                                        "the score threshold."
                                    )
                                st.caption(
                                    f"Latency: {detection_result['metrics']['latency_ms']:.1f} ms"
                                )
                            except Exception as exc:
                                st.error(f"{spec.display_name} failed to run: {exc}")
                else:
                    st.info(
                        "No polyp was detected by the classification ensemble, so "
                        "segmentation and localisation were skipped."
                    )


# 9. Classification Explorer Tab - browse stored evaluation artifacts
with tab_classification:
    st.subheader("Classification Model Explorer")
    selected_key = st.selectbox(
        "Select a classification model",
        options=[model.key for model in config.CLASSIFICATION_MODELS],
        format_func=lambda key: config.MODELS_BY_KEY[key].display_name,
        key="explorer_classification_select",
    )
    spec = config.MODELS_BY_KEY[selected_key]
    st.markdown(f"<p class='model-note'>{spec.notes}</p>", unsafe_allow_html=True)

    info_columns = st.columns(3)
    info_columns[0].metric("Architecture", spec.architecture)
    info_columns[1].metric(
        "Checkpoint Size", f"{artifacts_reader.get_checkpoint_size_mb(spec)} MB"
    )
    info_columns[2].metric("Input Resolution", f"{spec.input_size} x {spec.input_size}")

    history = artifacts_reader.load_history(spec)
    if history:
        st.plotly_chart(
            viz.plot_history_curves(history, f"{spec.display_name} - Training History"),
            width="stretch",
        )
    else:
        st.caption("No history.json found for this model.")

    stored_images = artifacts_reader.list_available_images(spec)
    if stored_images:
        st.subheader("Stored Evaluation Charts")
        image_columns = st.columns(2)
        for index, (name, path) in enumerate(stored_images.items()):
            with image_columns[index % 2]:
                caption = name.replace("_", " ").replace(".png", "").title()
                st.image(str(path), caption=caption, width="stretch")
    else:
        st.caption("No stored evaluation charts found for this model.")


# 10. Segmentation Explorer Tab - browse stored evaluation artifacts
with tab_segmentation:
    st.subheader("Segmentation Model Explorer")
    selected_key = st.selectbox(
        "Select a segmentation model",
        options=[model.key for model in config.SEGMENTATION_MODELS],
        format_func=lambda key: config.MODELS_BY_KEY[key].display_name,
        key="explorer_segmentation_select",
    )
    spec = config.MODELS_BY_KEY[selected_key]
    st.markdown(f"<p class='model-note'>{spec.notes}</p>", unsafe_allow_html=True)

    info_columns = st.columns(3)
    info_columns[0].metric("Architecture", "Custom (pickled module)")
    info_columns[1].metric(
        "Checkpoint Size", f"{artifacts_reader.get_checkpoint_size_mb(spec)} MB"
    )
    info_columns[2].metric("Input Resolution", f"{spec.input_size} x {spec.input_size}")

    stored_images = artifacts_reader.list_available_images(spec)
    if stored_images:
        st.subheader("Stored Evaluation Charts")
        image_columns = st.columns(2)
        for index, (name, path) in enumerate(stored_images.items()):
            with image_columns[index % 2]:
                caption = name.replace("_", " ").replace(".png", "").title()
                st.image(str(path), caption=caption, width="stretch")
    else:
        st.caption("No stored evaluation charts found for this model.")


# 11. Localisation Explorer Tab - browse stored evaluation artifacts
with tab_detection:
    st.subheader("Localisation Model Explorer")
    selected_key = st.selectbox(
        "Select a localisation model",
        options=[model.key for model in config.DETECTION_MODELS],
        format_func=lambda key: config.MODELS_BY_KEY[key].display_name,
        key="explorer_detection_select",
    )
    spec = config.MODELS_BY_KEY[selected_key]
    st.markdown(f"<p class='model-note'>{spec.notes}</p>", unsafe_allow_html=True)

    info_columns = st.columns(2)
    info_columns[0].metric("Architecture", "RetinaNet (pickled module)")
    info_columns[1].metric(
        "Checkpoint Size", f"{artifacts_reader.get_checkpoint_size_mb(spec)} MB"
    )

    stored_images = artifacts_reader.list_available_images(spec)
    if stored_images:
        st.subheader("Stored Evaluation Charts")
        for name, path in stored_images.items():
            st.image(str(path), caption=name, width="stretch")
    else:
        st.caption(
            "No stored evaluation charts are available for this model. "
            "Run an image through the Image Analysis tab to see live results."
        )

    with st.expander("Files in model folder"):
        for file_name in artifacts_reader.list_all_files(spec):
            st.text(file_name)


# 12. Compute and Sustainability Tab - cross-category live metric comparison
with tab_compute:
    st.subheader("Compute and Sustainability Comparison")
    st.caption(
        "Metrics below are captured live via the profiler during this "
        "session. Run models from the Image Analysis tab to populate this view."
    )

    if not st.session_state.run_log:
        st.info("No inference runs recorded yet in this session.")
    else:
        run_log = st.session_state.run_log
        metric_columns = st.columns(2)
        with metric_columns[0]:
            st.plotly_chart(
                viz.plot_compute_comparison(
                    run_log, "latency_ms", "Inference Latency", "Latency (ms)"
                ),
                width="stretch",
            )
            st.plotly_chart(
                viz.plot_compute_comparison(
                    run_log, "emissions_kg_co2", "Estimated CO2 Emissions", "kg CO2"
                ),
                width="stretch",
            )
        with metric_columns[1]:
            st.plotly_chart(
                viz.plot_compute_comparison(
                    run_log, "ram_delta_mb", "RAM Delta per Run", "RAM Delta (MB)"
                ),
                width="stretch",
            )
            st.plotly_chart(
                viz.plot_compute_comparison(
                    run_log, "cpu_utilization_pct", "CPU Utilisation", "CPU (%)"
                ),
                width="stretch",
            )

        st.divider()
        if st.button("Clear session log"):
            st.session_state.run_log = []
            st.rerun()
