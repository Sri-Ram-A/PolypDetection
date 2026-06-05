from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

from backend.video_processor import VideoProcessor

st.set_page_config(
    page_title="Polyp Detection and Segmentation System",
    page_icon="🩺",
    layout="wide"
)

st.title(
    "🩺 AI-Based Polyp Detection System"
)

st.markdown(
    """
    Upload an endoscopy video and perform:

    - Classification
    - Segmentation
    - Video Analysis
    """
)

with st.sidebar:

    st.title("About")

    st.write(
        """
        AI-Based Polyp Detection System

        Models:
        • DenseNet-121
        • PraNet

        Dataset:
        • Hyper-Kvasir
        • Kvasir-SEG
        """
    )

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = (
    BASE_DIR
    / "uploads"
)

RESULT_DIR = (
    BASE_DIR
    / "results"
)

UPLOAD_DIR.mkdir(
    exist_ok=True
)

RESULT_DIR.mkdir(
    exist_ok=True
)

uploaded_file = st.file_uploader(
    "Upload Endoscopy Video",
    type=[
        "avi",
        "mp4",
        "mov",
        "mkv"
    ]
)

if uploaded_file:

    st.subheader(
        "Uploaded Video"
    )

    st.video(
        uploaded_file
    )
    
if uploaded_file:

    upload_path = (
        UPLOAD_DIR
        / uploaded_file.name
    )

    with open(
        upload_path,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )
        
if uploaded_file:

    if st.button("Run Analysis"):
        with st.spinner("Processing Video..."):
            processor = VideoProcessor()

            video_name = Path(
                uploaded_file.name
            ).stem

            output_path = (
                RESULT_DIR
                / f"processed_{video_name}.mp4"
            )

            stats = (
                processor.process_video(
                    input_video_path=upload_path,
                    output_video_path=output_path
                )
            )
            
            st.success(
                "Processing Complete!"
            )
            
            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Frames Processed",
                stats[
                    "frames_processed"
                ]
            )

            c2.metric(
                "Polyp Frames",
                stats[
                    "polyp_frames"
                ]
            )

            c3.metric(
                "Polyp %",
                f"{stats['polyp_percentage']:.2f}"
            )
            
            st.subheader(
                "Clinical Summary"
            )

            summary_df = pd.DataFrame(
                {
                    "Metric": [
                        "Predominant Finding",
                        "Frames Processed",
                        "Polyp Frames",
                        "Polyp Percentage",
                        "Average Confidence",
                        "Median Confidence",
                        "Maximum Confidence"
                    ],
                    "Value": [
                        stats["dominant_class"],
                        stats["frames_processed"],
                        stats["polyp_frames"],
                        f"{stats['polyp_percentage']:.2f}%",
                        f"{stats['average_confidence']:.2%}",
                        f"{stats['median_confidence']:.2%}",
                        f"{stats['max_confidence']:.2%}"
                    ]
                }
            )
            
            summary_df["Value"] = summary_df["Value"].astype(str)
            
            st.table(summary_df)
            
            st.subheader(
                "Class Distribution"
            )

            df = pd.DataFrame(
                list(
                    stats["class_counts"].items()
                ),
                columns=[
                    "Class",
                    "Count"
                ]
            )

            c1, c2 = st.columns(2)

            with c1:
            
                fig = px.pie(
                    df,
                    names="Class",
                    values="Count",
                    hole=0.5,
                    title="Frame Distribution"
                )

                st.plotly_chart(
                    fig,
                    width='stretch'
                )

            with c2:
            
                fig2 = px.bar(
                    df,
                    x="Class",
                    y="Count",
                    title="Class Frequency"
                )

                st.plotly_chart(
                    fig2,
                    width='stretch'
                )
            
            st.subheader(
                "Processed Video"
            )
            
            print(output_path)
            print(output_path.exists())
            print(output_path.stat().st_size)

            with open(output_path, "rb") as f:

                video_bytes = f.read()

            st.video(
                video_bytes,
                format="video/mp4"
            )
            
            with open(
                output_path,
                "rb"
            ) as f:

                st.download_button(
                    label=
                    "Download Processed Video",

                    data=f,

                    file_name=
                    output_path.name,

                    mime=
                    "video/mp4"
                )