from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
MODEL_CANDIDATES = [
    MODEL_DIR / "artClassifier_model.keras",
    MODEL_DIR / "ResNet50_Final_Classifier.keras",
    MODEL_DIR / "ResNet50_best.keras",
]

CLASS_NAMES = [
    "Art_Nouveau",
    "Baroque",
    "Japanese_Art",
    "Realism",
    "Western_Medieval",
]

CLASS_LABELS_KO = {
    "Art_Nouveau": "아르누보",
    "Baroque": "바로크",
    "Japanese_Art": "일본 미술",
    "Realism": "사실주의",
    "Western_Medieval": "서양 중세",
}

CLASS_DESCRIPTIONS = {
    "Art_Nouveau": "식물적 곡선, 장식적 패턴, 포스터풍 구성이 자주 나타납니다.",
    "Baroque": "극적인 명암, 역동적인 구도, 종교적/궁정적 장면이 특징입니다.",
    "Japanese_Art": "평면적 색면, 간결한 선, 목판화적 구도와 여백이 두드러집니다.",
    "Realism": "일상적 소재와 관찰 중심의 사실적 묘사가 강합니다.",
    "Western_Medieval": "종교 도상, 평면적 표현, 금색 배경과 상징적 구성이 자주 보입니다.",
}

IMAGE_SIZE = (224, 224)


def apply_custom_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --gallery-ink: #f6ead3;
                --gallery-muted: #c8bba3;
                --gallery-wall: #263231;
                --gallery-wall-deep: #151b1b;
                --gallery-slate: #354342;
                --gallery-velvet: #6f1724;
                --gallery-velvet-deep: #3c0e17;
                --gallery-cream: #efe3cd;
                --gallery-wainscot: #ded3c1;
                --gallery-gold: #c8a15a;
                --gallery-gold-dark: #7c5924;
                --gallery-bronze: #9b6b34;
                --gallery-line: rgba(200, 161, 90, 0.32);
                --gallery-shadow: rgba(6, 8, 8, 0.62);
            }

            .stApp {
                color: var(--gallery-ink);
                background: linear-gradient(180deg, #111616 0%, var(--gallery-wall) 100%);
                background-attachment: fixed;
                font-family: Inter, "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
            }

            [data-testid="stHeader"] {
                background: linear-gradient(180deg, rgba(13, 16, 16, 0.88), rgba(13, 16, 16, 0.58));
                border-bottom: 1px solid rgba(200, 161, 90, 0.26);
                backdrop-filter: blur(10px);
            }

            [data-testid="stToolbar"] {
                color: var(--gallery-gold);
            }

            .block-container {
                max-width: 1180px;
                padding-top: 3.2rem;
                padding-bottom: 4rem;
                background: transparent;
            }

            h1, h2, h3, [data-testid="stMarkdownContainer"] h1,
            [data-testid="stMarkdownContainer"] h2,
            [data-testid="stMarkdownContainer"] h3 {
                color: var(--gallery-cream);
                font-family: Inter, "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
                font-weight: 650;
                letter-spacing: 0;
            }

            h1 {
                display: inline-block;
                margin-bottom: 1.6rem;
                padding: 0 0 0.7rem;
                border-bottom: 1px solid rgba(200, 161, 90, 0.58);
                text-shadow: none;
            }

            h2, h3 {
                border-left: 4px solid var(--gallery-gold);
                padding-left: 0.85rem;
            }

            p, label, span, div {
                letter-spacing: 0;
            }

            p, label, [data-testid="stMarkdownContainer"],
            [data-testid="stMarkdownContainer"] p {
                color: var(--gallery-ink);
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, var(--gallery-velvet), var(--gallery-velvet-deep));
                border-right: 1px solid rgba(200, 161, 90, 0.38);
                box-shadow: 8px 0 28px rgba(0, 0, 0, 0.20);
            }

            [data-testid="stSidebar"] * {
                color: #fff0d3 !important;
            }

            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 {
                border-left: 0;
                border-bottom: 1px solid rgba(227, 190, 112, 0.62);
                padding-left: 0;
                padding-bottom: 0.45rem;
            }

            [data-testid="stSidebar"] code {
                color: #ffe2a6 !important;
                background: rgba(18, 9, 8, 0.35);
                border: 1px solid rgba(227, 190, 112, 0.38);
                border-radius: 4px;
            }

            [data-testid="stFileUploader"] {
                background: rgba(239, 227, 205, 0.96);
                border: 1px solid rgba(200, 161, 90, 0.34);
                border-radius: 8px;
                padding: 1rem 1rem 0.75rem;
                box-shadow: 0 14px 32px rgba(0, 0, 0, 0.18);
            }

            [data-testid="stFileUploader"] * {
                color: #2c2119 !important;
            }

            [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] p,
            [data-testid="stFileUploader"] label p {
                font-size: calc(1rem + 2pt) !important;
                font-weight: 700 !important;
                line-height: 1.35;
            }

            [data-testid="stFileUploader"] section {
                background: rgba(250, 244, 232, 0.82);
                border: 1px dashed rgba(124, 89, 36, 0.52);
                border-radius: 8px;
            }

            .stButton button,
            [data-testid="stFileUploader"] button {
                color: #fff !important;
                background: var(--gallery-velvet);
                border: 1px solid rgba(231, 195, 116, 0.48);
                border-radius: 6px;
                box-shadow: none;
                font-family: Inter, "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
                font-weight: 700;
            }

            .stButton button *,
            [data-testid="stFileUploader"] button * {
                color: #fff !important;
                font-weight: 700 !important;
            }

            .stButton button:hover,
            [data-testid="stFileUploader"] button:hover {
                color: #fff !important;
                border-color: #f0c979;
                background: #7f2030;
            }

            [data-testid="stImage"] img {
                background: #111616;
                border: 1px solid rgba(200, 161, 90, 0.55);
                border-radius: 8px;
                box-shadow: 0 18px 42px rgba(0, 0, 0, 0.34);
            }

            [data-testid="stImageCaption"] {
                color: var(--gallery-muted);
                font-style: italic;
            }

            [data-testid="stMetric"] {
                background: rgba(24, 31, 31, 0.86);
                border: 1px solid rgba(200, 161, 90, 0.32);
                border-radius: 8px;
                padding: 1rem 1.1rem;
                box-shadow: 0 14px 32px rgba(0, 0, 0, 0.22);
            }

            [data-testid="stMetricLabel"] {
                color: #f2d9ae;
            }

            [data-testid="stMetricValue"] {
                color: #ffe2a6;
                font-family: Inter, "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
                font-weight: 700;
            }

            [data-testid="stDataFrame"] {
                border: 1px solid rgba(200, 161, 90, 0.38);
                border-radius: 8px;
                box-sizing: border-box;
                box-shadow: 0 14px 32px rgba(0, 0, 0, 0.18);
                max-width: 100%;
                overflow: hidden;
                width: 100%;
            }

            [data-testid="stDataFrame"] > div,
            [data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] {
                box-sizing: border-box;
                max-width: 100% !important;
                width: 100% !important;
            }

            [data-testid="stAlert"] {
                background: rgba(239, 227, 205, 0.96);
                border: 1px solid rgba(200, 161, 90, 0.44);
                border-left: 4px solid var(--gallery-gold);
                color: #2c2119;
            }

            [data-testid="stAlert"] * {
                color: #2c2119 !important;
            }

            [data-testid="stCaptionContainer"] {
                color: var(--gallery-muted);
                font-style: italic;
            }

            [data-testid="stVegaLiteChart"] {
                background: transparent;
                border: 0;
                border-radius: 8px;
                box-shadow: none;
                box-sizing: border-box;
                max-width: 100%;
                overflow: hidden;
                padding: 0;
                width: 100%;
            }

            [data-testid="stVegaLiteChart"] > div,
            [data-testid="stVegaLiteChart"] canvas,
            [data-testid="stVegaLiteChart"] svg {
                box-sizing: border-box;
                max-width: 100% !important;
            }

            hr {
                border-color: var(--gallery-line);
            }

            ::selection {
                color: #fff3d8;
                background: var(--gallery-velvet);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@tf.keras.utils.register_keras_serializable(package="ArtClassifier")
def resnet50_preprocess_input_layer(x):
    return tf.keras.applications.resnet50.preprocess_input(x)


def find_model_path() -> Path:
    for model_path in MODEL_CANDIDATES:
        if model_path.exists():
            return model_path
    candidate_names = "\n".join(f"- {path}" for path in MODEL_CANDIDATES)
    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다.\n{candidate_names}")


def get_model_custom_objects() -> dict[str, object]:
    preprocess_input = tf.keras.applications.resnet50.preprocess_input
    custom_objects = {
        "ArtClassifier>resnet50_preprocess_input_layer": resnet50_preprocess_input_layer,
        "resnet50_preprocess_input_layer": resnet50_preprocess_input_layer,
        "preprocess_input": preprocess_input,
        "function": preprocess_input,
    }
    tf.keras.utils.get_custom_objects().update(custom_objects)
    return custom_objects


def patch_dense_quantization_config() -> None:
    if getattr(tf.keras.layers.Dense, "_art_classifier_quantization_patch", False):
        return

    original_from_config = tf.keras.layers.Dense.from_config

    @classmethod
    def from_config_without_quantization(cls, config):
        config = dict(config)
        config.pop("quantization_config", None)
        return original_from_config(config)

    tf.keras.layers.Dense.from_config = from_config_without_quantization
    tf.keras.layers.Dense._art_classifier_quantization_patch = True


@st.cache_resource(show_spinner="모델을 불러오는 중입니다...")
def load_model(model_path_text: str) -> tf.keras.Model:
    model_path = Path(model_path_text)
    patch_dense_quantization_config()
    custom_objects = get_model_custom_objects()

    try:
        return tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False,
        )
    except TypeError:
        return tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False,
        )


def prepare_image(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    image_array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


def predict_genre(model: tf.keras.Model, image: Image.Image) -> pd.DataFrame:
    model_input = prepare_image(image)
    probabilities = model.predict(model_input, verbose=0)[0]
    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            f"모델 출력({len(probabilities)})과 클래스 수({len(CLASS_NAMES)})가 일치하지 않습니다."
        )

    result = pd.DataFrame(
        {
            "genre": CLASS_NAMES,
            "label": [CLASS_LABELS_KO[name] for name in CLASS_NAMES],
            "probability": probabilities,
        }
    )
    return result.sort_values("probability", ascending=False).reset_index(drop=True)


def render_probability_chart(result: pd.DataFrame) -> None:
    chart_data = result.copy()
    chart_data["probability_percent"] = chart_data["probability"] * 100

    chart = (
        alt.Chart(chart_data)
        .mark_bar(color="#0F52BA", cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(
                "label:N",
                sort=None,
                title=None,
                axis=alt.Axis(labelAngle=0, labelColor="#f6ead3", tickColor="#c8a15a"),
            ),
            y=alt.Y(
                "probability_percent:Q",
                title="확률 (%)",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(
                    gridColor="rgba(246, 234, 211, 0.16)",
                    labelColor="#f6ead3",
                    tickColor="#c8a15a",
                    titleColor="#f6ead3",
                ),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="장르"),
                alt.Tooltip("genre:N", title="Class"),
                alt.Tooltip("probability_percent:Q", title="확률", format=".2f"),
            ],
        )
        .properties(background="transparent", height=280)
        .configure_axis(
            domainColor="rgba(200, 161, 90, 0.38)",
            labelFont="Segoe UI",
            titleFont="Segoe UI",
        )
        .configure_view(stroke=None)
    )
    st.altair_chart(chart, width="stretch", theme=None)


def render_result(result: pd.DataFrame) -> None:
    top = result.iloc[0]
    top_genre = str(top["genre"])
    top_label = str(top["label"])
    confidence = float(top["probability"])

    st.metric("예측 장르", top_label, f"{confidence * 100:.2f}%")
    st.caption(top_genre)
    st.write(CLASS_DESCRIPTIONS[top_genre])

    table = result.copy()
    table["probability"] = table["probability"].map(lambda value: f"{value * 100:.2f}%")
    table = table.rename(
        columns={
            "genre": "Class",
            "label": "장르",
            "probability": "확률",
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=248,
        row_height=34,
        column_config={
            "Class": st.column_config.TextColumn("Class", width="medium"),
            "장르": st.column_config.TextColumn("장르", width="small"),
            "확률": st.column_config.TextColumn("확률", width="small"),
        },
    )


def main() -> None:
    st.set_page_config(
        page_title="미술 사조 분류기",
        layout="wide",
    )
    apply_custom_css()

    st.title("미술 사조 분류기")

    try:
        model_path = find_model_path()
        model = load_model(str(model_path))
    except Exception as exc:
        st.error("모델을 불러오지 못했습니다.")
        st.exception(exc)
        st.stop()

    with st.sidebar:
        st.header("모델")
        st.write("ResNet-50")
        st.caption(str(model_path.relative_to(APP_DIR)))
        st.header("분류 장르")
        for class_name in CLASS_NAMES:
            st.write(f"{CLASS_LABELS_KO[class_name]} · `{class_name}`")

    uploaded_file = st.file_uploader(
        "이미지 업로드",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("분류할 회화 이미지를 업로드하세요.")
        return

    try:
        image = Image.open(uploaded_file)
    except Exception as exc:
        st.error("이미지 파일을 열 수 없습니다.")
        st.exception(exc)
        return

    image = ImageOps.exif_transpose(image).convert("RGB")

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.image(image, caption=uploaded_file.name, width="stretch")

    with right:
        with st.spinner("장르를 예측하는 중입니다..."):
            result = predict_genre(model, image)
        render_result(result)

    st.subheader("클래스별 예측 확률")
    render_probability_chart(result)


if __name__ == "__main__":
    main()
