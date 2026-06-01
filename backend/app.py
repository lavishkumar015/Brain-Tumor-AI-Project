
from flask import Flask, render_template, request, send_file
import os
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
import cv2
from datetime import datetime
import gdown
import os

# PDF
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# =====================================================
# FLASK APP
# =====================================================

app = Flask(__name__)

# =====================================================
# FOLDERS
# =====================================================

UPLOAD_FOLDER = "static/uploads"
HEATMAP_FOLDER = "static/heatmaps"
REPORT_FOLDER = "static/reports"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# =====================================================
# LOAD MODEL
# =====================================================
MODEL_PATH = "brain_tumor_model.keras"

def download_model():

    if not os.path.exists(MODEL_PATH):

        print("Downloading model...")

        FILE_ID = "1-98jTf3tvhzNB1l07twXS1HH0A2rULI9"

        gdown.download(
    id=FILE_ID,
    output=MODEL_PATH,
    fuzzy=True,
     quiet=False
    )
        print("Model Downloaded Successfully!")

download_model()

model = load_model(MODEL_PATH)

# =====================================================
# CLASSES
# =====================================================

classes = [

    "Glioma Tumor",

    "Meningioma Tumor",

    "No Tumor",

    "Pituitary Tumor"

]
# Railway redeploy test
# =====================================================
# GRADCAM FUNCTION
# =====================================================

def generate_gradcam(img_array, model, image_path):

    try:

        # Prediction
        preds = model.predict(img_array)

        # Get VGG16 layer
        base_model = model.get_layer(
            "vgg16"
        )

        # Feature extractor
        feature_model = tf.keras.models.Model(

            inputs=base_model.input,

            outputs=base_model.get_layer(
                "block5_conv3"
            ).output

        )

        # Feature maps
        conv_output = feature_model.predict(
            img_array
        )

        # Heatmap
        heatmap = np.mean(
            conv_output[0],
            axis=-1
        )

        heatmap = np.maximum(
            heatmap,
            0
        )

        if np.max(heatmap) != 0:

            heatmap /= np.max(heatmap)

        # Original image
        img = cv2.imread(image_path)

        heatmap = cv2.resize(

            heatmap,

            (
                img.shape[1],
                img.shape[0]
            )

        )

        heatmap = np.uint8(
            255 * heatmap
        )

        # Apply color
        heatmap = cv2.applyColorMap(

            heatmap,

            cv2.COLORMAP_JET

        )

        # Overlay
        superimposed_img = cv2.addWeighted(

            img,
            0.6,
            heatmap,
            0.4,
            0

        )

        # Save
        heatmap_filename = (

            "heatmap_" +

            os.path.basename(image_path)

        )

        heatmap_path = os.path.join(

            HEATMAP_FOLDER,

            heatmap_filename

        )

        cv2.imwrite(

            heatmap_path,

            superimposed_img

        )

        return heatmap_path

    except Exception as e:

        print("GradCAM Error:", e)

        return image_path

# =====================================================
# PDF GENERATION
# =====================================================


def generate_pdf(

    prediction,
    confidence,
    image_path,
    heatmap_path,

    patient_name,
    age,
    gender,

    doctor_name,
    hospital_name,

    scan_id,
    scan_date,

    symptoms,
    notes

):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    report_id = (
        "BTAI-" + timestamp
    )

    pdf_filename = (
        f"report_{timestamp}.pdf"
    )

    pdf_path = os.path.join(
        REPORT_FOLDER,
        pdf_filename
    )

    # =====================================
    # TUMOR INFO
    # =====================================

    tumor_info = {

        "Glioma Tumor":
        """
        Glioma originates from glial cells and
        may affect brain function depending on
        location and severity.
        """,

        "Meningioma Tumor":
        """
        Meningioma develops in the protective
        membranes surrounding the brain.
        Usually slow growing.
        """,

        "Pituitary Tumor":
        """
        Pituitary tumor occurs in the pituitary
        gland and may influence hormone levels.
        """,

        "No Tumor":
        """
        No significant tumor-related abnormality
        detected by the AI model.
        """

    }

    # =====================================
    # RISK LEVEL
    # =====================================

    if confidence >= 90:

        risk = "HIGH"

    elif confidence >= 70:

        risk = "MODERATE"

    else:

        risk = "LOW"

    # =====================================
    # RECOMMENDATION
    # =====================================

    if confidence >= 90:

        recommendation = """
        Immediate consultation with a neurologist
        and further radiological evaluation
        is recommended.
        """

    elif confidence >= 70:

        recommendation = """
        Follow-up MRI and specialist
        consultation recommended.
        """

    else:

        recommendation = """
        Additional scans may be required
        for accurate diagnosis.
        """

    # =====================================
    # PDF
    # =====================================

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter
    )

    styles = getSampleStyleSheet()

    elements = []

    # =====================================
    # TITLE
    # =====================================

    elements.append(

        Paragraph(

            "BRAIN TUMOR AI DIAGNOSTIC REPORT",

            styles["Title"]

        )

    )

    elements.append(
        Spacer(1, 15)
    )

    # =====================================
    # REPORT INFO
    # =====================================

    elements.append(

        Paragraph(

            f"<b>Report ID:</b> {report_id}",

            styles["BodyText"]

        )

    )

    elements.append(

        Paragraph(

            f"<b>Generated:</b> {datetime.now()}",

            styles["BodyText"]

        )

    )

    elements.append(
        Spacer(1, 15)
    )

    # =====================================
    # HOSPITAL INFO
    # =====================================

    elements.append(

        Paragraph(

            "<b>Hospital Information</b>",

            styles["Heading2"]

        )

    )

    elements.append(

        Paragraph(

            f"Hospital Name: {hospital_name}",

            styles["BodyText"]

        )

    )

    elements.append(

        Paragraph(

            f"Doctor Name: {doctor_name}",

            styles["BodyText"]

        )

    )

    elements.append(
        Spacer(1, 15)
    )

    # =====================================
    # PATIENT INFO
    # =====================================

    elements.append(

        Paragraph(

            "<b>Patient Details</b>",

            styles["Heading2"]

        )

    )

    elements.append(

        Paragraph(

            f"Patient Name: {patient_name}",

            styles["BodyText"]

        )

    )

    elements.append(

        Paragraph(

            f"Age: {age}",

            styles["BodyText"]

        )

    )

    elements.append(

        Paragraph(

            f"Gender: {gender}",

            styles["BodyText"]

        )

    )

    elements.append(

        Paragraph(

            f"Scan ID: {scan_id}",

            styles["BodyText"]

        )

    )

    elements.append(

        Paragraph(

            f"Scan Date: {scan_date}",

            styles["BodyText"]

        )

    )

    elements.append(
        Spacer(1, 15)
    )

    # =====================================
    # SYMPTOMS
    # =====================================

    elements.append(

        Paragraph(

            "<b>Symptoms</b>",

            styles["Heading2"]

        )

    )

    elements.append(

        Paragraph(

            symptoms,

            styles["BodyText"]

        )

    )

    elements.append(
        Spacer(1, 15)
    )

    # =====================================
    # AI RESULT
    # =====================================

    elements.append(

        Paragraph(

            "<b>AI Diagnosis Result</b>",

            styles["Heading2"]

        )

    )

    elements.append(

        Paragraph(

            f"Prediction: {prediction}",

            styles["BodyText"]

        )

    )

    elements.append(

        Paragraph(

            f"Confidence Score: {confidence}%",

            styles["BodyText"]

        )

    )

    elements.append(

        Paragraph(

            f"Risk Level: {risk}",

            styles["BodyText"]

        )

    )

    elements.append(
        Spacer(1, 15)
    )

    # =====================================
    # TUMOR DESCRIPTION
    # =====================================

    elements.append(

        Paragraph(

            "<b>Tumor Description</b>",

            styles["Heading2"]

        )

    )

    elements.append(

        Paragraph(

            tumor_info[prediction],

            styles["BodyText"]

        )

    )

    elements.append(
        Spacer(1, 15)
    )

    # =====================================
    # MRI IMAGE
    # =====================================

    elements.append(

        Paragraph(

            "<b>Original MRI Scan</b>",

            styles["Heading2"]

        )

    )

    mri_img = PDFImage(image_path)

    mri_img.width = 250
    mri_img.height = 250

    elements.append(mri_img)

    elements.append(
        Spacer(1, 15)
    )

    # =====================================
    # HEATMAP IMAGE
    # =====================================

    elements.append(

        Paragraph(

            "<b>AI Heatmap Analysis</b>",

            styles["Heading2"]

        )

    )

    heat_img = PDFImage(
        heatmap_path
    )

    heat_img.width = 250
    heat_img.height = 250

    elements.append(
        heat_img
    )

    elements.append(
        Spacer(1, 15)
    )

    # =====================================
    # CLINICAL INTERPRETATION
    # =====================================

    elements.append(

        Paragraph(

            "<b>Clinical Interpretation</b>",

            styles["Heading2"]

        )

    )

    elements.append(

        Paragraph(

            f"""
            The MRI scan was analyzed
            using a VGG16 Deep Learning model.

            The system detected
            {prediction}

            with confidence score
            of {confidence}%.
            """,

            styles["BodyText"]

        )

    )

    elements.append(
        Spacer(1, 15)
    )

    # =====================================
    # RECOMMENDATION
    # =====================================

    elements.append(

        Paragraph(

            "<b>Recommendations</b>",

            styles["Heading2"]

        )

    )

    elements.append(

        Paragraph(

            recommendation,

            styles["BodyText"]

        )

    )

    elements.append(
        Spacer(1, 15)
    )

    # =====================================
    # DOCTOR NOTES
    # =====================================

    elements.append(

        Paragraph(

            "<b>Doctor Notes</b>",

            styles["Heading2"]

        )

    )

    elements.append(

        Paragraph(

            notes,

            styles["BodyText"]

        )

    )

    elements.append(
        Spacer(1, 30)
    )

    # =====================================
    # SIGNATURE
    # =====================================

    elements.append(

        Paragraph(

            """
            ______________________________

            <br/><br/>

            Doctor Signature

            """,

            styles["BodyText"]

        )

    )

    elements.append(

        Paragraph(

            f"Dr. {doctor_name}",

            styles["BodyText"]

        )

    )

    elements.append(
        Spacer(1, 20)
    )

    # =====================================
    # DISCLAIMER
    # =====================================

    elements.append(

        Paragraph(

            """
            This report was generated by
            Brain Tumor AI Diagnostic System.

            The result should be reviewed
            by a qualified medical
            professional before final
            diagnosis.
            """,

            styles["Italic"]

        )

    )

    doc.build(elements)

    return pdf_filename

# =====================================================
# HOME
# =====================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )

# =====================================================
# ABOUT
# =====================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )

# =====================================================
# FEATURES
# =====================================================

@app.route("/features")
def features():

    return render_template(
        "features.html"
    )

# =====================================================
# CONTACT
# =====================================================

@app.route("/contact")
def contact():

    return render_template(
        "contact.html"
    )

# =====================================================
# DOWNLOAD REPORT
# =====================================================

@app.route("/download/<filename>")
def download_file(filename):

    path = os.path.join(

        REPORT_FOLDER,
        filename

    )

    return send_file(

        path,
        as_attachment=True

    )

# =====================================================
# PREDICT
# =====================================================

@app.route(
    "/predict",
    methods=["POST"]
)

def predict():

    # No image
    if "image" not in request.files:

        return "No image uploaded"

    file = request.files["image"]

    # Empty filename
    if file.filename == "":

        return "No selected image"

    # =================================================
    # PATIENT DETAILS
    # =================================================

    patient_name = request.form["patient_name"]

    age = request.form["age"]

    gender = request.form["gender"]

    doctor_name = request.form["doctor_name"]

    hospital_name = request.form["hospital_name"]

    scan_id = request.form["scan_id"]

    scan_date = request.form["scan_date"]

    symptoms = request.form["symptoms"]

    notes = request.form["notes"]

    # =================================================
    # SAVE IMAGE
    # =================================================

    file_path = os.path.join(

        app.config["UPLOAD_FOLDER"],
        file.filename

    )

    file.save(file_path)

    # =================================================
    # IMAGE PROCESSING
    # =================================================

    image = Image.open(file_path)

    image = image.convert("RGB")

    image = image.resize((224, 224))

    image_array = np.array(image)

    image_array = image_array / 255.0

    image_array = np.expand_dims(

        image_array,
        axis=0

    )

    # =================================================
    # PREDICTION
    # =================================================

    prediction = model.predict(
        image_array
    )

    predicted_index = np.argmax(
        prediction
    )

    predicted_class = classes[
        predicted_index
    ]

    confidence = round(

        np.max(prediction) * 100,
        2

    )

    # =================================================
    # HEATMAP
    # =================================================

    heatmap_path = generate_gradcam(

        image_array,
        model,
        file_path

    )

    # =================================================
    # PDF
    # =================================================

    pdf_filename = generate_pdf(

    predicted_class,
    confidence,

    file_path,
    heatmap_path,

    patient_name,
    age,
    gender,

    doctor_name,
    hospital_name,

    scan_id,
    scan_date,

    symptoms,
    notes

)

    # =================================================
    # RETURN RESULT
    # =================================================

    return render_template(

        "index.html",

        prediction=predicted_class,

        confidence=confidence,

        image_path=file_path,

        heatmap_path=heatmap_path,

        pdf_filename=pdf_filename

    )

# =====================================================
# RUN APP
# =====================================================

if __name__ == "__main__":

    app.run(debug=True)
