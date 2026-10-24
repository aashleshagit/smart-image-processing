import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO

st.set_page_config(
    page_title="Smart Image Processing",
    page_icon="📷",
    layout="wide"
)

st.title("📷 Smart Image Processing & Background Extraction")
st.write(
    "Upload an image and explore Computer Vision techniques "
    "using Python and OpenCV."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

second_file = None

if uploaded_file is not None:
    second_file = st.file_uploader(
        "Upload a second image for feature matching",
        type=["jpg", "jpeg", "png"],
        key="second_image"
    )

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    operation = st.selectbox(
        "Choose an image processing operation",
        
[
    "Original Image",
    "Grayscale",
    "Resize",
    "Crop Image",
    "Thresholding",
    "Edge Detection",
    "Contour Detection",
    "Image Smoothing",
    "Histogram Equalization",
    "Color Space Conversion",
    "Image Annotation",
    "Blob Detection",
    "Image Gradients",
    "Hough Line Detection",
    "Fourier Transform",
    "ORB Feature Detection",
    "ORB Feature Matching",
    "GrabCut Background Extraction"

]

    )

    result = image_bgr.copy()

    if operation == "Grayscale":
        result = cv2.cvtColor(
            image_bgr, cv2.COLOR_BGR2GRAY
        )
        
    
    elif operation == "Crop Image":
        h, w = image_bgr.shape[:2]

        x1 = st.slider("Left coordinate", 0, w - 1, 0)
        x2 = st.slider("Right coordinate", 1, w, w)

        y1 = st.slider("Top coordinate", 0, h - 1, 0)
        y2 = st.slider("Bottom coordinate", 1, h, h)

        if x2 > x1 and y2 > y1:
            result = image_bgr[y1:y2, x1:x2]
        else:
            st.warning("Choose valid crop coordinates.")

    elif operation == "Color Space Conversion":
        color_space = st.selectbox(
            "Select color space",
            ["HSV", "LAB", "YCrCb"]
        )

        if color_space == "HSV":
            result = cv2.cvtColor(
                image_bgr, cv2.COLOR_BGR2HSV
            )
        elif color_space == "LAB":
            result = cv2.cvtColor(
                image_bgr, cv2.COLOR_BGR2LAB
            )
        else:
            result = cv2.cvtColor(
                image_bgr, cv2.COLOR_BGR2YCrCb
            )

        # Convert for display so the channels are visible
        result = cv2.cvtColor(
            result, cv2.COLOR_BGR2RGB
        )

    elif operation == "Image Annotation":
        result = image_bgr.copy()

        h, w = result.shape[:2]

        cv2.rectangle(
            result, (10, 10),
            (min(w - 1, 200), min(h - 1, 100)),
            (0, 255, 0), 3
        )

        cv2.circle(
            result,
            (w // 2, h // 2),
            max(1, min(w, h) // 8),
            (0, 0, 255), 3
        )

        cv2.putText(
            result, "Computer Vision",
            (10, min(h - 10, 140)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 0, 0), 2
        )

    elif operation == "Blob Detection":
        gray = cv2.cvtColor(
            image_bgr, cv2.COLOR_BGR2GRAY
        )

        _, binary = cv2.threshold(
            gray, 127, 255, cv2.THRESH_BINARY
        )

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        result = image_bgr.copy()
        blob_count = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            if area > 100:
                x, y, bw, bh = cv2.boundingRect(contour)

                cv2.rectangle(
                    result, (x, y), (x + bw, y + bh),
                    (0, 255, 0), 2
                )

                blob_count += 1

        st.write("Blobs detected:", blob_count)


    elif operation == "Resize":
        scale = st.slider(
            "Resize percentage", 10, 200, 50
        )
        width = max(1, int(image_bgr.shape[1] * scale / 100))
        height = max(1, int(image_bgr.shape[0] * scale / 100))
        result = cv2.resize(image_bgr, (width, height))

    elif operation == "Thresholding":
        gray = cv2.cvtColor(
            image_bgr, cv2.COLOR_BGR2GRAY
        )
        threshold = st.slider(
            "Threshold value", 0, 255, 127
        )
        _, result = cv2.threshold(
            gray, threshold, 255, cv2.THRESH_BINARY
        )

    elif operation == "Edge Detection":
        gray = cv2.cvtColor(
            image_bgr, cv2.COLOR_BGR2GRAY
        )
        result = cv2.Canny(gray, 100, 200)

    elif operation == "Contour Detection":
        gray = cv2.cvtColor(
            image_bgr, cv2.COLOR_BGR2GRAY
        )
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        result = image_bgr.copy()
        cv2.drawContours(
            result, contours, -1, (0, 255, 0), 2
        )
        st.write("Contours detected:", len(contours))
        
    
    elif operation == "Image Gradients":
        gray = cv2.cvtColor(
            image_bgr, cv2.COLOR_BGR2GRAY
        )

        grad_x = cv2.Sobel(
            gray, cv2.CV_64F, 1, 0, ksize=3
        )
        grad_y = cv2.Sobel(
            gray, cv2.CV_64F, 0, 1, ksize=3
        )

        magnitude = cv2.magnitude(grad_x, grad_y)
        result = cv2.convertScaleAbs(magnitude)

    elif operation == "Hough Line Detection":
        gray = cv2.cvtColor(
            image_bgr, cv2.COLOR_BGR2GRAY
        )
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=50,
            minLineLength=40,
            maxLineGap=10
        )

        result = image_bgr.copy()
        line_count = 0

        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = map(int, line.reshape(-1))
                cv2.line(
                    result,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )
                line_count += 1


        st.write("Lines detected:", line_count)

    elif operation == "Fourier Transform":
        gray = cv2.cvtColor(
            image_bgr, cv2.COLOR_BGR2GRAY
        )

        # Fourier transform
        f = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f)

        # Convert frequency magnitudes into a visible image
        magnitude = 20 * np.log(
            np.abs(f_shift) + 1
        )

        result = cv2.normalize(
            magnitude, None, 0, 255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

    elif operation == "ORB Feature Detection":
        gray = cv2.cvtColor(
            image_bgr, cv2.COLOR_BGR2GRAY
        )

        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(
            gray, None
        )

        result = cv2.drawKeypoints(
            image_bgr,
            keypoints,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        st.write("ORB keypoints detected:", len(keypoints))


    elif operation == "Image Smoothing":
        result = cv2.GaussianBlur(
            image_bgr, (7, 7), 0
        )

    elif operation == "Histogram Equalization":
        gray = cv2.cvtColor(
            image_bgr, cv2.COLOR_BGR2GRAY
        )
        result = cv2.equalizeHist(gray)
        
    
    elif operation == "ORB Feature Matching":
        if second_file is None:
            st.info(
                "Upload a second image to compare features."
            )
            result = image_bgr.copy()

        else:
            second_image = Image.open(
                second_file
            ).convert("RGB")

            second_np = np.array(second_image)
            second_bgr = cv2.cvtColor(
                second_np, cv2.COLOR_RGB2BGR
            )

            gray1 = cv2.cvtColor(
                image_bgr, cv2.COLOR_BGR2GRAY
            )
            gray2 = cv2.cvtColor(
                second_bgr, cv2.COLOR_BGR2GRAY
            )

            orb = cv2.ORB_create(nfeatures=1000)

            kp1, des1 = orb.detectAndCompute(
                gray1, None
            )
            kp2, des2 = orb.detectAndCompute(
                gray2, None
            )

            result = image_bgr.copy()

            if des1 is not None and des2 is not None:
                matcher = cv2.BFMatcher(
                    cv2.NORM_HAMMING,
                    crossCheck=True
                )

                matches = matcher.match(des1, des2)
                matches = sorted(
                    matches, key=lambda match: match.distance
                )

                matched_image = cv2.drawMatches(
                    image_bgr, kp1,
                    second_bgr, kp2,
                    matches[:50], None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )

                result = matched_image
                st.write("Image 1 keypoints:", len(kp1))
                st.write("Image 2 keypoints:", len(kp2))
                st.write("Feature matches:", len(matches))
            else:
                st.warning(
                    "Not enough features found. Try clearer images."
                )


    elif operation == "GrabCut Background Extraction":
        st.write("Extracting foreground object...")

        h, w = image_bgr.shape[:2]

        rect = (
            int(w * 0.05),
            int(h * 0.05),
            int(w * 0.90),
            int(h * 0.90)
        )

        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(
                image_bgr,
                mask,
                rect,
                bgd_model,
                fgd_model,
                5,
                cv2.GC_INIT_WITH_RECT
            )

            foreground_mask = np.where(
                (mask == cv2.GC_FGD) |
                (mask == cv2.GC_PR_FGD),
                255,
                0
            ).astype(np.uint8)

        
            rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
            
            rgba[:, :, 3] = foreground_mask
            
            result = rgba
    
 

            st.success("Background extraction completed!")

        except cv2.error as e:
            st.error(f"GrabCut failed: {e}")


    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image_np, use_container_width=True)

    with col2:
        st.subheader("Processed Image")

        if len(result.shape) == 2:
            st.image(
                result, channels="GRAY",
                use_container_width=True
            )
            
        elif len(result.shape) == 3 and result.shape[2] == 4:
            result_rgba = cv2.cvtColor(
                result, cv2.COLOR_BGRA2RGBA
            )
            st.image(
                result_rgba,
                use_container_width=True
            )

        else:
            result_rgb = cv2.cvtColor(
                result, cv2.COLOR_BGR2RGB
            )
            st.image(
                result_rgb,
                use_container_width=True
            )

            st.image(
                result_rgb,
                use_container_width=True
            )

    
    if len(result.shape) == 2:
        output_image = Image.fromarray(result)
    elif len(result.shape) == 3 and result.shape[2] == 4:
        output_image = Image.fromarray(
            cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA)
        )
    else:
        output_image = Image.fromarray(
            cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        )


    buffer = BytesIO()
    output_image.save(buffer, format="PNG")

    st.download_button(
        "Download Processed Image",
        data=buffer.getvalue(),
        file_name="processed_image.png",
        mime="image/png"
    )

else:
    st.info("Upload an image to begin processing.")
