from rembg import remove
import numpy as np
import cv2
import os, shutil

def preprocess_image(image):
    image = remove(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), cv2.BORDER_DEFAULT)
    return blur

def outline(image):
    edged = cv2.Canny(image, 30, 80)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def extract_features(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(template_descriptors, test_descriptors):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(template_descriptors, test_descriptors)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def calculate_homography(template_keypoints, test_keypoints, matches):
    src_pts = np.float32([template_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([test_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    return homography

def calculate_rotation(homography):
    rotation_rad = np.arctan2(homography[0, 1], homography[0, 0])
    rotation_deg = np.degrees(rotation_rad)
    return rotation_deg

def main():
    try:
        shutil.rmtree('output_images')
    except Exception as e:
        print(e)
    finally:
        os.makedirs('output_images')

    test_images = [f for f in os.listdir('test_images')]
    template_images = [f for f in os.listdir('template_images')]

    for template_num, t_image in enumerate(template_images):
        os.makedirs(r"output_images\template_" + str(template_num))

        template_image = cv2.imread(r"template_images\\" + t_image)
        template_gray = preprocess_image(template_image)
        cv2.imwrite(r"output_images\template_" + str(template_num) + r"\template.jpg", template_image)

        for test_num, image in enumerate(test_images):
            test_image = cv2.imread(r"test_images\\" + image)
            test_gray = preprocess_image(test_image)

            template_keypoints, template_descriptors = extract_features(template_gray)
            test_keypoints, test_descriptors = extract_features(test_gray)

            matches = match_features(template_descriptors, test_descriptors)

            homography = calculate_homography(template_keypoints, test_keypoints, matches)

            rotation_angle = calculate_rotation(homography)

            contours = outline(test_gray)
            
            cv2.drawContours(test_image, contours, -1, (0, 255, 0), 2)
            cv2.putText(test_image, f"Rotation: {rotation_angle:.2f} degrees", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imwrite(r"output_images\template_" + str(template_num) + f"\\{test_num}.jpg", test_image)

if __name__ == "__main__":
    main()
