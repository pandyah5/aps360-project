import cv2


def get_bbox(image, threshold, vis=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if vis:
        cv2.imshow("gray", gray)

    canny_output = cv2.Canny(gray, threshold, threshold * 2)

    if vis:
        cv2.imshow("canny", canny_output)

    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours:" + str(len(contours)))

    for contour in contours:
        poly = cv2.approxPolyDP(contour, 3, True)
        x, y, w, h = cv2.boundingRect(poly)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)

    if vis:
        cv2.imshow("bounding box", image)
