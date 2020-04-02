import cv2
import glob
import math
import tifffile
import numpy as np
from random import randint

# The Model is responsible for all the application logic
class Model():
    def __init__(self):
        self.image_array = []
        self.circles = []
        self.blobs = []
        self.blob_diameters = []

    # loading and processing of all images in a specified directory
    # TODO -- prevent error when directory is invalid or empyt --
    def load_sampleset(self, path_to_folder):
        for file in glob.glob(path_to_folder):
            # load image and write to temporary file
            raw_image = tifffile.imread(file)
            raw_image <<= 4
            tifffile.imwrite('temp.tif', raw_image)
            image = cv2.imread("temp.tif", cv2.IMREAD_GRAYSCALE)
            # removal of periodic interference
            image = self.remove_periodic_interference(image)
            image = self.aplly_mask(image)
            # add processed image to sampleset
            self.image_array.append(image)

    # removal of periodic stripe pattern in the lower half of the provided image
    # using fourrier analysis
    def remove_periodic_interference(self, image):
        height, width = image.shape
        # min and max values of image
        min_v = np.amin(image)
        max_v = np.amax(image)
        mean_v = int(np.mean(image))
        # adding padding to image
        padded_height = math.ceil(math.log2(height))
        padded_height = int(math.pow(2, padded_height))
        padded_width = math.ceil(math.log2(width))
        padded_width = int(math.pow(2, padded_width))
        padded_image = np.full((padded_height, padded_width), mean_v, dtype=np.uint8)
        padded_image[0:height, 0:width] = image
        # convert image to float and save as complex output
        dft = cv2.dft(np.float32(padded_image), flags=cv2.DFT_COMPLEX_OUTPUT)
        # shift of origin from upper left corner to center of image
        dft_shifted = np.fft.fftshift(dft)
        # magnitude and phase images
        mag, phase = cv2.cartToPolar(dft_shifted[:, :, 0], dft_shifted[:, :, 1])
        # extract spectrum
        spectrum = np.log(mag) / 20
        min, max = np.amin(spectrum, (0, 1)), np.amax(spectrum, (0, 1))
        # threshold spectrum to find bright spots
        thresh_spec = (255 * spectrum).astype(np.uint8)
        thresh_spec = cv2.threshold(thresh_spec, 155, 255, cv2.THRESH_BINARY)[1]
        # cover center rows of thresh with black
        yc = padded_height // 2
        cv2.line(thresh_spec, (0, yc), (padded_width - 1, yc), 0, 5)
        # get y coordinates bright spots
        bright_spots = np.column_stack(np.nonzero(thresh_spec))
        # mask
        mask = thresh_spec.copy()
        for b in bright_spots:
            y = b[0]
            cv2.line(mask, (0, y), (padded_width - 1, y), 255, 5)
        # apply mask to magnitude
        mag[mask != 0] = 0
        # convert new magnitude and old phase into cartesian real and imaginary components
        real, imag = cv2.polarToCart(mag, phase)
        # combine cart comp into one compl image
        back = cv2.merge([real, imag])
        # shift origin from center to upper left corner
        back_ishift = np.fft.ifftshift(back)
        # do idft saving as complex output
        img_back = cv2.idft(back_ishift)
        # combine complex components into original image again
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        # crop to original size
        img_back = img_back[0:height, 0:width]
        # re-normalize to 8-bits in range of original
        min, max = np.amin(img_back, (0, 1)), np.amax(img_back, (0, 1))
        notched = cv2.normalize(img_back, None, alpha=min_v, beta=max_v,
                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        return notched

    # application off mask to isolate the usable 'droples' from each image
    # TODO-- optimization --
    # TODO-- offset nog aanpassen--
    def aplly_mask(self, image):
        th, threshed = cv2.threshold(image, 116, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C | cv2.ADAPTIVE_THRESH_MEAN_C)

        # Find the min-area contour
        cnts = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(1, 1))[-2]
        cnts = sorted(cnts, key=cv2.contourArea)

        # Manual scaling of contours
        for c in cnts:
            M = cv2.moments(c)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            scale = 1.5

            cnt_norm = c - [cx, cy]
            cnt_scaled = cnt_norm * scale
            cnt_scaled = cnt_scaled + [cx, cy]
            cnt_scaled = cnt_scaled.astype(np.int32)

        # Mask application
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.drawContours(mask, cnts, -1, 255, -1)
        dst = cv2.bitwise_and(image, image, mask=mask)
        dst = cv2.bitwise_not(dst, dst)

        #cv2.imshow("no mask", image)
        #cv2.imshow("masked", dst)
        #cv2.waitKey(0)

        return dst

    # detection of circles by the HoughCircles algorithm
    # //obsolete//
    def detect_circles(self, image_set):
        for image in image_set:
            rows = image.shape[0]

            circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, rows / 16,
                                       param1=25, param2=15,
                                       minRadius=0, maxRadius=12)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    center = (i[0], i[1])
                    # circle center
                    cv2.circle(image, center, 1, (0, 100, 100), 3)
                    # circle outline
                    radius = i[2]
                    self.circles.append(radius * 4.98)
                    cv2.circle(image, center, radius, (255, 0, 0), 3)

        return self.circles


    # detection of blobs, favourable as the 'droplets' don't have
    # to be perfect circles for the algorithm to detect them
    # circularity can be specified
    def detect_blobs(self, image_set):
        parameters = cv2.SimpleBlobDetector_Params()

        parameters.minThreshold = 115
        parameters.maxThreshold = 255

        parameters.minDistBetweenBlobs = 25

        parameters.filterByArea = True
        parameters.minArea = 4
        parameters.maxArea = 300

        parameters.filterByCircularity = True
        parameters.minCircularity = 0.75

        parameters.filterByConvexity = True
        parameters.minConvexity = 0.75

        parameters.filterByInertia = False
        parameters.minInertiaRatio = 0.5

        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            detector = cv2.SimpleBlobDetector(parameters)
        else:
            detector = cv2.SimpleBlobDetector_create(parameters)

        for image in image_set:
            keypoints = detector.detect(image)
            image_with_keypoints = cv2.drawKeypoints(image, keypoints, (0, 0, 255),
                                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            #cv2.imshow("keypoints", image_with_keypoints)
            #cv2.waitKey(0)

            if keypoints is not None:
                self.blobs.append(keypoints)
                for keypoint in keypoints:
                    diameter = keypoint.size
                    self.blob_diameters.append(diameter * 4.98)

    # returns the blob diameters
    # TODO-- prevent nullpointer --
    def get_blob_diameters(self):
        return self.blob_diameters

    def blob_tracking(self, image_set):

        #1 Single object tracker
        tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

        def create_tracker(type):
            if type == 'BOOSTING':
                tracker = cv2.TrackerBoosting_create()
            elif type == 'MIL':
                tracker = cv2.TrackerMIL_create()
            elif type == 'KCF':
                tracker = cv2.TrackerKCF_create()
            elif type == 'TLD':
                tracker = cv2.TrackerTLD_create()
            elif type == 'MEDIANFLOW':
                tracker = cv2.TrackerMedianFlow_create()
            elif type == 'GOTURN':
                tracker = cv2.TrackerGOTURN_create()
            elif type == 'MOSSE':
                tracker = cv2.TrackerMOSSE_create()
            elif type == "CSRT":
                tracker = cv2.TrackerCSRT_create()
            else:
                tracker = None
                print("Provide correct tracker name.")

            return tracker

        #2 read first frame
        if image_set is None:
            print("image set is empty")
        else:
            frame = image_set[0]

        #3 locate objects/ select bounding boxes
        bboxes = []
        colors = []

        while True:
            bbox = cv2.selectROI('Multitracker', frame, False)
            bboxes.append(bbox)
            colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))

            print("bbox added")

            k = cv2.waitKey(0) & 0xFF
            if k == ord('q'):
                break

            print('Selected bounding boxes {}'.format(bboxes))

        #4 Initialize multitracker
        tracker_type = tracker_types[3]
        multiTracker = cv2.MultiTracker_create()

        for bbox in bboxes:
            multiTracker.add(create_tracker(tracker_type), frame, bbox)

        #5 Update & Results
        for f in image_set:
            success, boxes = multiTracker.update(frame)

            for i, newbox in enumerate(boxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

            cv2.imshow('Multitracker', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break #exit on ESC pressed