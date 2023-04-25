import cv2
import numpy as np
import pickle


class ParkingLot:
    def __init__(self, pos_list_file):
        with open(pos_list_file, 'rb') as f:
            self.pos_list = pickle.load(f)
        self.frame = None

    def __repr__(self):
        return f"ParkingLot({self.pos_list_file})"

    def __str__(self):
        return f"ParkingLot with {len(self.pos_list)} parking spaces"

    def __len__(self):
        return len(self.pos_list)

    def check_parking_space(self, img):
        free_spaces = 0
        for pos in self.pos_list:
            p1, p2 = pos
            # Cropping the image to get only the parking space area
            img_crop = img[p1[1]:p2[1], p1[0]:p2[0]]
            count = cv2.countNonZero(img_crop)

            if count > 800:
                color = (0, 0, 255)

            else:
                free_spaces += 1
                color = (0, 255, 0)

            # Drawing a rectangle around the parking space and displaying the count of non-zero pixels inside it
            cv2.rectangle(self.frame, p1, p2, color, 2)
            cv2.putText(self.frame, str(count), (p1[0], p1[1] - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                        color, 1)

        # Displaying the total number of free parking spaces out of the total number of parking spaces
        cv2.putText(img, f'{free_spaces} / {len(self.pos_list)}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 0, 255), 3)

    def process_image(self, image_path):
        # Reading the image
        self.frame = cv2.imread(image_path)

        # Converting the image to grayscale
        gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        # Blurring the grayscale image using a Gaussian filter
        blurred_frame = cv2.GaussianBlur(gray_frame, (3, 3), 1)

        # Applying adaptive thresholding to the blurred image to binarize it
        threshold_frame = cv2.adaptiveThreshold(blurred_frame, 255,
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 25, 16)

        # Applying median filtering to the thresholded image to remove noise
        frame_median = cv2.medianBlur(threshold_frame, 5)

        # Dilating the filtered image to fill in gaps in the parking space boundaries
        kernel = np.ones((5, 5), np.uint8)
        dilated_frame = cv2.dilate(frame_median, kernel, iterations=1)

        # Checking parking space status
        self.check_parking_space(dilated_frame)
        # Displaying the image
        cv2.imshow('image', self.frame)
        cv2.waitKey(0)


if __name__ == '__main__':
    pos_list_file = 'park_positions'
    parking_lot = ParkingLot(pos_list_file)
    parking_lot.process_image('parking_lot.png')

