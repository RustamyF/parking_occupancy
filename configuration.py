import cv2
import pickle
import argparse


class BoxDrawer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.pos_list = self.positions()
        self.img = cv2.imread(self.image_path)

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw_box)

    @staticmethod
    def positions():
        try:
            with open('park_positions', 'rb') as f:
                pos_list = pickle.load(f)
        except Exception as e:
            print(f"An error occurred: {e}, starting a new postion list")
            pos_list = []
        return pos_list

    def draw_box(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            cv2.rectangle(self.img, self.start_point, self.end_point, (0, 255, 0), 2)
            cv2.imshow('image', self.img)
            self.pos_list.append((self.start_point, self.end_point))

        if event == cv2.EVENT_RBUTTONDOWN:
            for i, pos in enumerate(self.pos_list):
                p1, p2 = pos
                if p1[0] < x < p2[0] and p1[1] < y < p2[1]:
                    self.pos_list.pop(i)
                    self.img = cv2.imread(self.image_path)

        with open('park_positions', 'wb') as f:
            pickle.dump(self.pos_list, f)

    def start(self):
        while True:

            cv2.imshow('image', self.img)
            for pos in self.pos_list:
                cv2.rectangle(self.img, pos[0], pos[1], (255, 0, 255), 1)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', default='new_park.png', help='Path to the image')
    args = parser.parse_args()
    box_drawer = BoxDrawer(args.image)
    box_drawer.start()
