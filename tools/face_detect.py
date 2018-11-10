
import dlib

class FaceDetect(object):
    def __init__(self):
        self.face_detect = dlib.get_frontal_face_detector()

    def detect(self, img):
        win = dlib.image_window()
        img = dlib.load_rgb_image(img)
        det = self.face_detect(img)
        win.clear_overlay()
        win.set_image(img)
        win.add_overlay(det)
        win.set_title('{} faces found'.format(len(det)))
        dlib.hit_enter_to_continue()
