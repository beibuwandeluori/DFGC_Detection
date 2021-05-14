import cv2
import dlib
import numpy as np


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../dlib_model/shape_predictor_68_face_landmarks.dat')


class Mask():
    """ Parent class for masks
        the output mask will be <mask_type>.mask
        channels: 1, 3 or 4:
                    1 - Returns a single channel mask
                    3 - Returns a 3 channel mask
                    4 - Returns the original image with the mask in the alpha channel """

    def __init__(self, landmarks, face, channels=4):
        # logger.info("Initializing %s: (face_shape: %s, channels: %s, landmarks: %s)",
                     # self.__class__.__name__, face.shape, channels, landmarks)
        self.landmarks = landmarks
        self.face = face
        self.channels = channels

        mask = self.build_mask()
        self.mask = self.merge_mask(mask)
        # logger.info("Initialized %s", self.__class__.__name__)

    def build_mask(self):
        """ Override to build the mask """
        raise NotImplementedError

    def merge_mask(self, mask):
        """ Return the mask in requested shape """
        # logger.info("mask_shape: %s", mask.shape)
        assert self.channels in (1, 3, 4), "Channels should be 1, 3 or 4"
        assert mask.shape[2] == 1 and mask.ndim == 3, "Input mask be 3 dimensions with 1 channel"

        if self.channels == 3:
            retval = np.tile(mask, 3)
        elif self.channels == 4:
            retval = np.concatenate((self.face, mask), -1)
        else:
            retval = mask

        # logger.info("Final mask shape: %s", retval.shape)
        return retval


class dfl_full(Mask):  # pylint: disable=invalid-name
    """ DFL facial mask """
    def build_mask(self):
        mask = np.zeros(self.face.shape[0:2] + (1, ), dtype=np.float32)

        nose_ridge = (self.landmarks[27:31], self.landmarks[33:34])
        jaw = (self.landmarks[0:17],
               self.landmarks[48:68],
               self.landmarks[0:1],
               self.landmarks[8:9],
               self.landmarks[16:17])
        eyes = (self.landmarks[17:27],
                self.landmarks[0:1],
                self.landmarks[27:28],
                self.landmarks[16:17],
                self.landmarks[33:34])
        parts = [jaw, nose_ridge, eyes]

        for item in parts:
            merged = np.concatenate(item)
            cv2.fillConvexPoly(mask, cv2.convexHull(merged), 255.)  # pylint: disable=no-member
        return mask
        

class extended(Mask):  # pylint: disable=invalid-name
    """ Extended mask
        Based on components mask. Attempts to extend the eyebrow points up the forehead
    """
    def build_mask(self):
        mask = np.zeros(self.face.shape[0:2] + (1, ), dtype=np.float32)

        landmarks = self.landmarks.copy()
        # mid points between the side of face and eye point
        ml_pnt = (landmarks[36] + landmarks[0]) // 2
        mr_pnt = (landmarks[16] + landmarks[45]) // 2

        # mid points between the mid points and eye
        ql_pnt = (landmarks[36] + ml_pnt) // 2
        qr_pnt = (landmarks[45] + mr_pnt) // 2

        # Top of the eye arrays
        bot_l = np.array((ql_pnt, landmarks[36], landmarks[37], landmarks[38], landmarks[39]))
        bot_r = np.array((landmarks[42], landmarks[43], landmarks[44], landmarks[45], qr_pnt))

        # Eyebrow arrays
        top_l = landmarks[17:22]
        top_r = landmarks[22:27]

        # Adjust eyebrow arrays
        landmarks[17:22] = top_l + ((top_l - bot_l) // 2)
        landmarks[22:27] = top_r + ((top_r - bot_r) // 2)

        r_jaw = (landmarks[0:9], landmarks[17:18])
        l_jaw = (landmarks[8:17], landmarks[26:27])
        r_cheek = (landmarks[17:20], landmarks[8:9])
        l_cheek = (landmarks[24:27], landmarks[8:9])
        nose_ridge = (landmarks[19:25], landmarks[8:9],)
        r_eye = (landmarks[17:22], landmarks[27:28], landmarks[31:36], landmarks[8:9])
        l_eye = (landmarks[22:27], landmarks[27:28], landmarks[31:36], landmarks[8:9])
        nose = (landmarks[27:31], landmarks[31:36])
        parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]

        for item in parts:
            merged = np.concatenate(item)
            cv2.fillConvexPoly(mask, cv2.convexHull(merged), 255.)  # pylint: disable=no-member
        return mask


def get_face_mask(face_img):
    # face_img in the range[0, 255], rgb image
    try:
        rect = detector(face_img)[0]
        sp = predictor(face_img, rect)
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])
        mask = extended(landmarks=landmarks.astype('int32'), face=face_img, channels=3).mask
        mask = mask/255.0
    except:
        mask = np.ones(face_img.shape)
    return mask



if __name__ == '__main__':
    image_path = 'fake1.png'
    face_img = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    mask = get_face_mask(face_img)
    print(mask.shape)
    mask = (mask*255).astype(np.uint8)
    cv2.imwrite('mask.png', mask)
    
    
    
    
    

