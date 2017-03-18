from net.common import *



def imshow(name, image, resize=1, is_rgb=False):

    if is_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    H,W,_ = image.shape
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))

def normalise(image, limit=255.0):
    image -= image.min()
    image *= (limit/image.max())


    return image