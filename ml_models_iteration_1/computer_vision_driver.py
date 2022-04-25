import tensorflow as tf
import numpy as np
from yolov3_tf2.models import YoloV3
from yolov3_tf2.utils import load_darknet_weights
import os
import cv2
import time
from seaborn import color_palette
from PIL import Image, ImageDraw, ImageFont


def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train

def draw_outputs(img, outputs, class_names):
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font='futur.ttf',
                              size=(img.size[0] + img.size[1]) // 100)
    for i in range(nums):
        color = colors[int(classes[i])]
        x1y1 = ((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = ((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        thickness = (img.size[0] + img.size[1]) // 200
        x0, y0 = x1y1[0], x1y1[1]
        for t in np.linspace(0, 1, thickness):
            x1y1[0], x1y1[1] = x1y1[0] - t, x1y1[1] - t
            x2y2[0], x2y2[1] = x2y2[0] - t, x2y2[1] - t
            draw.rectangle([x1y1[0], x1y1[1], x2y2[0], x2y2[1]], outline=tuple(color))
        confidence = '{:.2f}%'.format(objectness[i]*100)
        text = '{} {}'.format(class_names[int(classes[i])], confidence)
        print(text)
        text_size = draw.textsize(text, font=font)
        draw.rectangle([x0, y0 - text_size[1], x0 + text_size[0], y0],
                        fill=tuple(color))
        draw.text((x0, y0 - text_size[1]), text, fill='black',
                              font=font)
    rgb_img = img.convert('RGB')
    img_np = np.asarray(rgb_img)
    img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    return img

if __name__ == '__main__':

    #initialize model
    #physical_devices = tf.config.experimental.list_physical_devices('GPU')

    #if len(physical_devices) > 0:
    #    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    yolo = YoloV3()
    #yolo.summary()
    load_darknet_weights(yolo, 'yolov3.weights')
    #img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    #output = yolo(img)
    print('sanity check passed')
    yolo.save_weights('yolov3.tf')
    yolo.load_weights('yolov3.tf')

    class_names = [c.strip() for c in open('coco.names').readlines()]

    #load video
    video_path = os.path.join(os.getcwd(),'video.mp4')
    vid = cv2.VideoCapture(video_path)

    fps = 0.0
    empty_frame_counter = 0
    frame_counter = 0
    output_file = True

    if output_file:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter('processed_video.mp4', codec, fps, (width, height))
    

    #process frame by frame
    while True:
        _,img = vid.read()
            
        if img is None:
            time.sleep(0.1)
            empty_frame_counter+=1
            if empty_frame_counter < 3:
                continue
            else: 
                break
        
        yolo = YoloV3()
        #yolo.summary()
        #load_darknet_weights(yolo, 'yolov3.weights')
        #img = np.random.random((1, 320, 320, 3)).astype(np.float32)
        #output = yolo(img)
        #print('sanity check passed')
        #yolo.save_weights('yolov3.tf')
        yolo.load_weights('yolov3.tf')


        img_in = None
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, 416)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        frame_counter += 1

        print('processing frame',frame_counter)
        print('number of detected objects is',classes.sum())

        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        img = cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    
        if output_file:
            out.write(img)
            
        if cv2.waitKey(1) == ord('q'):
                break
    
    out.release()
    cv2.destroyAllWindows()