import cv2
import mouse
import numpy as np
import screeninfo
# import matplotlib.pyplot as plt
import pyaudio
# import librosa
import struct


class WebcamReader:
    def __init__(self, webcam, min_hue, max_hue):
        self._min_hue = min_hue
        self._max_hue = max_hue
        self._webcam = webcam

    def read_webcam(self):
        r_rval, r_frame = self._webcam.read()
        return r_rval, r_frame

    def calibrate_hue(self, frame_rgb):
        cv2.namedWindow("hue_calibration")
        frame_hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV)
        print('Ciauz')
        hue_channel = frame_hsv[:, :, 0]*2
        # Why `*2`? https://stackoverflow.com/questions/21737613/image-of-hsv-color-wheel-for-opencv
        hue_channel[hue_channel < self._min_hue] = 0
        hue_channel[hue_channel > self._max_hue] = 0
        hue_channel[hue_channel > 0] = 255
        cv2.imshow("hue_calibration", hue_channel)

    def find_centroid(self, frame_rgb):
        # Centroid visualization.
        cv2.namedWindow("cen_vis")
        # RGB -> HSV.
        frame_hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV)
        hue_channel = frame_hsv[:, :, 0]*2
        # Why `*2`? https://stackoverflow.com/questions/21737613/image-of-hsv-color-wheel-for-opencv
        hue_channel[hue_channel < self._min_hue] = 0
        hue_channel[hue_channel > self._max_hue] = 0
        hue_channel[hue_channel > 0] = 255
        # We must now filter the image, in order to remove noise.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))  # (5, 5)   # ELLIPSE, RECT, CROSS
        hue_channel = cv2.morphologyEx(hue_channel, cv2.MORPH_OPEN, kernel)
        # hue_channel = np.flip(hue_channel, axis=1)
        # Blob identification.
        contours, _ = cv2.findContours(hue_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = np.zeros(len(contours))
        x_centers = np.zeros(len(contours))
        y_centers = np.zeros(len(contours))
        aspect_ratios = np.zeros(len(contours))

        for k, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            areas[k] = area
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratios[k] = float(w) / h if h != 0 else 0
            M = cv2.moments(contour)
            if M["m00"] != 0:
                x_centers[k] = int(M["m10"] / M["m00"])
                y_centers[k] = int(M["m01"] / M["m00"])

        x_centers = x_centers.astype(int)
        y_centers = y_centers.astype(int)

        if areas.shape[0] != 0:
            k_max = np.argmax(areas)
            cX = x_centers[k_max]
            cY = y_centers[k_max]
            img = cv2.line(hue_channel, (cX - 10, cY), (cX + 10, cY), (0, 0, 255), 5)
            img = cv2.line(img, (cX, cY - 10), (cX, cY + 10), (0, 0, 255), 5)
            img = np.flip(img, axis=1)
            cv2.imshow("cen_vis", img)
        else:
            cX = -100
            cY = -100
            cv2.imshow("cen_vis", hue_channel)
        return cX, cY  # Centroid x and y position in image coordinate.


class MicrophoneReader:
    def __init__(self, th_frequency, th_amplitude):
        self._format = pyaudio.paInt16
        self._channels = 1
        self._rate = 48000
        self._chunk = 1024*2
        self._threshold_frequency = th_frequency
        self._threshold_amplitude = th_amplitude
        self._audio = pyaudio.PyAudio()
        self._stream = self._audio.open(format=self._format,
                                        channels=self._channels,
                                        rate=self._rate,
                                        input=True,
                                        input_device_index=1,
                                        frames_per_buffer=self._chunk)
        # Initialize microphone.
        for _ in range(10):
            self.read_microphone()

    def read_microphone(self):
        data = self._stream.read(self._chunk)
        int_values = struct.unpack(f"{len(data)//2}h", data)
        return int_values

    def process_microphone(self):
        data = self.read_microphone()
        max_amplitude = np.max(data)
        print('Max amplitude: ', max_amplitude)
        action = (max_amplitude > self._threshold_amplitude)*1
        return action

    def kill_stream(self):
        self._stream.stop_stream()
        self._stream.close()
        self._audio.terminate()


class Mouse:
    def __init__(self, error_threshold, cam_h, cam_w, move_th, max_signal):
        # Parameters.
        self._error_threshold = error_threshold
        self._max_signal = max_signal
        screen_height, screen_width = read_monitor()
        self._cam_h = cam_h
        self._cam_w = cam_w
        self._ratio_h = screen_height/cam_h
        self._ratio_w = screen_width/cam_w
        self._move_threshold = move_th
        initial_x_mouse, initial_y_mouse = mouse.get_position()
        self._last_x_img = initial_x_mouse  # Initialization to grant initial movement.
        self._last_y_img = initial_y_mouse  # Initialization to grant initial movement.

    def act(self, x_img, y_img, act_from_mic):  # x_image, y_image):
        x_image = self._cam_w - x_img
        y_image = y_img  # self._cam_h - y_img
        if x_image != -100 and y_image != -100:
            x_screen = x_image*self._ratio_w
            y_screen = y_image*self._ratio_h

            delta_x = x_screen - self._last_x_img
            delta_y = y_screen - self._last_y_img

            x_to_move = (x_screen*(abs(delta_x) > self._move_threshold) +
                         self._last_x_img*(abs(delta_x) < self._move_threshold))

            y_to_move = (y_screen*(abs(delta_y) > self._move_threshold) +
                         self._last_y_img*(abs(delta_y) < self._move_threshold))

            mouse.move(x_to_move, y_to_move, absolute=True)
            if act_from_mic != 0:
                button_to_press = (act_from_mic == 1)*'left' + (act_from_mic == 2)*'right'
                mouse.press(button=button_to_press)
            else:
                mouse.release()
            self._last_x_img = x_image
            self._last_y_img = y_image
        else:
            print('Nothing is detected in front of the webcam.')
        return


def read_monitor():
    monitor = screeninfo.get_monitors()[0]
    return monitor.height, monitor.width


if __name__ == '__main__':
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    hue_values = [55, 80]
    err_th = 25
    mouse_mov_th = 5
    threshold_frequency = 440
    threshold_amplitude = 10000

    webcam_reader = WebcamReader(webcam=vc, min_hue=hue_values[0], max_hue=hue_values[1])
    microphone_reader = MicrophoneReader(threshold_frequency, threshold_amplitude)

    if vc.isOpened():
        rval, frame = webcam_reader.read_webcam()
        mouse_device = Mouse(error_threshold=err_th, cam_h=frame.shape[0],
                             cam_w=frame.shape[1], move_th=mouse_mov_th, max_signal=100)
    else:
        rval = False

    while rval:
        # cv2.imshow("preview", frame)  # We can avoid it.
        rval, frame = webcam_reader.read_webcam()
        # webcam_reader.calibrate_hue(frame)
        mic_action = microphone_reader.process_microphone()
        x_center, y_center = webcam_reader.find_centroid(frame)
        mouse_device.act(x_center, y_center, mic_action)
        # hue_plus_centroid = cv2.circle(hue_channel, (cX, cY), 5, (0, 0, 255), -1)
        # cv2.imshow("centroid_visualization", hue_plus_centroid)
        # img = np.flip(frame, axis=1)
        img = cv2.line(frame, (x_center - 10, y_center), (x_center + 10, y_center), (0, 0, 255), 5)
        img = cv2.line(img, (x_center, y_center - 10), (x_center, y_center + 10), (0, 0, 255), 5)
        img = np.flip(img, axis=1)
        cv2.imshow("preview", img)
        key = cv2.waitKey(20)
        if key == 27:
            microphone_reader.kill_stream()
            break

