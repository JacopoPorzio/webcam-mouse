import cv2
import mouse
import numpy as np
import screeninfo
import matplotlib.pyplot as plt
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
        # cv2.namedWindow("cen_vis")
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
            # img = np.flip(img, axis=1)
            # cv2.imshow("cen_vis", img)
        else:
            cX = -100
            cY = -100
            # cv2.imshow("cen_vis", hue_channel)
        return cX, cY  # Centroid x and y position in image coordinate.


class MicrophoneReader:
    def __init__(self, th_frequency, th_amplitude):
        self._format = pyaudio.paInt16
        self._channels = 1
        self._rate = 48000
        self._chunk = int(1024*2.5)
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

    def process_microphone_max_amp(self):
        data = self.read_microphone()
        max_amplitude = np.max(data)
        print('Max amplitude: ', max_amplitude)
        action = (max_amplitude > self._threshold_amplitude)*1
        return action, [round(max_amplitude, 2)]

    def process_microphone(self):
        data = self.read_microphone()
        dft = np.fft.fft(data) / len(data)
        dft = abs(dft[range(int(len(data) / 2))])
        values = np.arange(int(len(data) / 2))
        time_period = len(data) / self._rate
        frequencies = values / time_period
        # Limit the frequencies to meaningful ones.
        freq_lim = np.argwhere(frequencies > 5000)[0][0]
        dft = dft[:freq_lim]
        frequencies = frequencies[:freq_lim]
        # Find the index of the maximum amplitude in the DFT.
        max_amplitude_idx = np.argmax(np.abs(dft))
        # Get the corresponding frequency.
        max_amplitude_freq = frequencies[max_amplitude_idx]
        max_amplitude = dft[max_amplitude_idx]
        # If frequency < threshold: left click,
        # if frequency > threshold: right click

        action = (
                (max_amplitude > self._threshold_amplitude) *
                (
                        (max_amplitude_freq <= self._threshold_frequency) * 1 +
                        (max_amplitude_freq > self._threshold_frequency) * 2
                )
                )

        return action, [round(max_amplitude, 2), round(max_amplitude_freq, 2)]

    def kill_stream(self):
        self._stream.stop_stream()
        self._stream.close()
        self._audio.terminate()


class Mouse:
    def __init__(self, error_threshold, cam_h, cam_w, alpha):
        # Parameters.
        self._error_threshold = error_threshold
        self._alpha = alpha
        screen_height, screen_width = read_monitor()
        self._cam_h = cam_h
        self._cam_w = cam_w
        self._ratio_h = screen_height/cam_h
        self._ratio_w = screen_width/cam_w
        initial_x_mouse, initial_y_mouse = mouse.get_position()
        self._last_x_img = initial_x_mouse  # Initialization to grant initial movement.
        self._last_y_img = initial_y_mouse  # Initialization to grant initial movement.
        self._last_button_pressed = ' '

    def act(self, x_img, y_img, act_from_mic):  # x_image, y_image):
        x_image = self._cam_w - x_img
        y_image = y_img  # self._cam_h - y_img
        if x_image != -100 and y_image != -100:
            x_screen = x_image*self._ratio_w
            y_screen = y_image*self._ratio_h

            x_to_move = self._alpha*x_screen + (1 - self._alpha)*self._last_x_img
            y_to_move = self._alpha*y_screen + (1 - self._alpha)*self._last_y_img

            mouse.move(x_to_move, y_to_move, absolute=True)
            if act_from_mic != 0:
                button_to_press = (act_from_mic == 1)*'left' + (act_from_mic == 2)*'right'
                if button_to_press != self._last_button_pressed:
                    if self._last_button_pressed != ' ':
                        mouse.release(self._last_button_pressed)
                    mouse.press(button=button_to_press)
                    self._last_button_pressed = button_to_press
            else:
                if self._last_button_pressed != ' ':
                    mouse.release(self._last_button_pressed)
                    self._last_button_pressed = ' '
            self._last_x_img = x_image
            self._last_y_img = y_image
        else:
            print('Nothing is detected in front of the webcam.')
        return


class ImageWindow:
    def __init__(self):
        cv2.namedWindow("preview")

    def update_image(self, frm, xc, yc, img_add):
        # hue_plus_centroid = cv2.circle(hue_channel, (cX, cY), 5, (0, 0, 255), -1)
        # cv2.imshow("centroid_visualization", hue_plus_centroid)
        img = cv2.line(frm, (xc - 10, yc), (xc + 10, yc), (0, 0, 255), 5)
        img = cv2.line(img, (xc, yc - 10), (xc, yc + 10), (0, 0, 255), 5)
        img = np.flip(img, axis=1)
        txt = f'Max a: {img_add[0]} [-]'
        img = cv2.putText(img.astype(int), txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if len(img_add) == 2:
            txt = f'Max f: {img_add[1]} [Hz]'
            img = cv2.putText(img, txt, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("preview", img.astype('uint8'))

def read_monitor():
    monitor = screeninfo.get_monitors()[0]
    return monitor.height, monitor.width


if __name__ == '__main__':
    hue_values = [55, 80]
    err_th = 25
    mouse_mov_th = 0.4
    threshold_frequency = 1500
    threshold_amplitude = 8000

    img_window = ImageWindow()
    vc = cv2.VideoCapture(0)

    webcam_reader = WebcamReader(webcam=vc, min_hue=hue_values[0], max_hue=hue_values[1])
    microphone_reader = MicrophoneReader(threshold_frequency, threshold_amplitude)

    if vc.isOpened():
        rval, frame = webcam_reader.read_webcam()
        mouse_device = Mouse(error_threshold=err_th, cam_h=frame.shape[0],
                             cam_w=frame.shape[1], alpha=mouse_mov_th)
    else:
        rval = False

    while rval:
        rval, frame = webcam_reader.read_webcam()
        # webcam_reader.calibrate_hue(frame)  # You can use this line to calibrate the hue values.
        mic_action, add_to_image = microphone_reader.process_microphone()
        x_center, y_center = webcam_reader.find_centroid(frame)
        mouse_device.act(x_center, y_center, mic_action)
        img_window.update_image(frame, x_center, y_center, add_to_image)
        key = cv2.waitKey(20)
        if key == 27:
            microphone_reader.kill_stream()
            break

