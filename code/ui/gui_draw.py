import numpy as np
import time
import cv2
from PyQt4.QtCore import *
from PyQt4.QtGui import *
try:
    from PyQt4.QtCore import QString
except ImportError:
    QString = str
from .ui_control import UIControl
# from data.lab_gamut import *
from data import lab_gamut
from skimage import color
import os
import datetime
import glob
import sys
from pdb import set_trace as st
import matplotlib.pyplot as plt

import skimage

class GUIDraw(QWidget):
    def __init__(self, model, dist_model=None, load_size=256, win_size=512, user_study=False, ui_time=60):
        QWidget.__init__(self)
        self.model = None
        self.image_file = None
        self.pos = None
        self.model = model
        self.dist_model = dist_model # distribution predictor, could be empty
        self.win_size = win_size
        self.load_size = load_size
        # self.scale = win_size / float(load_size)
        self.setFixedSize(win_size, win_size)
        self.uiControl = UIControl(win_size=win_size, load_size=load_size)
        self.move(win_size, win_size)
        self.movie = True
        self.init_color() # initialize color
        self.im_gray3 = None
        self.eraseMode = False
        self.ui_mode = 'none'   # stroke or point
        self.image_loaded = False
        self.use_gray = True
        self.total_images = 0
        self.image_id = 0
        self.user_study = user_study
        self.method = 'with_dist'
        self.ui_time = ui_time
        if user_study:
            self.reset_timer()



    def reset_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.nextImage)
        self.count_secs = self.ui_time
        self.timer.start((self.count_secs+5)*1000)
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.clock_count)
        self.clock_timer.start(1000)

    def clock_count(self):
        self.count_secs -= 1
        self.update()

    def init_result(self, image_file):
        self.read_image(image_file)  # read an image
        self.reset()

    def get_batches(self, img_dir):
        self.img_list = glob.glob(os.path.join(img_dir, '*.JPEG'))
        self.total_images = len(self.img_list)
        img_first = self.img_list[0]
        self.init_result(img_first)

    def nextImage(self):
        self.save_result()
        self.image_id += 1
        if self.image_id == self.total_images:
            print('you have finished all the results')
            sys.exit()
        img_current = self.img_list[self.image_id]
        # self.reset()
        self.init_result(img_current)
        self.reset_timer()


    def read_image(self, image_file):
        # self.result = None
        self.image_loaded = True
        self.image_file = image_file
        print(image_file)
        im_bgr = cv2.imread(image_file)
        self.im_full = im_bgr.copy()
        # get image for display
        h, w, c = self.im_full.shape
        max_width = max(h, w)
        r = self.win_size / float(max_width)
        # print('max_width = %d' % max_width)
        self.scale = float(self.win_size) / self.load_size
        print('scale = %f' % self.scale)
        # print('read_image_scale = %3.3f' % self.scale)
        rw = int(round(r * w / 4.0 ) * 4)
        rh = int(round(r * h / 4.0 ) * 4)
        # dim = (int(r*w), int(r*h))

        self.im_win = cv2.resize(self.im_full, (rw, rh), interpolation=cv2.INTER_CUBIC)

        self.dw = int((self.win_size - rw) / 2)
        self.dh = int((self.win_size - rh) / 2)
        self.win_w = rw
        self.win_h = rh
        self.uiControl.setImageSize((rw, rh))
        im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
        self.im_gray3 = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)

        self.gray_win = cv2.resize(self.im_gray3, (rw, rh), interpolation=cv2.INTER_CUBIC)
        im_bgr = cv2.resize(im_bgr, (self.load_size, self.load_size), interpolation=cv2.INTER_CUBIC)
        # cv2.imshow('im_win', self.im_win)
        # cv2.waitKey(1)
        self.im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        lab_win = color.rgb2lab(self.im_win[:,:,::-1])

        self.im_lab = color.rgb2lab(im_bgr[:,:,::-1])
        self.im_l = self.im_lab[:,:,0]
        self.l_win = lab_win[:,:,0]#cv2.resize(self.im_l, (rw, rh),  interpolation=cv2.INTER_CUBIC)
        self.im_ab = self.im_lab[:,:,1:]
        self.im_size = self.im_rgb.shape[0:2]

        self.im_ab0 = np.zeros((2, self.load_size, self.load_size))
        self.im_mask0 = np.zeros((1, self.load_size, self.load_size))
        self.brushWidth = 2 * self.scale

        self.model.load_image(image_file)

        if (self.dist_model is not None):
            self.dist_model.set_image(self.im_rgb)
            self.predict_color()


    def update_im(self):
        self.update()
        QApplication.processEvents()


    def update_ui(self, move_point=True):
        if self.ui_mode == 'none':
            return False
        is_predict = False
        snap_qcolor = self.calibrate_color(self.user_color, self.pos)
        self.color = snap_qcolor
        self.emit(SIGNAL('update_color'), QString('background-color: %s' % self.color.name()))

        if self.ui_mode == 'point':
            if move_point:
                self.uiControl.movePoint(self.pos, snap_qcolor, self.user_color, self.brushWidth)
            else:
                self.user_color, self.brushWidth, isNew = self.uiControl.addPoint(self.pos, snap_qcolor, self.user_color, self.brushWidth)
                if isNew:
                    is_predict = True
                    # self.predict_color()

        if self.ui_mode == 'stroke':
            self.uiControl.addStroke(self.prev_pos, self.pos, snap_qcolor, self.user_color, self.brushWidth)
        if self.ui_mode == 'erase':
            isRemoved = self.uiControl.erasePoint(self.pos)
            if isRemoved:
                is_predict = True
                # self.predict_color()
        return is_predict

    def reset(self):
        self.ui_mode = 'none'
        self.pos = None
        self.result = None
        self.user_color = None
        self.color = None
        self.uiControl.reset()
        self.init_color()
        self.compute_result()
        self.predict_color()
        self.update()

    def scale_point(self, pnt):
        x = int((pnt.x() - self.dw) / float(self.win_w) * self.load_size)
        y = int((pnt.y() - self.dh) / float(self.win_h) * self.load_size)
        # print('scale point (%d, %d)' % (x, y))
        return x, y

    def valid_point(self, pnt):
        if pnt is None:
            print('WARNING: no point\n')
            return None
        else:
            if pnt.x() >= self.dw and pnt.y() >= self.dh and pnt.x() < self.win_size-self.dw and pnt.y() < self.win_size-self.dh:
                x = int(np.round(pnt.x()))
                y = int(np.round(pnt.y()))
                return QPoint(x, y)
            else:
                print('WARNING: invalid point (%d, %d)\n' % (pnt.x(), pnt.y()))
                return None

    def init_color(self):
        self.user_color = QColor(128, 128, 128)  # default color red
        self.color = self.user_color

    def change_color(self, pos=None):
        if pos is not None:
            x, y = self.scale_point(pos)
            L = self.im_lab[y, x, 0]
            self.emit(SIGNAL('update_gamut'), L)
            rgb_colors = self.suggest_color(h=y, w=x, K=9)
            rgb_colors[-1, :] = 0.5
            # print('rgb_colors', rgb_colors)
            # if self.user_color is None:
            #     self.emit(SIGNAL('change_color_id'), 0)
            self.emit(SIGNAL('suggest_colors'), rgb_colors)
            used_colors = self.uiControl.used_colors()
            # print('used_colors', used_colors)
            self.emit(SIGNAL('used_colors'), used_colors)
            # print('change_color L', L)
            snap_color = self.calibrate_color(self.user_color, pos)
            # print('change_color snap_color', snap_color.red(), snap_color.green(), snap_color.blue())
            c = np.array((snap_color.red(), snap_color.green(), snap_color.blue()), np.uint8)

            self.emit(SIGNAL('update_ab'), c)

    def calibrate_color(self, c, pos):
        x, y = self.scale_point(pos)
        P = int(self.brushWidth / self.scale)

            # snap color based on L color
        color_array = np.array((c.red(), c.green(), c.blue())).astype(
            'uint8')
        mean_L =  self.im_l[y, x]#np.mean(self.im_l[y - P:y + P + 1, x - P:x + P + 1])
        snap_color = lab_gamut.snap_ab(mean_L, color_array)
            # print('  RGB (snapping): (%i,%i,%i)'%(snap_color[0],snap_color[1],snap_color[2]))
        snap_qcolor = QColor(snap_color[0], snap_color[1], snap_color[2])
        return snap_qcolor

    def set_color(self, c_rgb):
        # print('set color', c_rgb)
        c = QColor(c_rgb[0], c_rgb[1], c_rgb[2])
        self.user_color = c
        snap_qcolor = self.calibrate_color(c, self.pos)
        # print('set_color snap_color', snap_qcolor.red(), snap_qcolor.green(), snap_qcolor.blue())
        self.color = snap_qcolor
        self.emit(SIGNAL('update_color'), QString('background-color: %s' % self.color.name()))
        self.uiControl.update_color(snap_qcolor, self.user_color)
        self.compute_result()

    def erase(self):
        self.eraseMode = not self.eraseMode

    def load_image(self):
        img_path = str(QFileDialog.getOpenFileName(self, 'load an input image'))
        self.init_result(img_path)

    def save_result(self):
        path = os.path.abspath(self.image_file)
        path, ext = os.path.splitext(path)

        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = "_".join([path, self.method, suffix])

        print('saving result to <%s>\n' % save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        np.save(os.path.join(save_path, 'im_l.npy'), self.model.img_l)
        np.save(os.path.join(save_path, 'im_ab.npy'), self.im_ab0)
        np.save(os.path.join(save_path, 'im_mask.npy'), self.im_mask0)

        result_bgr = cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)
        mask = self.im_mask0.transpose((1,2,0)).astype(np.uint8)*255
        cv2.imwrite(os.path.join(save_path, 'input_mask.png'), mask)
        cv2.imwrite(os.path.join(save_path, 'ours.png'), result_bgr)
        cv2.imwrite(os.path.join(save_path, 'ours_fullres.png'), self.model.get_img_fullres()[:, :, ::-1])
        cv2.imwrite(os.path.join(save_path, 'input_fullres.png'), self.model.get_input_img_fullres()[:, :, ::-1])
        cv2.imwrite(os.path.join(save_path, 'input.png'), self.model.get_input_img()[:, :, ::-1])
        cv2.imwrite(os.path.join(save_path, 'input_ab.png'), self.model.get_sup_img()[:, :, ::-1])


    def enable_gray(self):
        self.use_gray = not self.use_gray
        self.update()


    def predict_color(self):
        if self.dist_model is not None and self.image_loaded:
            # t0 = time.time()
            im, mask = self.uiControl.get_input()
            im_mask0 = mask > 0.0
            self.im_mask0 = im_mask0.transpose((2, 0, 1))
            im_lab = color.rgb2lab(im).transpose((2, 0, 1))
            self.im_ab0 = im_lab[1:3, :, :]

            self.dist_model.net_forward(self.im_ab0, self.im_mask0)
            # print('dist timing = %dms' % int((time.time() - t0)*1000))

    def suggest_color(self, h, w, K=5):
        if self.dist_model is not None and self.image_loaded:
            ab, conf = self.dist_model.get_ab_reccs(h=h, w=w, K=K, N=25000, return_conf=True)
            L = np.tile(self.im_lab[h, w, 0], (K, 1))
            colors_lab = np.concatenate((L, ab), axis=1)
            colors_lab3 = colors_lab[:,np.newaxis, :]
            colors_rgb = np.clip(np.squeeze(color.lab2rgb(colors_lab3)),0,1) # [0, 1] Kx3
            colors_rgb_withcurr = np.concatenate((self.model.get_img_forward()[h,w,np.newaxis,:]/255.,colors_rgb),axis=0)
            return colors_rgb_withcurr
        else:
            return None

    def compute_result(self):
        im, mask = self.uiControl.get_input()
        im_mask0 = mask > 0.0
        self.im_mask0 = im_mask0.transpose((2,0,1))
        im_lab = color.rgb2lab(im).transpose((2,0,1))
        self.im_ab0 = im_lab[1:3, :, :]

        # t0 = time.time()
        self.model.net_forward(self.im_ab0,self.im_mask0)
        ab = self.model.output_ab.transpose((1,2,0))
        # print('l_shape', self.l_win.shape)
        # print('result (w,h)=(%d,%d)' % (self.win_w, self.win_h))
        ab_win = cv2.resize(ab, (self.win_w, self.win_h), interpolation=cv2.INTER_CUBIC)
        pred_lab = np.concatenate((self.l_win[..., np.newaxis], ab_win), axis=2)
        pred_rgb = (np.clip(color.lab2rgb(pred_lab), 0, 1) * 255).astype('uint8')
        self.result = pred_rgb
        # print('predict timing = %dms' % int((time.time()-t0)*1000))
        self.emit(SIGNAL('update_result'), self.result)
        self.update()


    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.fillRect(event.rect(),QColor(49, 54, 49))
        painter.setRenderHint(QPainter.Antialiasing)
        if self.use_gray or self.result is None:
            im = self.gray_win
        else:
            im = self.result

        if im is not None:
            qImg = QImage(im.tostring(), im.shape[1], im.shape[0], QImage.Format_RGB888)
            painter.drawImage(self.dw, self.dh, qImg)

        if im is not None and self.user_study and self.count_secs >= 0:
            if self.count_secs == 10 or self.count_secs <= 5 or self.count_secs == 30:
                # print('count seconds = %d\n' % self.count_secs)
                painter.setPen(QPen(Qt.red, 20, Qt.SolidLine, cap=Qt.RoundCap, join=Qt.RoundJoin))
                font_sz = 36
                painter.setFont(QFont("times", font_sz))
                painter.drawText(self.win_size/2-font_sz/2, font_sz, QString(str(self.count_secs)))

        if im is not None and self.user_study and self.count_secs >= -5 and self.count_secs < 0:

            painter.setPen(QPen(Qt.red, 20, Qt.SolidLine, cap=Qt.RoundCap, join=Qt.RoundJoin))
            font_sz = 36
            painter.setFont(QFont("times", font_sz))
            painter.drawText(self.win_size / 2 - font_sz / 2-10, font_sz, QString(str('Timeout')))
        self.uiControl.update_painter(painter)
        painter.end()


    def wheelEvent(self, event):
        d = event.delta() / 120
        self.brushWidth = min(4.05*self.scale, max(0, self.brushWidth+ d*self.scale))
        print('update brushWidth = %f' % self.brushWidth)
        self.update_ui(move_point=True)
        self.update()

    def is_same_point(self, pos1, pos2):
        if pos1 is None or pos2 is None:
            return False
        dx = pos1.x() - pos2.x()
        dy = pos1.y() - pos2.y()
        d = dx * dx + dy * dy
        # print('distance between points = %f' % d)
        return d < 25

    def mousePressEvent(self, event):
        if self.user_study:
            if self.count_secs < 0:
                return
        print('mouse press', event.pos())
        pos = self.valid_point(event.pos())


        if pos is not None:
            if event.button() == Qt.LeftButton:
                self.pos = pos
                self.ui_mode = 'point'
                self.change_color(pos)
                self.update_ui(move_point=False)
                self.compute_result()

            if event.button() == Qt.RightButton:
                # draw the stroke
                self.pos = pos
                self.ui_mode = 'erase'# if self.eraseMode else 'stroke'
                self.update_ui(move_point=False)
                self.compute_result()



    def mouseMoveEvent(self, event):
        if self.user_study:
            if self.count_secs < 0:
                return
        print('mouse move', event.pos())
        self.pos = self.valid_point(event.pos())
        if self.pos is not None:
            if self.ui_mode == 'point':
                self.update_ui(move_point=True)
                self.compute_result()

    def mouseReleaseEvent(self, event):
        pass

    def sizeHint(self):
        return QSize(self.win_size, self.win_size)  # 28 * 8
