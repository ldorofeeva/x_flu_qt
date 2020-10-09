# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 17:16:09 2019

@author: elisa
"""
import os, re
import sys

from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from rangeslider import RangeSlider
import h5py
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('QT5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from scipy.interpolate import griddata


class GUI(QDialog):

    def __init__(self):
        super().__init__()

        self.Nexus_spec = dict(
            ftype=('.hdf5', '.nxs', '.nex'),
            grp_prefix='entry',
            grp='entry1',
            ds_x='sample/positioner/samx',
            ds_y='sample/positioner/samz',
            ds_signal='data/32elem'
        )

        self.smoothness = ['none', 'spline16', 'bilinear', 'bicubic']
        # initialize vars
        self.xs = None
        self.ys = None
        self.signal = None
        self.xmin = self.ymin = 0
        self.xmax = self.ymax = 1

        self.xi = self.yi = None

        self.grid_step = 0.025
        self.interp_method = 'nearest'
        self.cmap = 'viridis'
        self.smooth = self.smoothness[0]

        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.zi = np.zeros([3, 3])
        self.im = self.ax.imshow(self.zi, cmap=self.cmap, extent=[self.xmin, self.xmax, self.ymin, self.ymax],
                                 aspect='auto')
        self.cb = self.fig.colorbar(self.im, ax=self.ax, use_gridspec=True)
        self.ax.set_xlabel('X, mm'), self.ax.set_ylabel('Y, mm')
        plt.tight_layout()

        # get x-ray energy lines
        data_path = os.path.join(os.path.dirname(__file__), 'data')
        self.xrelines_names = np.loadtxt(os.path.join(data_path, 'pertxre_lines.csv'), delimiter=',', dtype=object,
                                         usecols=(0, 1))
        self.xrelines_vals = np.loadtxt(os.path.join(data_path, 'pertxre_lines.csv'), delimiter=',', dtype=int,
                                        usecols=2)

        self.setWindowTitle('X_flu')
        self.setFixedSize(1200, 600)

        self.create_data_grp()
        self.create_plot_grp()
        self.create_erange_grp()
        self.create_options_grp()

        mainLayout = QGridLayout()
        mainLayout.addWidget(self.select_data_grp, 0, 0)
        mainLayout.addWidget(self.erange_grp, 1, 0)
        mainLayout.addWidget(self.opt_grp, 2, 0)
        mainLayout.setRowStretch(1, 2)
        self.erange_grp.setEnabled(False)
        mainLayout.addWidget(self.plot_grp, 0, 1, 3, 1)
        self.plot_grp.setEnabled(False)
        self.setLayout(mainLayout)
        self.show()

    def create_data_grp(self):
        self.select_data_grp = QGroupBox("Select data")

        self.fname_lbl = QLabel(self)
        self.fname_lbl.setText("Select file:")
        self.fname = QLineEdit(self)
        self.fname.setReadOnly(True)
        self.fopen = QPushButton("...")
        self.fopen.clicked.connect(self.open_file)

        self.entry = QComboBox()
        self.entry.activated[str].connect(self.load_entry)
        self.entry_lbl = QLabel("Select entry:")
        # self.entry_lbl.setBuddy(self.entry)

        layout = QGridLayout()
        layout.addWidget(self.fname_lbl, 0, 0)
        layout.addWidget(self.fname, 1, 0, 1, 3)
        layout.addWidget(self.fopen, 1, 4)
        layout.addWidget(self.entry_lbl, 2, 0, 1, 2)
        layout.addWidget(self.entry, 3, 0)
        self.select_data_grp.setLayout(layout)
        # layout.setColumnStretch(1, 4)
        # layout.setColumnStretch(2, 4)

    def create_plot_grp(self):
        self.plot_grp = QGroupBox("X-Ray fluorescence map")

        self.wgt_plot = QWidget(self)
        self.wgt_plot.setLayout(QVBoxLayout())
        self.plot = FigureCanvas(self.fig)
        self.wgt_plot.canvas = self.plot
        self.wgt_plot.toolbar = NavigationToolbar(self.wgt_plot.canvas, self.wgt_plot)
        self.wgt_plot.layout().addWidget(self.wgt_plot.toolbar)
        self.wgt_plot.layout().addWidget(self.wgt_plot.canvas)

        self.smooth_lbl0 = QLabel('Coarse')
        self.smooth_lbl1 = QLabel('Smooth')

        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setMinimumHeight(20)
        self.smooth_slider.setMinimum(0)
        self.smooth_slider.setMaximum(3)
        self.smooth_slider.setValue(0)
        self.smooth_slider.setTickPosition(QSlider.TicksBelow)
        self.smooth_slider.setTickInterval(1)
        self.smooth_slider.valueChanged.connect(self.move_smooth_slider)

        layout = QGridLayout()
        layout.addWidget(self.wgt_plot, 0, 0, 1, 5)
        layout.setRowStretch(0, 20)
        layout.addWidget(self.smooth_lbl0, 1, 1)
        layout.addWidget(self.smooth_lbl1, 1, 3)
        layout.addWidget(self.smooth_slider, 2, 2)
        self.plot_grp.setLayout(layout)

    def create_erange_grp(self):
        # Create the QML user interface.  
        self.erange_grp = QGroupBox("Select energy range")

        self.erange_slider = RangeSlider(Qt.Horizontal)
        self.erange_slider.setMinimumHeight(20)
        self.erange_slider.setMinimum(0)
        self.erange_slider.setMaximum(20470)
        self.erange_slider.setLow(0)
        self.erange_slider.setHigh(20470)
        self.erange_slider.setTickInterval(10)
        self.erange_slider.sliderMoved.connect(self.move_erange_slider)

        self.e_from_lbl = QLabel('From: ')
        self.e_from = QLineEdit(self)
        self.e_from.setText(str(self.erange_slider.low()))
        self.e_from.editingFinished.connect(self.type_e_from)
        self.e_to_lbl = QLabel('to: ')
        self.e_to = QLineEdit(self)
        self.e_to.setText(str(self.erange_slider.high()))
        self.e_to.editingFinished.connect(self.type_e_to)
        self.e_units_lbl = QLabel(' eV ')

        self.e_lines_lbl = QLabel('Lines')
        self.e_lines = QPlainTextEdit(self)
        self.e_lines.setPlainText('Lines in energy region\n\n\n')
        self.e_lines.setReadOnly(True)

        layout = QGridLayout()
        layout.addWidget(self.erange_slider, 0, 0, 1, 4)
        layout.addWidget(self.e_from_lbl, 1, 0)
        layout.addWidget(self.e_from, 1, 1)
        layout.addWidget(self.e_to_lbl, 1, 2)
        layout.addWidget(self.e_to, 1, 3)
        layout.addWidget(self.e_units_lbl, 1, 4)
        layout.addWidget(self.e_lines_lbl, 2, 0, Qt.AlignTop)
        layout.addWidget(self.e_lines, 2, 1, 4, 3)
        self.erange_grp.setLayout(layout)

    def create_options_grp(self):
        self.opt_grp = QGroupBox("Advanced options")
        # self.opt_grp.setCheckable(True)
        # self.opt_grp.setChecked(False)

        self.grid_step_lbl = QLabel('Grid step:')
        self.grid_step_entry = QLineEdit(self)
        self.grid_step_entry.setText(str(self.grid_step))
        self.int_type_lbl = QLabel('Interpolation:')
        self.int_type = QComboBox()
        self.int_type.addItems(['nearest', 'linear'])
        self.int_type.setCurrentIndex(0)

        self.cmap_lbl = QLabel('Colormap:')
        self.cmap_choose = QComboBox()
        self.cmap_choose.addItems(['viridis', 'jet', 'hsv', 'Greys'])
        self.cmap_choose.setCurrentIndex(0)

        self.opt_apply = QPushButton("Apply")
        self.opt_apply.clicked.connect(self.apply_advanced_opts)
        self.opt_apply.setEnabled(False)

        layout = QGridLayout()
        layout.addWidget(self.grid_step_lbl, 0, 0, )
        layout.addWidget(self.grid_step_entry, 1, 0)
        layout.addWidget(self.int_type_lbl, 0, 1)
        layout.addWidget(self.int_type, 1, 1)
        layout.addWidget(self.cmap_lbl, 0, 2)
        layout.addWidget(self.cmap_choose, 1, 2)
        layout.addWidget(self.opt_apply, 1, 3)

        self.opt_grp.setLayout(layout)

    def open_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fname_val, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "All Files (*);;Python Files (*.py)", options=options)
        if fname_val:
            if not os.path.isfile(fname_val):
                alert = QMessageBox()
                alert.setText(f'Unable to open file: {fname_val} - no such file')
                alert.exec_()
                return
            if not fname_val.endswith(self.Nexus_spec['ftype']):
                alert = QMessageBox()
                alert.setText('File format not supported')
                alert.exec_()
            else:
                with h5py.File(fname_val, "r") as h5f:
                    # load entry names
                    grps = ()
                    for k in h5f.keys():
                        dsinf = str(h5f[k])
                        enames = re.search(f'{self.Nexus_spec["grp_prefix"]}([0-9]+)', dsinf)
                        if enames:
                            ename = enames.group(0)
                            grps = grps + (ename,)
                            # check the correctness of datasets naming
                if len(grps) == 0:
                    alert = QMessageBox()
                    alert.setText('File does not correspond NeXus format. Please select another file')
                    alert.exec_()
                    return

                self.fname.setText(fname_val)
                self.entry.clear()
                self.entry.addItem('')
                self.entry.addItems(grps)
                self.entry.setCurrentIndex(0)
                self.clear_view()

    def load_entry(self, ename_val):
        if ename_val:
            fname_val = self.fname.text()
            with h5py.File(fname_val, "r") as h5f:
                xs = np.asarray(h5f[ename_val + '/' + self.Nexus_spec['ds_x']])
                self.ys = np.asarray(h5f[ename_val + '/' + self.Nexus_spec['ds_y']])
                self.signal = np.sum(np.asarray(h5f[ename_val + '/' + self.Nexus_spec['ds_signal']]), axis=1)
                # check the shape of datasets, if wrong
                if not xs.shape[0] == self.ys.shape[0]:
                    alert = QMessageBox()
                    alert.setText(f'Datasets shape do not correspond, group {ename_val}')
                    alert.exec_()
                    return
                if self.signal.shape[0] < xs.shape[0]:
                    alert = QMessageBox()
                    alert.setText(f'Axes dataset is longer then signal dataset. Reducing axes')
                    alert.exec_()
                    xs = xs[:len(self.signal)]
                    self.ys = self.ys[:len(self.signal)]
                elif self.signal.shape[0] > xs.shape[0]:
                    alert = QMessageBox()
                    alert.setText(f'Signal dataset is longer then axes dataset. Reducing signal')
                    alert.exec_()
                    self.signal = self.signal[:len(xs)]
            ### ************************************************************************************
            ###
            ### The following part of the code is taken from the example provided  
            ### 
            ### although the physical meaning of the following transformation is not clear to me,
            ### I leave this part it as is, asumming that such preprocessing is required
            ### due to specifics of data collection process
            ###
            ### ************************************************************************************

            # Shift the X data to line up the rows at center.
            # Pre-process the x values. The data needs to be shifted to the midway point between x[n+1] and x[n].
            sgm = False
            if sgm:
                shift = 0.5
                self.xs = np.zeros(len(xs))
                self.xs[0] = xs[0]
                for i in range(1, len(self.xs)):
                    self.xs[i] = xs[i] + shift * (xs[i] - xs[i - 1])
            else:
                self.xs = xs

            self.xmin = min(xs)
            self.xmax = max(xs)
            self.ymin = min(self.ys)
            self.ymax = max(self.ys)

            e0 = 0
            e1 = 10 * (len(self.signal[0]) - 1)

            self.erange_slider.setMinimum(e0)
            self.erange_slider.setMaximum(e1)
            self.erange_slider.setLow(e0)
            self.erange_slider.setHigh(e1)
            self.e_from.setText(str(self.erange_slider.low()))
            self.e_to.setText(str(self.erange_slider.high()))
            self.e_lines.setPlainText('Lines in energy region\n\n\n')

            self.interpolate(True)

            self.erange_grp.setEnabled(True)
            self.opt_apply.setEnabled(True)
            self.update_plot()
            self.update_lines()
        else:
            self.erange_grp.setEnabled(False)
            self.plot_grp.setEnabled(False)
            self.opt_apply.setEnabled(False)

    def interpolate(self, reset_axlimits=False):
        self.grid_step = float(self.grid_step_entry.text())
        self.interp_method = self.int_type.currentText()
        self.cmap = self.cmap_choose.currentText()
        self.xi = np.arange(self.xmin, self.xmax + self.grid_step, self.grid_step)
        self.yi = np.arange(self.ymin, self.ymax + self.grid_step, self.grid_step)
        self.xi, self.yi = np.meshgrid(self.xi, self.yi)
        self.zi = griddata((self.xs, self.ys), self.signal, (self.xi, self.yi), method=self.interp_method)
        self.plot_grp.setEnabled(True)
        self.update_plot(reset_axlimits)

    def update_plot(self, reset_axlimits=False):
        idx0 = int(self.erange_slider.low() / 10)
        idx1 = int(self.erange_slider.high() / 10)
        self.im = self.ax.imshow(self.zi[:, :, idx0:idx1].sum(axis=2), interpolation=self.smooth, cmap=self.cmap,
                                 extent=[self.xmin, self.xmax, self.ymin, self.ymax], aspect='auto', origin='lower')
        if reset_axlimits:
            self.ax.set_xlim(self.xmin, self.xmax)
            self.ax.set_ylim(self.ymin, self.ymax)
        self.cb.remove()
        self.cb = self.fig.colorbar(self.im, ax=self.ax, use_gridspec=True)
        self.fig.canvas.draw_idle()

    def apply_advanced_opts(self):
        # do not reinterpolate if only cmap has changed
        if (self.grid_step == float(self.grid_step_entry.text())) and (
                self.interp_method == self.int_type.currentText()):
            self.cmap = self.cmap_choose.currentText()
            self.update_plot()
        else:
            self.interpolate()

    def clear_view(self):
        self.erange_slider.setMinimum(0)
        self.erange_slider.setMaximum(2550)
        self.erange_slider.setLow(0)
        self.erange_slider.setHigh(2550)
        self.erange_slider.setTickInterval(10)
        self.e_from.setText(str(self.erange_slider.low()))
        self.e_to.setText(str(self.erange_slider.high()))
        self.e_lines.setPlainText('Lines in energy region\n\n\n')

        self.xmin = self.ymin = 0
        self.xmax = self.ymax = 1

        self.zi = np.zeros([3, 3])
        self.im = self.ax.imshow(self.zi, cmap=self.cmap, extent=[self.xmin, self.xmax, self.ymin, self.ymax],
                                 aspect='auto')
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)
        self.cb.remove()
        self.cb = self.fig.colorbar(self.im, ax=self.ax, use_gridspec=True)
        self.fig.canvas.draw_idle()

        self.erange_grp.setEnabled(False)
        self.plot_grp.setEnabled(False)
        self.opt_apply.setEnabled(False)

    def move_erange_slider(self):
        changed = False
        if self.e_from.text() != str(self.erange_slider.low()):
            self.e_from.setText(str(self.erange_slider.low()))
            changed = True
        if self.e_to.text() != str(self.erange_slider.high()):
            self.e_to.setText(str(self.erange_slider.high()))
            changed = True
        if changed:
            self.update_plot()
            self.update_lines()

    def type_e_from(self):
        e_from = int(self.e_from.text())
        if e_from != str(self.erange_slider.low()):
            self.erange_slider.setLow(e_from)
            self.update_plot()
            self.update_lines()

    def type_e_to(self):
        e_to = int(self.e_to.text())
        if e_to != str(self.erange_slider.high()):
            self.erange_slider.setHigh(e_to)
            self.update_plot()
            self.update_lines()

    def update_lines(self):
        e_from_val = self.erange_slider.low()
        e_to_val = self.erange_slider.high()
        if e_to_val >= e_from_val:
            matches = self.xrelines_names[
                np.where(np.logical_and(self.xrelines_vals >= e_from_val, self.xrelines_vals <= e_to_val))[0]]
            s = '\n'.join(el + ' (' + ', '.join(l for l in matches[np.where(matches[:, 0] == el)][:, 1]) + ')' for el in
                          np.unique(matches[:, 0]))
            self.e_lines.setPlainText(s)
        else:
            self.e_lines.setPlainText('No lines in the region')

    def move_smooth_slider(self):
        self.smooth = self.smoothness[self.smooth_slider.value()]
        self.update_plot()


def main():
    app = QApplication(sys.argv)
    ex = GUI()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
