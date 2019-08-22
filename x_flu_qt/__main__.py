import sys
from PyQt5.QtWidgets import QApplication
from x_flu_qt import x_flu_qt

def main():
    app = QApplication(sys.argv)
    ex = x_flu_qt.GUI()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
