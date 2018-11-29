import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QIcon
from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtWebKit import *
from PyQt5.QtWebKitWidgets import *

# I created main.ui in QTDesigner found in ProgramData/Anaconda3/Library/bin on Windows
uifile = os.path.join(os.getcwd(), 'UI', 'main.ui') 
print(uifile)
form, base = uic.loadUiType(uifile)

class Example(base, form):
    def __init__(self):
        super(base,self).__init__()
        self.setupUi(self)
        self.actionQuit.triggered.connect(self.actionQuit_slot)
        self.radioButton_2.toggled.connect(self.radioButton_2_slot)
        self.horizontalSlider.valueChanged.connect(self.valuechange_slot)
        self.pushButton_3.clicked.connect(self.goto_URL)
        self.webView.load(QUrl("http://selavo.lv/wiki/index.php/LU-pysem"))
        #.valueChanged.connect(self.valuechange)

    def actionQuit_slot(self):
        print('Going to Quit Seriously!')
        sys.exit(app.exec_())

    def radioButton_2_slot(self): #Notice how it also works when connected Radio Button 1 is toggled!
        print('Radio Button Toggled!')

    def valuechange_slot(self):
        print(f'Slider Value: {self.horizontalSlider.value()}')
        self.mysliderLabel.setText(f'Slider Value: {self.horizontalSlider.value()}')

    def goto_URL(self):
        print(f'going to URL: {self.lineEdit.text()}')
        self.webView.load(QUrl(self.lineEdit.text()))

if __name__ == '__main__':
    print('Starting qtui')
    app = QApplication(sys.argv)
    mpage = Example()
    mpage.show()
    sys.exit(app.exec_())
