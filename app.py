import sys
from PyQt6.QtWidgets import QApplication
from gui_app import LaserTurretApp

def main():
    app = QApplication(sys.argv)
    window = LaserTurretApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()