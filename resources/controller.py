"""
Расчётный модуль
"""

import os
import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from resources.calculation import main_calc


class UI(QtWidgets.QMainWindow):
    """
    Класс для запуска и обработки главного окна

    QtWidgets.QMainWindow: Виджет главного окна
    """

    def __init__(self):
        """
        Метод инициализации окна
        """
        # Инициализация окна
        super(UI, self).__init__()
        # Подгрузка UI файла (bundle_dir требуется для корректной работы
        bundle_dir = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
        uic.loadUi(os.path.join(bundle_dir, 'GUI\\GUI.ui'), self)
        self.show()

        # Поиск полей для путей для файлов
        self.path_binary = self.findChild(QtWidgets.QLineEdit, 'binary_file_box')
        self.path_composition = self.findChild(QtWidgets.QLineEdit, 'components_file_box')

        # Соединение кнопок поиска путей
        self.binary_button = self.findChild(QtWidgets.QPushButton, 'binary_file_but')
        self.binary_button.clicked.connect(self.file_input)
        self.composition_but = self.findChild(QtWidgets.QPushButton, 'components_file_but')
        self.components_file_but.clicked.connect(self.file_input)

        # Соединение кнопки расчёта
        self.calc_button = self.findChild(QtWidgets.QPushButton, 'calculate_button')
        self.calc_button.clicked.connect(self.start_calculation)

    def start_calculation(self):
        """
        Обработчик кнопки запуска расчёта

        :return: String сообщение во фронтенд с результатом работы
        """
        save_path = QFileDialog.getExistingDirectory(self, "Select Directory")

        # Запуск проверок данных на корректность
        try:
            pressure = float(self.pressure_box.text().replace(",", "."))
            try:
                temperature = float(self.temperature_box.text().replace(",", "."))
                if temperature < 0:
                    error_dialog = QtWidgets.QErrorMessage()
                    error_dialog.showMessage('Введено значение температуры меньше абсолютного нуля!')
                    error_dialog.exec_()
                else:
                    if not os.path.exists(self.path_binary.text()):
                        error_dialog = QtWidgets.QErrorMessage()
                        error_dialog.showMessage('Введён некорректный путь до файла бинарного распределения!')
                        error_dialog.exec_()
                    elif not os.path.exists(self.path_composition.text()):
                        error_dialog = QtWidgets.QErrorMessage()
                        error_dialog.showMessage('Введён некорректный путь до файла состава!')
                        error_dialog.exec_()
                    elif not os.path.exists(save_path):
                        error_dialog = QtWidgets.QErrorMessage()
                        error_dialog.showMessage('Введён некорректный путь до папки сохранения!')
                        error_dialog.exec_()
                    else:
                        msgBox = QMessageBox()
                        msgBox.setText(main_calc.main(self.path_binary.text(), self.path_composition.text(),
                                                      pressure, temperature, save_path))
                        msgBox.exec_()

            except ValueError:
                error_dialog = QtWidgets.QErrorMessage()
                error_dialog.showMessage('Введено неправильное значение температуры!')
                error_dialog.exec_()
        except ValueError:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage('Введено неправильное значение давления!')
            error_dialog.exec_()

    def file_input(self):
        """
        Обработчик кнопок 'Обзор'

        :return: Запись значений пути в соответствующите поля
        """

        button = self.sender()

        if button.objectName() == "binary_file_but":
            # 0 необходим для возврата именно пути файла
            self.path_binary.setText(QFileDialog.getOpenFileName(self,
                                                                 'Open a file', '', 'Excel Files (*.xlsx)')[0])
        else:
            self.path_composition.setText(QFileDialog.getOpenFileName(self,
                                                                      'Open a file', '', 'Excel Files (*.xlsx)')[0])


def main():
    """
    Функция запуска прилоения и инициализация главного окна
    :return: не возвращает значений
    """
    app = QtWidgets.QApplication(sys.argv)
    window = UI()
    window.show()
    app.exec_()
