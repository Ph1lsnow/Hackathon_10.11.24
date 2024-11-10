import sys
import time

import pyedflib
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout,
    QWidget, QFileDialog, QMessageBox, QToolTip, QHBoxLayout, QFormLayout, QSlider, QLabel, QDoubleSpinBox,
    QProgressDialog, QDialog
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QCursor, QMovie
from pyqtgraph import PlotWidget, mkPen, setConfigOptions, LinearRegionItem
import numpy as np
import copy
from AIka import func_with_nn

stackk = {'swd': 'Фаза эпилептического разряда', 'ds': 'Фаза глубокого сна', 'is': 'Фаза промежуточного сна', 'an': 'Аномалии'}

# Класс EDFHandler с исправленным методом read_file
class EDFHandler:
    def __init__(self):
        self.metadata = {}
        self.channels_info = []
        self.data = []
        self.annotations = []
        self.original_file = None

    def read_file(self, filepath):
        """
        Читает EDF/EDF+ файл и инициализирует атрибуты класса.

        :param filepath: Путь к EDF файлу
        """
        self.original_file = filepath
        with pyedflib.EdfReader(filepath) as edf:
            # Чтение основной информации
            self.metadata = edf.getHeader()

            # Чтение информации о каналах
            self.channels_info = edf.getSignalHeaders()

            # Чтение данных сигналов
            n_signals = edf.signals_in_file
            self.data = []
            for i in range(n_signals):
                signal = edf.readSignal(i)
                self.data.append(signal)

            # Чтение аннотаций, если они есть
            try:
                onsets, durations, descriptions = edf.readAnnotations()
                self.annotations = list(zip(onsets, durations, descriptions))
            except:
                self.annotations = []

    def modify_metadata(self, new_metadata):
        """
        Модифицирует основную информацию о EDF файле.

        :param new_metadata: Словарь с новыми метаданными
        """
        if not isinstance(new_metadata, dict):
            raise ValueError("new_metadata должен быть словарем.")
        self.metadata.update(new_metadata)

    def modify_channel_info(self, channel_index, new_info):
        """
        Модифицирует информацию о конкретном канале.

        :param channel_index: Индекс канала (начинается с 0)
        :param new_info: Словарь с новыми метаданными для канала
        """
        if not (0 <= channel_index < len(self.channels_info)):
            raise IndexError("Индекс канала вне диапазона.")
        if not isinstance(new_info, dict):
            raise ValueError("new_info должен быть словарем.")
        self.channels_info[channel_index].update(new_info)

    def set_annotations(self, new_annotations):
        """
        Заменяет все существующие аннотации на новые.

        :param new_annotations: Список кортежей (onset, duration, description)
        """
        if not isinstance(new_annotations, list):
            raise ValueError("new_annotations должен быть списком кортежей.")
        for annotation in new_annotations:
            if not (isinstance(annotation, tuple) and len(annotation) == 3):
                raise ValueError(
                    "Каждая аннотация должна быть кортежем из трех элементов: (onset, duration, description).")
        self.annotations = new_annotations

    def save_file(self, save_path):
        """
        Сохраняет изменения в новый EDF/EDF+ файл.

        :param save_path: Путь для сохранения нового EDF файла
        """
        n_channels = len(self.channels_info)
        data_np = np.array(self.data)
        with pyedflib.EdfWriter(save_path, n_channels=n_channels, file_type=pyedflib.FILETYPE_EDFPLUS) as writer:
            writer.setHeader(self.metadata)
            writer.setSignalHeaders(self.channels_info)
            writer.writeSamples(data_np)
            for annotation in self.annotations:
                onset, duration, description = annotation
                writer.writeAnnotation(onset, duration, description)

    def get_metadata(self):
        return copy.deepcopy(self.metadata)

    def get_channels_info(self):
        return copy.deepcopy(self.channels_info)

    def get_data(self):
        return copy.deepcopy(self.data)

    def get_annotations(self):
        return copy.deepcopy(self.annotations)

# Настройка pyqtgraph
setConfigOptions(useOpenGL=True)
setConfigOptions(antialias=False)
setConfigOptions(background='w', foreground='k')


class EEGPlot:
    def __init__(self, parent=None, app=None):
        """
        Initialize the EEGPlot.

        :param parent: Parent widget.
        :param app: Reference to the main EEGApp to access key states.
        """
        self.app = app  # Reference to the main application
        # Create the plot widget
        self.plot_widget = PlotWidget(parent)
        self.plot_widget.setDownsampling(mode='peak')
        self.plot_widget.setClipToView(True)
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.enableAutoRange('xy', False)

        # Connect the mouse wheel event
        self.plot_widget.wheelEvent = self.wheel_event

        # **Добавлено: Подключение обработчика события перемещения мыши**
        self.plot_widget.setMouseTracking(True)
        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_moved)

    def plot_data(self, time_axis, signal):
        # Plot data on the graph
        self.plot_widget.plot(
            time_axis,
            signal,
            pen=mkPen(color='b', width=1),
            downsampleMethod='peak',
            autoDownsample=True,
            clear=True  # Добавлено для очистки предыдущих данных при повторном построении
        )

    def clear(self):
        # Clear the plot
        self.plot_widget.clear()

    def set_x_range(self, x_min, x_max):
        # Set the X-axis range with no padding
        self.plot_widget.setXRange(x_min, x_max, padding=0)

    def set_y_range(self, y_min, y_max):
        # Set the Y-axis range with no padding
        self.plot_widget.setYRange(y_min, y_max, padding=0)

    def set_x_link(self, other_plot):
        # Link the X-axis to another plot
        self.plot_widget.setXLink(other_plot.plot_widget)

    def wheel_event(self, event):
        """
        Handle the mouse wheel event.

        - Ctrl + Z + Wheel: Zoom X-axis
        - Ctrl + Wheel: Zoom Y-axis
        - Wheel: Scroll X-axis
        """
        # Check the current key states from the main application
        ctrl_pressed = self.app.ctrl_pressed if self.app else False
        z_pressed = self.app.z_pressed if self.app else False

        if ctrl_pressed and z_pressed:
            # Zoom on the X-axis
            factor = 0.9 if event.angleDelta().y() > 0 else 1.1  # Determine zoom direction
            x_min, x_max = self.plot_widget.viewRange()[0]
            center_x = (x_max + x_min) / 2
            new_width = (x_max - x_min) * factor / 2
            new_x_min = center_x - new_width
            new_x_max = center_x + new_width
            self.plot_widget.setXRange(new_x_min, new_x_max, padding=0)
        elif ctrl_pressed:
            # Zoom on the Y-axis
            factor = 0.9 if event.angleDelta().y() > 0 else 1.1  # Determine zoom direction
            y_min, y_max = self.plot_widget.viewRange()[1]
            center_y = (y_max + y_min) / 2
            new_y_min = center_y - (center_y - y_min) * factor
            new_y_max = center_y + (y_max - center_y) * factor
            self.plot_widget.setYRange(new_y_min, new_y_max, padding=0)
        else:
            # Scroll on the X-axis when Ctrl is not pressed
            delta = event.angleDelta().y() / 120  # Each wheel "click"
            step = 1  # Scroll step on the X-axis

            # Get the current visible X-axis range
            x_min, x_max = self.plot_widget.viewRange()[0]

            # Calculate the new X-axis range
            shift = step * delta
            new_x_min = x_min + shift
            new_x_max = x_max + shift

            # Set the new X-axis range
            self.plot_widget.setXRange(new_x_min, new_x_max, padding=0)

    # **Добавлено: Обработчик события перемещения мыши**
    def on_mouse_moved(self, point):
        # Convert the point to plot coordinates
        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(point)
        x = mouse_point.x()

        if self.app:
            flag, element, predict = self.app.is_cursor_over_label(x)
            if flag:
                if predict != -1:
                    QToolTip.showText(QCursor.pos(), stackk[element] + f"  {int(predict*100)}%")
                else:
                    QToolTip.showText(QCursor.pos(), stackk[element])
        else:
            QToolTip.hideText()


class LabelRegion:
    def __init__(self, start_time, end_time, plots, label_class):
        self.start_time = start_time
        self.end_time = end_time
        self.plots = plots  # list of EEGPlot instances
        self.label_class = label_class
        self.regions = []
        self.color = self.get_color(label_class)
        self.updating = False  # Flag to prevent recursive updates

        # Create regions on each plot and connect signals
        for plot in self.plots:
            region = LinearRegionItem([self.start_time, self.end_time], brush=self.color, movable=False)
            region.setZValue(10)
            plot.plot_widget.addItem(region)
            self.regions.append(region)

        # Connect signals after all regions are created
        for region in self.regions:
            region.sigRegionChanged.connect(self.region_changed)

    def get_color(self, label_class):
        colors = {'swd': (255, 0, 0, 50), 'ds': (0, 255, 0, 50), 'is': (0, 0, 255, 50), 'an': (3, 12, 255, 50)}
        return colors.get(label_class, (128, 128, 128, 50))

    def region_changed(self, region):
        if self.updating:
            return
        self.updating = True

        # Get the new region boundaries
        minX, maxX = region.getRegion()

        # Update all regions to the new boundaries
        for reg in self.regions:
            if reg != region:
                reg.blockSignals(True)  # Prevent signals during update
                reg.setRegion([minX, maxX])
                reg.blockSignals(False)

        # Update the start_time и end_time
        self.start_time = minX
        self.end_time = maxX

        self.updating = False

    def set_movable(self, movable):
        for region in self.regions:
            region.setMovable(movable)


class EEGApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("EEG Analyzer")
        self.setGeometry(100, 100, 800, 800)

        # Инициализация обработчика EDF
        self.edf_handler = EDFHandler()

        # Initialize interface
        self.initUI()

        # Keep track of whether Shift is pressed
        self.shift_pressed = False

        # Track Ctrl and Z key states
        self.ctrl_pressed = False
        self.z_pressed = False

    def initUI(self):
        # Main widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        button_layout = QHBoxLayout()
        settings_layout = QFormLayout()

        # Create three plots using the EEGPlot class
        self.plot1 = EEGPlot(self, app=self)
        self.plot2 = EEGPlot(self, app=self)
        self.plot3 = EEGPlot(self, app=self)

        layout.addWidget(self.plot1.plot_widget)
        layout.addWidget(self.plot2.plot_widget)
        layout.addWidget(self.plot3.plot_widget)

        # Synchronize the X-axis of the plots
        self.plot2.set_x_link(self.plot1)
        self.plot3.set_x_link(self.plot1)

        layout.addLayout(button_layout)

        # Import data button
        self.import_button = QPushButton("Импорт данных", self)
        self.import_button.clicked.connect(self.import_data)
        button_layout.addWidget(self.import_button)

        # Save data button
        self.save_button = QPushButton("Сохранить EDF", self)
        self.save_button.clicked.connect(self.save_data)
        button_layout.addWidget(self.save_button)

        # Export annotations to txt button
        self.export_txt_button = QPushButton("Выгрузить аннотации в TXT", self)
        self.export_txt_button.clicked.connect(self.export_annotations_to_txt)
        button_layout.addWidget(self.export_txt_button)

        # **Settings Block**
        # 1. Уровень уверенности
        self.confidence_slider = QSlider(Qt.Horizontal, self)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider_label = QLabel(f"Порог уверенности (50%)")
        self.confidence_slider.valueChanged.connect(
            lambda value: self.confidence_slider_label.setText(f"Порог уверенности ({value:^3}%)")
        )
        settings_layout.addRow(self.confidence_slider_label, self.confidence_slider)

        # 2. Минимальное время между фазами
        self.min_time_between_phases = QDoubleSpinBox(self)
        self.min_time_between_phases.setRange(0.1, 10000.0)
        self.min_time_between_phases.setValue(0.5)
        self.min_time_between_phases.setSingleStep(0.1)
        settings_layout.addRow(QLabel("Минимальное время между фазами (сек.)"), self.min_time_between_phases)

        # 3. Минимальная длина фаз
        self.min_phase_length = QDoubleSpinBox(self)
        self.min_phase_length.setRange(0.1, 10000.0)
        self.min_phase_length.setValue(0.5)
        self.min_phase_length.setSingleStep(0.1)
        settings_layout.addRow(QLabel("Минимальная длина фаз (сек.)"), self.min_phase_length)

        # 4. Минимальное не аномальное время между фазами
        self.min_non_anomalous_time = QDoubleSpinBox(self)
        self.min_non_anomalous_time.setRange(0.1, 10000.0)
        self.min_non_anomalous_time.setValue(1.0)
        self.min_non_anomalous_time.setSingleStep(0.1)
        settings_layout.addRow(QLabel("Минимальное не аномальное время между фазами (сек.)"), self.min_non_anomalous_time)

        # 5. Минимальный перепад для аномалий
        self.min_anomaly_gradient = QSlider(Qt.Horizontal, self)
        self.min_anomaly_gradient.setRange(0, 100)
        self.min_anomaly_gradient.setValue(50)
        self.min_anomaly_gradient_label = QLabel(f"Минимальный перепад при определении аномалий (50%)")
        self.min_anomaly_gradient.valueChanged.connect(
            lambda value: self.min_anomaly_gradient_label.setText(f"Минимальный перепад при определении аномалий ({value:^3}%)")
        )

        settings_layout.addRow(self.min_anomaly_gradient_label, self.min_anomaly_gradient)

        layout.addLayout(settings_layout)

        # AI-helper button below the settings block
        self.ai_helper_button = QPushButton("AI-helper", self)
        self.ai_helper_button.clicked.connect(self.ai_helper)
        layout.addWidget(self.ai_helper_button)

        # Store data
        self.data = None
        self.labels = {'swd': [], 'ds': [], 'is': [], 'an': []}
        self.label_regions = []

    def export_annotations_to_txt(self):
        """
        Export annotations to a .txt file in the format:
        NN    время    маркер
        """
        if not self.edf_handler.get_annotations():
            QMessageBox.warning(self, "Предупреждение", "Нет аннотаций для экспорта.")
            return

        # Open file dialog to select save location
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить аннотации в TXT файл", "", "Text Files (*.txt)"
        )
        if save_path:
            try:
                annotations = self.edf_handler.get_annotations()
                with open(save_path, 'w') as file:
                    file.write("NN\tвремя\tмаркер\n")
                    for idx, (onset, _, description) in enumerate(annotations, start=1):
                        # Convert onset time (in seconds) to HH:MM:SS format
                        hours = int(onset // 3600)
                        minutes = int((onset % 3600) // 60)
                        seconds = int(onset % 60)
                        formatted_time = f"{hours:1}:{minutes:02}:{seconds:02}"

                        # Write line in the specified format
                        file.write(f"{idx}\t{formatted_time}\t{description}\n")
                QMessageBox.information(self, "Успех", f"Аннотации успешно сохранены в файл: {save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить аннотации: {e}")

    def import_data(self):
        # Open EDF file
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл EDF", "", "EDF Files (*.edf)"
        )
        if file_path:
            try:
                self.edf_handler.read_file(file_path)
                self.process_edf_data()
                self.plot_data()
                QMessageBox.information(self, "Успех", f"Файл '{file_path}' успешно импортирован.")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось импортировать файл: {e}")

    def ai_helper(self):
        if self.edf_handler.original_file is None:
            QMessageBox.critical(self, "Ошибка", f"Файл не был импортиртирован")
        else:
            # Create and set up the loading dialog
            loading_dialog = QDialog()
            layout = QVBoxLayout()
            loading_label = QLabel("Processing...")

            layout.addWidget(loading_label)
            loading_dialog.setLayout(layout)

            # Show the loading dialog
            loading_dialog.show()
            QApplication.processEvents()  # Update the UI to show the loading dialog

            try:
                # Run the AI processing
                self.process_edf_data_form_ai()
                self.plot_data()
                QMessageBox.information(self, "Успех", f"Файл был обработан \n Найдено swd: {len(self.labels['swd'])}, ds: {len(self.labels['ds'])}, is: {len(self.labels['is'])}, an: {len(self.labels['an'])}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Произошла ошибка: {str(e)}")
            finally:
                # Close the loading dialog once processing is complete
                loading_dialog.close()

    def process_edf_data(self):
        # Получаем данные из EDFHandler
        self.data = self.edf_handler.get_data()
        # Получение частоты дискретизации из метаданных
        # Частота дискретизации может быть разной для разных каналов, но для простоты используем первый канал
        self.sample_rate = self.edf_handler.metadata.get('sample_rate', 256)  # Предположим 256 Гц по умолчанию
        self.signal_length = len(self.data[0]) / self.sample_rate  # in seconds

        # Читаем аннотации
        annotations = self.edf_handler.get_annotations()

        # Обрабатываем аннотации в интервалы
        start_labels = {'swd1': 'swd', 'ds1': 'ds', 'is1': 'is', 'an1': 'an'}
        end_labels = {'swd2': 'swd', 'ds2': 'ds', 'is2': 'is', 'an2': 'an'}
        stacks = {'swd': [], 'ds': [], 'is': [], 'an': []}

        for annotation in annotations:
            if len(annotation) != 3:
                continue  # Пропустить аннотации с неправильной структурой
            onset, duration, description = annotation
            description = description.strip()
            if description in start_labels:
                cls = start_labels[description]
                stacks[cls].append(onset)
            elif description in end_labels:
                cls = end_labels[description]
                if stacks[cls]:
                    start_time = stacks[cls].pop()
                    end_time = onset
                    self.labels[cls].append((start_time, end_time, -1))

    def process_edf_data_form_ai(self):
        # Получаем данные из EDFHandler
        self.data = self.edf_handler.get_data()
        # Получение частоты дискретизации из метаданных
        # Частота дискретизации может быть разной для разных каналов, но для простоты используем первый канал
        self.sample_rate = self.edf_handler.metadata.get('sample_rate', 256)  # Предположим 256 Гц по умолчанию
        self.signal_length = len(self.data[0]) / self.sample_rate  # in seconds

        # Генерируем/Читаем аннотации
        data_annotations: list[tuple[float, int, str, int]] = func_with_nn(self.edf_handler.original_file, self.confidence_slider.value() / 100, self.min_time_between_phases.value(), self.min_phase_length.value(), self.min_non_anomalous_time.value(), self.min_anomaly_gradient.value())
        data_annotations_without_predict = [(i[0], i[1], i[2]) for i in data_annotations]
        self.edf_handler.set_annotations(data_annotations_without_predict)
        annotations = data_annotations.copy()

        # Обрабатываем аннотации в интервалы
        start_labels = {'swd1': 'swd', 'ds1': 'ds', 'is1': 'is', 'an1': 'an'}
        end_labels = {'swd2': 'swd', 'ds2': 'ds', 'is2': 'is', 'an2': 'an'}
        stacks = {'swd': [], 'ds': [], 'is': [], 'an': []}

        for annotation in annotations:
            if len(annotation) != 4:
                continue  # Пропустить аннотации с неправильной структурой
            onset, duration, description, predict = annotation
            description = description.strip()
            if description in start_labels:
                cls = start_labels[description]
                stacks[cls].append([onset, predict])
            elif description in end_labels:
                cls = end_labels[description]
                if stacks[cls]:
                    start_time, predict = stacks[cls].pop()
                    end_time = onset
                    self.labels[cls].append((start_time, end_time, predict))

    def plot_data(self):
        # Display data on plots
        if self.data and len(self.data) >= 3:
            # Clear existing plots
            self.plot1.clear()
            self.plot2.clear()
            self.plot3.clear()

            # Time axis for the signals
            time_axis = np.linspace(0, self.signal_length, len(self.data[0]))

            # Plot signals on respective plots
            self.plot1.plot_data(time_axis, self.data[0])
            self.plot2.plot_data(time_axis, self.data[1])
            self.plot3.plot_data(time_axis, self.data[2])

            # Set initial view range for the first plot and synchronize X for others
            self.plot1.set_x_range(0, min(10, self.signal_length))  # Show first 10 seconds
            self.plot1.set_y_range(np.min(self.data[0]), np.max(self.data[0]))
            self.plot2.set_y_range(np.min(self.data[1]), np.max(self.data[1]))
            self.plot3.set_y_range(np.min(self.data[2]), np.max(self.data[2]))

            # Create a list to store LabelRegion instances
            self.label_regions = []

            # For each label class
            for label_class, intervals in self.labels.items():
                for start_time, end_time, predict in intervals:
                    label_region = LabelRegion(
                        start_time, end_time, [self.plot1, self.plot2, self.plot3], label_class
                    )
                    self.label_regions.append(label_region)

            # Set initial movable state based on Shift key
            for label_region in self.label_regions:
                label_region.set_movable(self.shift_pressed)

    def keyPressEvent(self, event):
        """
        Handle key press events to track Ctrl and Z keys.
        """
        if event.key() == Qt.Key_Shift:
            self.shift_pressed = True
            # Set regions to movable
            for label_region in self.label_regions:
                label_region.set_movable(True)
        elif event.key() == Qt.Key_Control:
            self.ctrl_pressed = True
        elif event.key() == Qt.Key_Z:
            self.z_pressed = True
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """
        Handle key release events to update Ctrl and Z key states.
        """
        if event.key() == Qt.Key_Shift:
            self.shift_pressed = False
            # Set regions to not movable
            for label_region in self.label_regions:
                label_region.set_movable(False)
        elif event.key() == Qt.Key_Control:
            self.ctrl_pressed = False
        elif event.key() == Qt.Key_Z:
            self.z_pressed = False
        super().keyReleaseEvent(event)

    def save_data(self):
        """
        Сохранить текущие данные в новый EDF файл.
        """
        if not self.data:
            QMessageBox.warning(self, "Предупреждение", "Нет данных для сохранения.")
            return

        # Диалог выбора места сохранения файла
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить EDF файл", "", "EDF Files (*.edf)"
        )
        if save_path:
            try:
                # Обновим аннотации из текущих LabelRegion
                self.update_annotations_from_regions()
                # Сохраним файл через EDFHandler
                self.edf_handler.save_file(save_path)
                QMessageBox.information(self, "Успех", f"Файл успешно сохранен по пути: {save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл: {e}")

    def update_annotations_from_regions(self):
        """
        Обновить аннотации в EDFHandler на основе текущих LabelRegion.
        """
        new_annotations = []
        for label_region in self.label_regions:
            # Start point
            onset = label_region.start_time
            duration = -1
            description = label_region.label_class + "1"
            new_annotations.append((onset, duration, description))
            # End point
            onset = label_region.end_time
            duration = -1
            description = label_region.label_class + "2"
            new_annotations.append((onset, duration, description))
        self.edf_handler.set_annotations(new_annotations)

    # **Добавлено: Метод для проверки нахождения курсора над меткой**
    def is_cursor_over_label(self, x):
        """
        Проверяет, находится ли x-координата курсора внутри любой из меток.

        :param x: X-координата курсора в секундах.
        :return: True, если курсор над меткой, иначе False.
        """
        for label_class, intervals in self.labels.items():
            for start_time, end_time, predict in intervals:
                if start_time <= x <= end_time:
                    return True, label_class, predict
        return False, "", -1


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EEGApp()
    window.show()
    sys.exit(app.exec_())
