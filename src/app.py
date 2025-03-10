import sys
import numpy as np
import tensorflow as tf 
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog

class DeepLearningApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("App de Deep Learning")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        self.label = QLabel("Escolha um arquivo para processar", self)
        self.button = QPushButton("Selecionar Arquivo", self)
        self.button_run = QPushButton("Rodar Modelo", self)

        self.button.clicked.connect(self.select_file)
        self.button_run.clicked.connect(self.run_model)

        layout.addWidget(self.label)
        layout.addWidget(self.button)
        layout.addWidget(self.button_run)
        self.setLayout(layout)

        self.selected_file = None 

    def select_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Selecionar Arquivo", "", "Todos os Arquivos (*)")
        if file_path:
            self.selected_file = file_path
            self.label.setText(f"Arquivo: {file_path}")

    def run_model(self):
        if not self.selected_file:
            self.label.setText("Selecione um arquivo primeiro!")
            return

        result = self.process_model(self.selected_file)
        self.label.setText(f"Resultado: {result}")

    def process_model(self, file_path):
        random_input = np.random.rand(1, 10)
        model = tf.keras.Sequential([tf.keras.layers.Dense(1, activation="sigmoid")])
        model.compile(optimizer="adam", loss="binary_crossentropy")
        result = model.predict(random_input)
        
        return f"{result[0][0]:.4f}"

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeepLearningApp()
    window.show()
    sys.exit(app.exec_())
