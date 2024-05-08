import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from load_preprocess import DataHandler
from models import ModelsHandler
from visualisations import Visualizer

LARGE_FONT = ("Verdana", 12)


class LiverDiseaseApp(tk.Tk):
    def __init__(self, master):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, "Liver Disease Prediction")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, DataAnalysisPage, ModelsPage, PredictPage, ValidationPage):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        load_data_button = ttk.Button(self, text="Load Data",
                                      command=lambda: controller.show_frame(DataAnalysisPage))
        load_data_button.pack()


class DataAnalysisPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Data Analysis Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        back_button = ttk.Button(self, text="Back to Home",
                                 command=lambda: controller.show_frame(StartPage))
        back_button.pack()


class ModelsPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Models Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        back_button = ttk.Button(self, text="Back to Home",
                                 command=lambda: controller.show_frame(StartPage))
        back_button.pack()


class PredictPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Predict Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        back_button = ttk.Button(self, text="Back to Home",
                                 command=lambda: controller.show_frame(StartPage))
        back_button.pack()


class ValidationPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Validation Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        back_button = ttk.Button(self, text="Back to Home",
                                 command=lambda: controller.show_frame(StartPage))
        back_button.pack()


if __name__ == "__main__":
    root = ThemedTk(theme='adapta', background='White')
    app = LiverDiseaseApp()
    app.mainloop()
