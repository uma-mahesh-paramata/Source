import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from load_preprocess import DataHandler
from models import ModelsHandler
from visualisations import Visualizer

class App:
    def __init__(self, master):
        self.master = master
        self.Analysis_window = None
        self.Validation_window = None
        self.prompt_window = None
        self.form_window = None
        self.dh = DataHandler()
        self.mh = ModelsHandler()
        self.vi = Visualizer()

        master.title("GUI")
        master.minsize(width=500, height=500)
        master.geometry("1250x700")

        container = tk.Frame(self.master)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}

        for F in (StartPage,DataAnalysisPage):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont, callback=None,caller=None):
        frame = self.frames[cont]
        if caller:
            frame.set_caller(caller)
        frame.tkraise()
        if callback:
            callback()

    def destroy(self):
        self.master.destroy()

    def load_data(self):
        self.text_widget.insert(tk.END, str(self.dh.load_data()))

    def open_analysis_window(self):
        if self.Analysis_window is None or not self.Analysis_window.winfo_exists():
            print(self.Analysis_window)
            self.Analysis_window = tk.Toplevel(self.master)
            self.Analysis_window.title("Data Analysis")
            self.Analysis_window.geometry('1200x800')
        else:
            self.Analysis_window.lift(aboveThis=self.master)
            return

        

    def open_models_window(self):
        if self.prompt_window is None or not self.prompt_window.winfo_exists():
            self.prompt_window = tk.Toplevel(self.master)
            self.prompt_window.title("Models")
            self.prompt_window.geometry('450x220')
            self.prompt_window.resizable(False, False)
        else:
            self.prompt_window.lift(aboveThis=self.master)
            return

        frame = ttk.Frame(self.prompt_window, padding=10)
        frame.pack(fill='both')

        # create the preprocess data checkbutton
        self.hyper_var = tk.BooleanVar()
        hyper_checkbutton = ttk.Checkbutton(frame, text="Hyper parameter tuning", variable=self.hyper_var,command=self.hide_widgets)
        hyper_checkbutton.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        
        # create the save test data checkbutton
        self.save_test_data_var = tk.BooleanVar()
        save_test_data_checkbutton = ttk.Checkbutton(frame, text="Split & Save Test Data", variable=self.save_test_data_var, command=self.hide_widgets)
        save_test_data_checkbutton.grid(row=1, column=0, padx=5, pady=5, sticky='w')
        
        # create the test data ratio label and entry field
        self.test_data_ratio_label = ttk.Label(frame, text="Test DataRatio (0-1):",state="disabled")
        self.test_data_ratio_entry = ttk.Entry(frame, state="disabled")

        # create frame for two buttons
        button_frame=ttk.Frame(self.prompt_window)
        button_frame.pack(fill='both',expand=True)
        
        # create the generate models button
        generate_models_button = ttk.Button(button_frame, text="Generate Models", command=self.generate_models_action)
        generate_models_button.pack(pady=15)

        #create the Load models button
        self.Load_models_button=ttk.Button(button_frame, text=" Load Models ", command=self.Load_models_action)

        self.Load_models_button.pack()
        
        
    def hide_widgets(self):
        if self.save_test_data_var.get():
            self.test_data_ratio_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')
            self.test_data_ratio_entry.config(state="normal")
            self.test_data_ratio_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')
            self.test_data_ratio_label.config(state="normal")
        else:
            self.test_data_ratio_entry.delete(0, "end")
            self.test_data_ratio_entry.grid_remove()
            self.test_data_ratio_label.grid_remove()
        if self.hyper_var.get() or self.save_test_data_var.get():
            self.Load_models_button.pack_forget()
        else:
            self.Load_models_button.pack()
        
    def Load_models_action(self):
        self.text_widget.insert(tk.END,str(self.mh.load_models()))
        self.prompt_window.destroy()

        
    def generate_models_action(self):
        # get the user input
        hyper_tuning = self.hyper_var.get()
        save_test_data = self.save_test_data_var.get()
        test_data_ratio = self.test_data_ratio_entry.get()

        if save_test_data:
            self.dh.split_data(test_data_ratio)
            self.text_widget.insert(tk.END,str((1-float(test_data_ratio))*100)+"% data is being used for training of models"+"\n\n")
        else:self.dh.split_data()

        self.text_widget.insert(tk.END,str(self.dh.preprocess(data_type='train'))+"\n\n")

        if hyper_tuning:self.mh.hyperparameter_tuning(self.dh.X_train,self.dh.y_train)
        self.text_widget.insert(tk.END,str(self.mh.generate_models(self.dh.X_train,self.dh.y_train)+"\n\n"))
        
        self.text_widget.insert(tk.END,str(self.mh.save_models())+"\n\n")
        # close the new window
        self.prompt_window.destroy()

    def open_predict_window(self):
        if self.form_window is None or not self.form_window.winfo_exists():
            self.form_window = tk.Toplevel(self.master)
            self.form_window.geometry("700x700")
            self.form_window.resizable(False, False)
        else:
            self.form_window.lift(aboveThis=self.master)
            return

        form_frame = ttk.Frame(self.form_window)
        tk.Label(form_frame, text="Age").grid(row=0, column=0, padx=5, pady=5)
        age_entry = ttk.Entry(form_frame)
        age_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(form_frame, text="Gender").grid(row=1, column=0, padx=5, pady=5)
        gender_var = tk.StringVar(form_frame)
        gender = ["Male", "Female"]
        gender_var.set("Male")
        gender_dropdown = tk.OptionMenu(form_frame, gender_var, *gender)
        gender_dropdown.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(form_frame, text="Total Bilirubin").grid(row=2, column=0, padx=5, pady=5)
        tbilirubin_entry = ttk.Entry(form_frame)
        tbilirubin_entry.grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(form_frame, text="Direct Bilirubin").grid(row=3, column=0, padx=5, pady=5)
        dbilirubin_entry = ttk.Entry(form_frame)
        dbilirubin_entry.grid(row=3, column=1, padx=5, pady=5)

        ttk.Label(form_frame, text="Alkaline Phosphatase").grid(row=4, column=0, padx=5, pady=5)
        alk_phosphatase_entry = ttk.Entry(form_frame)
        alk_phosphatase_entry.grid(row=4, column=1, padx=5, pady=5)

        ttk.Label(form_frame, text="Alanine Aminotransferase").grid(row=5, column=0, padx=5, pady=5)
        alam_trans_entry = ttk.Entry(form_frame)
        alam_trans_entry.grid(row=5, column=1)

        ttk.Label(form_frame, text="Aspartate Aminotransferase").grid(row=6, column=0, padx=5, pady=5)
        asp_trans_entry = ttk.Entry(form_frame)
        asp_trans_entry.grid(row=6, column=1, padx=5, pady=5)

        ttk.Label(form_frame, text="Total Protiens").grid(row=7, column=0, padx=5, pady=5)
        total_prot_entry = ttk.Entry(form_frame)
        total_prot_entry.grid(row=7, column=1, padx=5, pady=5)

        ttk.Label(form_frame, text="Albumin").grid(row=8, column=0, padx=5, pady=5)
        albumin_entry = ttk.Entry(form_frame)
        albumin_entry.grid(row=8, column=1, padx=5, pady=5)

        ttk.Label(form_frame, text="Albumin and Globulin Ratio").grid(row=9, column=0, padx=5, pady=5)
        albumin_globulin_entry = ttk.Entry(form_frame)
        albumin_globulin_entry.grid(row=9, column=1, padx=5, pady=5)

        def focus_next_entry(event,index):
            if index<len(widgets)-1:
                widgets[index+1].focus()
            return "break"
        
        def focus_prev_entry(event, index):
            if index > 0:
                widgets[index-1].focus()
            return "break"
        
        widgets = [  age_entry,    gender_dropdown,    tbilirubin_entry,    dbilirubin_entry,    alk_phosphatase_entry,    alam_trans_entry,    asp_trans_entry,    total_prot_entry,    albumin_entry,    albumin_globulin_entry ]

        for i in range(len(widgets)):
            widget = widgets[i]
            widget.bind("<Return>", lambda event, index=i: focus_next_entry(event,index))
            widget.bind("<Up>", lambda event, index=i: focus_prev_entry(event, index))
            widget.bind("<Down>", lambda event, index=i: focus_next_entry(event, index))


        output_label=ttk.Label(self.form_window,font=("times", 14))

        def submit():
            self.predict_action( age_entry.get(), gender_var, tbilirubin_entry.get(), dbilirubin_entry.get(), alk_phosphatase_entry.get(), alam_trans_entry.get(), asp_trans_entry.get(), total_prot_entry.get(), albumin_entry.get(), albumin_globulin_entry.get(), output_label)
        
        form_frame.pack()
        submit_button=ttk.Button(self.form_window,text="Submit",command=submit)
        submit_button.pack( padx=5, pady=10)
        output_label.pack( padx=5, pady=10)
        

        
        
    
    def predict_action(self, age, gender, tb, db, ap, aa, sg, tp, al, alb, output_label):
        try:
            age = int(age)
            gender = gender.get()
            tb = float(tb)
            db = float(db)
            ap = int(ap)
            aa = int(aa)
            sg = float(sg)
            tp = float(tp)
            al = float(al)
            alb = float(alb)
        except ValueError:
            output_label.config(text="Please enter valid input data types")
            return
        self.dh.single_test=[[age, gender, tb, db, ap, aa, sg, tp, al, alb]]
        self.dh.preprocess(data_type='single')
        self.mh.predict(self.dh.single_test)
        prediction=str(self.mh.predictions.values())
        output_label.config(text="Predicted output: {}".format(prediction))

    def open_validation_window(self):
        self.dh.load_testdata()
        self.dh.preprocess(data_type='test')
        self.mh.predict(self.dh.X_test)
        self.text_widget.insert(tk.END,str(self.mh.validation(self.dh.y_test)))
        if self.Validation_window is None or not self.Validation_window.winfo_exists():
            self.Validation_window = tk.Toplevel(self.master)
            self.Validation_window.title("Validation")
            self.Validation_window.geometry('1200x800')
        else:
            self.Validation_window.lift(aboveThis=self.master)
            return

        graphs = self.vi.create_validation_graphs(self.mh.validation_scores)

        graph_type_var = tk.StringVar(self.Validation_window)
        graph_type_var.set("plot_1")
        graph_type_options = ["plot_1", "plot_2", "plot_3"]
        graph_type_dropdown = tk.OptionMenu(self.Validation_window, graph_type_var, *graph_type_options)
        graph_type_dropdown.pack()

        self.vi.visualize(self.Validation_window, graph_type_var, graphs)

class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        # label = tk.Label(self, text="Start Page", font=('times', 21, 'bold'))
        # label.pack(pady=10, padx=10)

        # load_data_button = ttk.Button(self, text="Load Data",
        #                               command=lambda: controller.show_frame(DataAnalysisPage))
        # load_data_button.pack()
        self.controller=controller

        title_frame = ttk.Frame(self, padding=10)
        title_frame.pack(side='top', fill='x')
        title = ttk.Label(title_frame, text='LIVER DISEASE PREDICTION', font=('times', 21, 'bold'))
        title.place(x=0, y=5)
        title.pack()

        text_frame = ttk.Frame(self, height=30, width=150)
        text_frame.pack()

        import tkinter.font as tkFont
        font = tkFont.Font(family="Courier new", size=13)
        self.text_widget = tk.Text(text_frame, font=font, wrap='word', state='normal', height=20, width=120)
        self.text_widget.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        scrollb = ttk.Scrollbar(text_frame, command=self.text_widget.yview)
        self.text_widget.config(yscrollcommand=scrollb.set)
        scrollb.grid(row=0, column=1, sticky='nsew')

        frame = ttk.Frame(self)
        frame.pack(fill='both', expand=True)
        button_frame = ttk.Frame(frame)
        style = ttk.Style()
        style.configure("Custom.TButton", font=("times", 14))

        load_data_button = ttk.Button(button_frame, text="Load Data", command=lambda: self.load_data(), style="Custom.TButton")
        data_analysis_button = ttk.Button(button_frame, text="Data Analysis", command=lambda: controller.show_frame(DataAnalysisPage,caller=self), style="Custom.TButton")
        generate_models_button = ttk.Button(button_frame, text="Generate Models", command=lambda: controller.show_frame(StartPage), style="Custom.TButton")
        predict_data_button = ttk.Button(button_frame, text="Predict", command=lambda: controller.show_frame(StartPage), style="Custom.TButton")
        model_validation_button = ttk.Button(button_frame, text="Model Validation", command=lambda: controller.show_frame(StartPage), style="Custom.TButton")
        Exit_button = ttk.Button(button_frame, text="Exit", command=lambda: controller.destroy(), style="Custom.TButton")

        load_data_button.grid(row=0, column=0, sticky='ew', padx=15, pady=15)
        data_analysis_button.grid(row=0, column=1, sticky='ew', padx=15, pady=15)
        generate_models_button.grid(row=0, column=2, sticky='ew', padx=15, pady=15)
        predict_data_button.grid(row=1, column=0, sticky='ew', padx=15, pady=15)
        model_validation_button.grid(row=1, column=1, sticky='ew', padx=15, pady=15)
        Exit_button.grid(row=1, column=2, sticky='ew', padx=15, pady=15)
        button_frame.pack()

    def load_data(self):
        self.text_widget.insert(tk.END, str(self.dh.load_data()))

class DataAnalysisPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        # label = tk.Label(self, text="Data Analysis Page", font=LARGE_FONT)
        # label.pack(pady=10, padx=10)
        self.controller=controller
        button_frame = ttk.Frame(self)
        style = ttk.Style()
        style.configure("Custom.TButton", font=("times", 14))

        back_button = ttk.Button(button_frame, text="Back to Home",
                                 command=lambda: controller.show_frame(StartPage))
        back_button.pack()

        

        controller.frames['Startpage'].text_widget.insert(tk.END, str(self.dh.df.describe()) + "\n\n")
        controller.text_widget.insert(tk.END, "No.of null values in each column\n\n" + str(self.dh.df.isnull().sum()) + "\n\n")

        graph_type_var = tk.StringVar(button_frame)
        graph_type_var.set("plot_1")
        graph_type_options = ["plot_1", "plot_2", "plot_3", "plot_4", "plot_5", "plot_6"]
        graph_type_dropdown = tk.OptionMenu(button_frame, graph_type_var, *graph_type_options)
        graph_type_dropdown.pack()

        
    def process_graphs(self, graph_type_var):
        graphs = self.controller.vi.create_analysis_graphs(self.caller.dh.df)
        self.caller.vi.visualize(self, graph_type_var, graphs)

    def set_caller(self,caller):
        self.caller=caller
root = ThemedTk(theme='adapta', background='White')
app = App(root)
root.mainloop()
