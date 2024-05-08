import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import ImageTk, Image
import os
import pandas as pd

class Visualizer:
    def __init__(self):
        pass

    def _create_directory(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Directory '{directory}' created successfully.")
            else:
                print(f"Directory '{directory}' already exists.")
        except Exception as e:
            print("An error occurred while creating the directory:", e) 

    def create_analysis_graphs(self, df):
        self._create_directory('./plots')
        
        sns.set(font_scale=0.6)                                                                     
        df_encoded = pd.get_dummies(df)  
        corr = df_encoded.corr()
        corr_matrix = sns.heatmap(corr, cmap="Greens", annot=True)
        corr_path = os.path.join("plots", "correlation_matrix.png")
        corr_matrix.figure.savefig(corr_path, bbox_inches='tight')

        sns.set()
        count_fig = plt.figure()
        count_ax = count_fig.add_subplot(111)
        sns.countplot(data=df, x="dataset", ax=count_ax, width=0.5)
        count_path = os.path.join("plots", "countplot.png")
        count_fig.savefig(count_path, bbox_inches='tight')

        def assign_age_group(age):
            if age<=12:return '0-12'
            elif age<=19:return '13-19'
            elif age<=35:return '20-44'
            elif age<=65:return '45-65'
            elif age<=80:return '66-80'
            else: return '<80'
        df['age_group'] = df['age'].apply(assign_age_group)

        male_fig=plt.figure()
        male_ax=male_fig.add_subplot(111)
        sns.countplot(data=df[df['gender']=='Male'], x="age_group", hue="dataset",
                    ax=male_ax,order=['0-12','13-19','20-44','45-65','66-80','<80'])
        male_path=os.path.join("plots", "male_plot.png")
        male_fig.savefig(male_path)

        female_fig=plt.figure()
        female_ax=female_fig.add_subplot(111)
        sns.countplot(data=df[df['gender']=='Female'], x="age_group", hue="dataset",
                    ax=female_ax,order=['0-12','13-19','20-44','45-65','66-80','<80'])
        female_path=os.path.join("plots", "female_plot.png")
        female_fig.savefig(female_path)
        df.drop('age_group',axis=1,inplace=True)

        fig1, ax1=plt.subplots(2,2)
        fig1.set_size_inches(12, 15)
        sns.boxplot(data=df,x='total_bilirubin',y='dataset',ax=ax1[0,0],orient='h')
        sns.boxplot(data=df,x='direct_bilirubin',y='dataset',ax=ax1[0,1],orient='h')
        sns.boxplot(data=df,x='alamine_aminotransferase',y='dataset',ax=ax1[1,0],orient='h')
        sns.boxplot(data=df,x='aspartate_aminotransferase',y='dataset',ax=ax1[1,1],orient='h')
        ax1_path=os.path.join("plots", "ax1_plot.png")
        fig1.savefig(ax1_path,bbox_inches='tight')

        fig2, ax2=plt.subplots(2,2)
        fig2.set_size_inches(12, 15)
        sns.boxplot(data=df,x='alkaline_phosphotase',y='dataset',ax=ax2[0,0],orient='h')
        sns.boxplot(data=df,x='total_protiens',y='dataset',ax=ax2[0,1],orient='h')
        sns.boxplot(data=df,x='albumin',y='dataset',ax=ax2[1,0],orient='h')
        sns.boxplot(data=df,x='albumin_and_globulin_ratio',y='dataset',ax=ax2[1,1],orient='h')
        ax2_path=os.path.join("plots", "ax2_plot.png")
        fig2.savefig(ax2_path,bbox_inches='tight')

        
        return {"plot_1": count_path, "plot_2":corr_path, "plot_3": male_path, "plot_4":female_path, "plot_5":ax1_path,"plot_6":ax2_path}


    def create_validation_graphs(self, validation_scores):
        accuracy_scores=validation_scores['accuracy_score']
        confusion_matrices=validation_scores['confusion_matrix']
        roc_curves=validation_scores['roc_curve']
        roc_auc_scores=validation_scores['roc_auc']

        if not os.path.exists('./validation_plots'):
            os.makedirs('./validation_plots') 
        
        cm_fig,ax=plt.subplots(2,2)
        ax=ax.flatten()
        i=0
        for model,cm in confusion_matrices.items():
            try:sns.heatmap(cm, annot=True, cmap="Blues",ax=ax[i],fmt='d')
            except:break
            ax[i].set_xlabel("Predicted label")
            ax[i].set_ylabel("True label")
            ax[i].set_title(model)
            i+=1
        cm_fig.tight_layout(pad=3.0)
        cm_path=os.path.join("validation_plots", "cm_plot.png")
        cm_fig.savefig(cm_path,bbox_inches='tight')

        bar_fig=plt.figure()
        bar_ax=bar_fig.add_subplot(111)
        sns.barplot(y=list(accuracy_scores.values()),x=list(accuracy_scores.keys()),ax=bar_ax)
        bar_ax.set_xlabel("Models")
        bar_ax.set_ylabel("Accuracy")
        for i,v in enumerate(accuracy_scores.values()):
            bar_ax.text(i,v+0.02,'{:.2f}'.format(v),ha='center')
        bar_path=os.path.join("validation_plots", "bar_plot.png")
        bar_fig.savefig(bar_path)#bbox='tight'

        roc_fig=plt.figure()
        roc_ax=roc_fig.add_subplot(111)
        for name,roc in roc_curves.items():
            fpr,tpr,_=roc
            sns.lineplot(x=fpr,y=tpr,label='{} (AUC = {:.2f})'.format(name, roc_auc_scores[name]),ax=roc_ax)
        roc_path=os.path.join("validation_plots", "roc_plot.png")
        roc_ax.set_xlabel("False positive rate")
        roc_ax.set_ylabel("True positive rate")
        roc_fig.savefig(roc_path)#bbox='tight'

        return {"plot_1": cm_path, "plot_2": bar_path, "plot_3": roc_path}

    def visualize(self, window, graph_type_var, graphs):
        def resize_image(*args):
            nonlocal graph_label
            frame_width = frame.winfo_width()
            frame_height = frame.winfo_height()
            graph_type = graph_type_var.get()
            img_path = graphs[graph_type]
            img = Image.open(img_path)
            new_height = frame_height
            new_width = frame_width
            img = img.resize((new_width, new_height))
            img_tk = ImageTk.PhotoImage(img)
            graph_label.config(image=img_tk)
            graph_label.image = img_tk

        frame = tk.Frame(window)
        frame.pack(side=tk.TOP, padx=10, pady=10, expand=True, fill='both')
        frame.bind('<Configure>', resize_image)

        img_path = graphs[graph_type_var.get()]
        img = Image.open(img_path)
        img_tk = ImageTk.PhotoImage(img)
        graph_label = tk.Label(frame, image=img_tk)
        graph_label.pack(padx=10, pady=10)

        graph_type_var.trace('w', resize_image)
