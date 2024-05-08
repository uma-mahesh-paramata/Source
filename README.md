# Liver Disease Prediction Project
This project implements a machine learning pipeline for predicting liver disease based on patient data. It leverages ensemble learning algorithms to analyze various biochemical features and assess the likelihood of liver disease in a patient. The project is designed with modularity and customization in mind, making it adaptable to other classification problems beyond liver disease prediction.

This codebase provides a foundation for building and evaluating machine learning models for classification tasks. By swapping the data and potentially modifying the model selection/hyperparameter tuning, you can adapt this project to analyze different datasets and predict various outcomes.

First, download the repository to your local machine and then follow the instructions below to run this project.

## Setting Up the Environment

Here's how to set up the development environment for this project:

### Install virtualenvwrapper (Optional):

While not strictly necessary, using virtualenvwrapper helps manage multiple virtual environments for different projects. You can install it using pip:

```bash
pip install virtualenvwrapper
```
Follow the instructions to set up virtualenvwrapper after installation.

### Create a Virtual Environment:

Open a terminal and navigate to your project directory. Create a virtual environment named ml-health-classification using the following command:

```bash
mkvirtualenv ml-health-classification
```
### Activate the virtual environment:

```bash
workon ml-health-classification
```
### Install Dependencies:

Install the required libraries using pip within the activated virtual environment:

```bash
pip install -r requirements.txt
```
## Project Structure
- **main_GUI.py**: This file is the main script containing the GUI logic for user interaction (loading data, training models, making predictions, and visualizing results).
- **models.py**: Contains code for loading, training, evaluating, and saving machine learning models.
- **load_preprocess.py**: This file contains functions for loading and preprocessing the dataset used for liver disease prediction.
- **visualisations.py**: Contains code for creating data analysis and validation visualizations.
- **requirements.txt**: Text file listing the project's dependencies.

## Running the Project
main_GUI.py is the entry point, you can typically run the project from the command line within the activated virtual environment:

```bash
python main_GUI.py
```
This will launch the graphical user interface, allowing you to interact with the application for tasks like:
- Loading patient data for prediction.
- Training or loading machine learning models.
- Making predictions on new data points.
- Visualizing analysis and validation results.
