from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTENC
import category_encoders as ce
from pandas import read_csv, concat
import pandas as pd
import tkinter.filedialog as fd

class DataHandler:
    def __init__(self, dataset=None):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.single_test = None
        self.df = None
        self.encoder = None
        self.imputer = KNNImputer()
        if dataset:
            self.load_dataset(dataset)

    # def load_dataset(self, dataset):
    #     try:
    #         self.df = read_csv(dataset)
    #         self.df.columns = self.df.columns.map(str.lower)
    #         self._detect_categorical_features()
    #         self._initialize_encoders()
    #         self.imputer.fit(self.df.drop('dataset', axis=1))  # Fit imputer excluding target column
    #     except Exception as e:
    #         print(f"Error loading dataset: {e}")
        

    def _detect_categorical_features(self):
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        self.categorical_indices = [self.df.columns.get_loc(col) for col in self.categorical_cols]
    
    def _initialize_encoders(self):
        self.encoders = {}
        for col in self.categorical_cols:
            num_unique = self.df[col].nunique()
            if num_unique <= 5:  # For low cardinality, use OneHotEncoder
                encoder = ce.OneHotEncoder(cols=[col], handle_unknown='return_nan', return_df=True, use_cat_names=True)
            elif num_unique <= 20:  # For moderate cardinality, use OrdinalEncoder
                encoder = ce.OrdinalEncoder(cols=[col], handle_unknown='return_nan', return_df=True)
            else:  # For high cardinality, use TargetEncoder
                encoder = ce.TargetEncoder(cols=[col], handle_unknown='return_nan', return_df=True)
            self.encoders[col] = encoder

    def preprocess(self, data_type='train'):
        if data_type == 'train':
            self._preprocess_train()
        elif data_type == 'test':
            self._preprocess_test()
        elif data_type == 'single':
            self._preprocess_single()

    def _preprocess_train(self):
        self.X_train = self._fit_features(self.X_train)
        self.X_train = pd.DataFrame(self.imputer.fit_transform(self.X_train), columns=self.X_train.columns)
        self.y_train = self.y_train.replace({1: 1, 2: 0})
        smote = SMOTENC(sampling_strategy=0.6, random_state=24, k_neighbors=5, categorical_features=self.categorical_indices)
        X_new, y_new = smote.fit_resample(self.X_train, self.y_train)
        self.X_train = pd.concat([self.X_train, pd.DataFrame(X_new, columns=self.X_train.columns)], axis=0)
        self.y_train = pd.concat([self.y_train, pd.Series(y_new)], axis=0)
        #print(self.X_train,self.y_train)

    def _preprocess_test(self):
        self.X_test = self._transform_features(self.X_test)
        self.X_test = pd.DataFrame(self.imputer.transform(self.X_test), columns=self.X_test.columns)
        self.y_test = self.y_test.replace({1: 1, 2: 0})
        #print(self.X_test,self.y_test)

    def _preprocess_single(self):
        df = self.df.drop('dataset', axis=1)
        self.single_test = pd.DataFrame(self.single_test, columns=df.columns)
        self.single_test = self._transform_features(self.single_test)

    def _transform_features(self, data):
        try:
            for col, encoder in self.encoders.items():
                data = encoder.transform(data)
            return data
        except:
            for col, encoder in self.encoders.items():
                data = encoder.fit_transform(data)
            return data
    
    def _fit_features(self, data):
            for col, encoder in self.encoders.items():
                data = encoder.fit_transform(data)
            return data
    
    def load_data(self):
        self.X_test = None
        self.y_test = None
        try:
            data = fd.askopenfilename()
        except Exception as e:
            print(f"Error loading data: {e}")
            return

        self.state = f"{data} loaded\n\n"
        try:
            self.df = read_csv(data)
            self.df.columns = self.df.columns.map(str.lower)
            self._detect_categorical_features()
            self._initialize_encoders()
        except Exception as e:
            print(f"Error opening dataset file: {e}")
        self.state += f"{self.df}\n\nNo.of duplicate rows : {self.df.duplicated().sum()}\n\n"
        self.df.drop_duplicates(inplace=True)
        return self.state
    
    def load_testdata(self):
        import tkinter.filedialog as fd
        try:test_data = fd.askopenfilename(initialdir="dataset",filetypes=[("csv","*.csv")],title='select test data to validate models')
        except:return
        test_df=read_csv(test_data)
        test_df.columns = test_df.columns.map(str.lower)
        test_df.drop_duplicates(inplace=True)
        ## dividing features
        self.X_test= test_df.drop('dataset', axis=1) # X -> features
        self.y_test = test_df['dataset'] # y -> target feature
        ##

    def split_data(self, test_size=None):
        X = self.df.drop('dataset', axis=1)
        y = self.df['dataset']
        if test_size is not None:
            try:
                test_size = float(test_size)
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=test_size, random_state=0)
                test_data = concat([self.X_test, self.y_test], axis=1)
                test_data.to_csv('TEST.csv', index=False)
            except Exception as e:
                print(f"Splitting operation failed: {e}")
                return
        else:
            self.X_train=X
            self.y_train=y