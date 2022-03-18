import abc, os, prefect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split


class PreprocessingBaseClass_TimeSeries(abc.ABC):
    @abc.abstractmethod
    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def load_data(self, file_path, file_name, file_sheet):
        csv_path = os.path.join(file_path, file_name)
        return pd.read_excel(csv_path, sheet_name=file_sheet)

    def cleaning_data(self, data, file_columns, file_columns_method, label):
        #selection useful columns
        if file_columns_method == "include":
            data = data.filter(items=file_columns, axis=1)
        elif file_columns_method == "exclude":
            data = data.drop(labels=file_columns, axis=1)
        else:
            raise ValueError("Wrong input! See config file_columns_method!")

        #delete incorrect text columns
        data = self.del_text_columns(data)

        #replace incorrect num columns
        data = self.del_num_columns(data)

        #counts number of labels
        unique, frequency = np.unique(data[label], return_counts = True)
        print("Number classes:", len(unique), "\nName classes:", *unique)
        print("Number samples of each class:", *frequency)
        return data 

    def del_text_columns(self, data): 
        #delete incorrect text columns (empty get deleted)
        #search all text columns
        old_data = len(data)
        text_vars = data.select_dtypes(include=["object"])
        
        #delete incorrect columns
        data = data.dropna(subset=text_vars.columns,how="any")
        
        #output
        count_deleted_rows = old_data - len(data)
        print("Due to missing textual information", count_deleted_rows, "rows deleted.")
        return data

    def del_num_columns(self, data): 
        #replace incorrect num columns (attention, value = 0 is NO incorrect value, but NaN)
        #calculate median/mean
        fill_mean = lambda col: col.fillna(col.mean())
        fill_median = lambda col: col.fillna(col.median())
        
        #search all numeric columns
        num_vars = data.select_dtypes(include=["number"])
        count_changed_values = num_vars.isna().sum().sum()
        
        #set median/mean value in numeric columns with fillna (see above)
        data[num_vars.columns] = data[num_vars.columns].apply(fill_median)
        
        #output
        print("Due to missing numerical data(NaN)", count_changed_values, "results replaced by median value.")
        return data

    def sklearn_normalization(self, data): 
        #normalization (d.h. values in range 0-1)
        #search all numeric columns
        num_vars = data.select_dtypes(include=["number"])
        
        #normalizer
        data[num_vars.columns] = MinMaxScaler().fit_transform(num_vars.values)
        return data

    def sklearn_standardization(self, data): 
        #here no range 0-1(!), but mean=0 and std=1, can cause problems by NN, but better for statistic outliner
        #search all numeric columns
        num_vars = data.select_dtypes(include=["number"])
        
        #standardizer
        data[num_vars.columns] = StandardScaler().fit_transform(num_vars.values)
        return data
       
    def sklearn_onehot(self, data, inverse=False):
        #attention, there 2 categoric types
        #ordinal cat. var. e.g. t-shirt size S<M<L, or nominal cat. var. e.g. colors of t-shirt(here no "ranking")
        #ordinal -> integer encoding, nominal -> onehot encoding
        #(check https://towardsdatascience.com/introduction-to-data-preprocessing-in-machine-learning-a9fa83a5dc9d) 
        #search all text columns
        text_vars = data.select_dtypes(include=["object"])

        #integer encoding (every object/string will assigned to an integer (ordinal cat.!))
        LabelEnco = LabelEncoder()
        text_vars_labelenco = LabelEnco.fit_transform(text_vars.values.ravel())

        #onehot encoding (additionally to integer encoding, integer will encoded to vector [0 0 1]... (nominal cat.!))
        OneHotEnco = OneHotEncoder(sparse=False)
        text_vars_onehot = text_vars_labelenco.reshape(len(text_vars_labelenco), 1)
        text_vars_onehot = OneHotEnco.fit_transform(text_vars_onehot)

        if inverse == True:
            #onehot -> integer
            inverted = OneHotEnco.inverse_transform(text_vars_onehot)
            #integer -> string
            inverted = LabelEnco.inverse_transform(inverted.ravel())
            return inverted
        if inverse == False:
            return text_vars_onehot

    def sklearn_labelencoder(self, data):
        #converts strings to integer(encodes label in alphabetic order(!))
        #search all text columns
        text_vars = data.select_dtypes(include=["object"])

        #labelencoder
        data[text_vars.columns] = LabelEncoder().fit_transform(text_vars.values.ravel()).reshape(len(text_vars), 1)
       
        return data

    def sklearn_split(self, data, train_split, val_split, test_split, shuffle):
        #split in features and labels/targets
        x = data.drop(labels=["Kategorie"], axis=1)
        y = data.filter(items=["Kategorie"], axis=1)

        #check if splits are correclty specified in config.yaml
        if train_split + val_split + test_split != 1:
            raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                          (train_split, val_split, test_split))

        #split in train/test/val
        relative_split = test_split / (val_split+test_split)
        if shuffle is False:
            x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=(1-train_split), shuffle=False)
            x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=relative_split, shuffle=False)
        else:
            x_train, x_temp, y_train, y_temp = train_test_split(x, y, stratify=y, test_size=(1-train_split))
            x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, stratify=y_temp, test_size=relative_split)

        #output
        print("Using %d samples for training, %d for validation , %d for testing."% (len(x_train), len(x_val), len(x_test)))
        print("Distribution of labels for Train-/Test-/Valset:")
        print(y_train.value_counts())
        print(y_val.value_counts())
        print(y_test.value_counts())
        
        return x_train, y_train, x_val, y_val, x_test, y_test
 

class PreprocessingBaseClass_StructuredData(abc.ABC):
   
    @abc.abstractmethod
    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def load_data(self, file_path, file_name, file_sheet):
        csv_path = os.path.join(file_path, file_name)
        return pd.read_excel(csv_path, sheet_name=file_sheet)

    def cleaning_data(self, data, file_columns, file_columns_method, label):
        #selection useful columns
        if file_columns_method == "include":
            data = data.filter(items=file_columns, axis=1)
        elif file_columns_method == "exclude":
            data = data.drop(labels=file_columns, axis=1)
        else:
            raise ValueError("Wrong input! See config file_columns_method!")

        #delete incorrect text columns
        data = self.del_text_columns(data)

        #replace incorrect num columns
        data = self.del_num_columns(data)

        #counts number of labels
        unique, frequency = np.unique(data[label], return_counts = True)
        print("Number classes:", len(unique), "\nName classes:", *unique)
        print("Number samples of each class:", *frequency)
        return data 

    def del_text_columns(self, data): 
        #delete incorrect text columns (empty or NaN get deleted)
        #search all text columns
        old_data = len(data)
        text_vars = data.select_dtypes(include=["object"])
        
        #delete incorrect columns
        data = data.dropna(subset=text_vars.columns,how="any")
        
        #output
        count_deleted_rows = old_data - len(data)
        print("Due to missing textual information", count_deleted_rows, "rows deleted.")
        return data

    def del_num_columns(self, data): 
        #replace incorrect num columns (attention, value = 0 is NO incorrect value, but NaN)
        #calculate median/mean
        fill_mean = lambda col: col.fillna(col.mean())
        fill_median = lambda col: col.fillna(col.median())
        
        #search all numeric columns
        num_vars = data.select_dtypes(include=["number"])
        count_changed_values = num_vars.isna().sum().sum()
        
        #set median/mean value in numeric columns with fillna (see above)
        data[num_vars.columns] = data[num_vars.columns].apply(fill_median)
        
        #output
        print("Due to missing numerical data(NaN)", count_changed_values, "results replaced by median value.")
        return data

    def sklearn_normalization(self, data): 
        #normalization (d.h. values in range 0-1)
        #search all numeric columns
        num_vars = data.select_dtypes(include=["number"])
        
        #normalizer
        data[num_vars.columns] = MinMaxScaler().fit_transform(num_vars.values)
        return data

    def sklearn_standardization(self, data): 
        #here no range 0-1(!), but mean=0 and std=1, can cause problems by NN, but better for statistic outliner
        #search all numeric columns
        num_vars = data.select_dtypes(include=["number"])
        
        #standardizer
        data[num_vars.columns] = StandardScaler().fit_transform(num_vars.values)
        return data
       
    def sklearn_onehot(self, data, inverse=False):
        #attention, there 2 categoric types
        #ordinal cat. var. e.g. t-shirt size S<M<L, or nominal cat. var. e.g. colors of t-shirt(here no "ranking")
        #ordinal -> integer encoding, nominal -> onehot encoding
        #(check https://towardsdatascience.com/introduction-to-data-preprocessing-in-machine-learning-a9fa83a5dc9d) 
        #search all text columns
        text_vars = data.select_dtypes(include=["object"])

        #integer encoding (every object/string will assigned to an integer (ordinal cat.!))
        LabelEnco = LabelEncoder()
        text_vars_labelenco = LabelEnco.fit_transform(text_vars.values.ravel())

        #onehot encoding (additionally to integer encoding, integer will encoded to vector [0 0 1]... (nominal cat.!))
        OneHotEnco = OneHotEncoder(sparse=False)
        text_vars_onehot = text_vars_labelenco.reshape(len(text_vars_labelenco), 1)
        text_vars_onehot = OneHotEnco.fit_transform(text_vars_onehot)

        if inverse == True:
            #onehot -> integer
            inverted = OneHotEnco.inverse_transform(text_vars_onehot)
            #integer -> string
            inverted = LabelEnco.inverse_transform(inverted.ravel())
            return inverted
        if inverse == False:
            return text_vars_onehot

    def tf_layer_normalization(self, name, dataset):
        #here no range 0-1(!), but mean=0 and std=1, can cause problems by NN, but better for statistic outliner
        
        #create normalization layer feature
        normalizer = preprocessing.Normalization(axis=None) #keras normalization = standardization
        
        #prepare dataset that only yields feature
        feature_ds = dataset.map(lambda x, y: x[name])

        #learn statistics
        normalizer.adapt(feature_ds)
        return normalizer

    def tf_set_normalization(self, columns, dataset, all_inputs, encoded_features):
        #set normalization to defined columns and initialize all_inputs and encoded_features
        for header in columns:
            num_col = tf.keras.Input(shape=(1,), name=header)
            layer = self.tf_layer_normalization(header, dataset)
            encoded_num_col = layer(num_col)
            all_inputs.append(num_col)
            encoded_features.append(encoded_num_col)

    def tf_layer_category_to_onehot(self, name, dataset, dtype, max_tokens=None): 
        #attention, there 2 categoric types
        #ordinal cat. var. e.g. t-shirt size S<M<L, or nominal cat. var. e.g. colors of t-shirt(here no "ranking")
        #ordinal -> integer encoding, nominal -> onehot encoding
        #(check https://towardsdatascience.com/introduction-to-data-preprocessing-in-machine-learning-a9fa83a5dc9d) 
        
        #categorical features (nominal -> string/integer to onehot)
        #create StringLookup layer which will turn strings into integer indices
        if dtype == 'string':
            index = preprocessing.StringLookup(max_tokens=max_tokens)
        else:
            index = preprocessing.IntegerLookup(max_tokens=max_tokens)
        
        #prepare dataset that only yields feature
        feature_ds = dataset.map(lambda x, y: x[name])
        
        #learn statistics
        index.adapt(feature_ds)
        
        #create a Discretization for our integer indices
        encoder = preprocessing.CategoryEncoding(num_tokens=index.vocabulary_size())
        #apply one-hot encoding to our indices, lambda function captures the
        #layer so we can use them, or include them in the functional model later
        return lambda feature: encoder(index(feature))

    def tf_set_category_to_onehot(self, columns, dataset, all_inputs, encoded_features, dtype, max_tokens=None):
        #set cat_to_onehot to defined columns and initialize all_inputs and encoded_features
        if dtype == 'string':
            #categorical features (string to onehot)
            for header in columns:
                str_col = tf.keras.Input(shape=(1,), name=header, dtype=dtype)
                layer = self.tf_layer_category_to_onehot(header, dataset, dtype, max_tokens)
                encoded_str_col = layer(str_col)
                all_inputs.append(str_col)
                encoded_features.append(encoded_str_col)
        else:
            #categorical features (integer to onehot)
            for header in columns:
                int_col = tf.keras.Input(shape=(1,), name=header, dtype=dtype)
                layer = self.tf_layer_category_to_onehot(header, dataset, dtype, max_tokens)
                encoded_int_col = layer(int_col)
                all_inputs.append(int_col)
                encoded_features.append(encoded_int_col)

    def tf_layer_category_to_integer(self, name, dataset, max_tokens=None):
        #attention, there 2 categoric types
        #ordinal cat. var. e.g. t-shirt size S<M<L, or nominal cat. var. e.g. colors of t-shirt(here no "ranking")
        #ordinal -> integer encoding, nominal -> onehot encoding
        #(check https://towardsdatascience.com/introduction-to-data-preprocessing-in-machine-learning-a9fa83a5dc9d) 
        
        #categorical features (ordinal -> string to integer)
        #create StringLookup layer which will turn strings into integer indices
        encoder = preprocessing.StringLookup(max_tokens=max_tokens)
        
        #prepare dataset that only yields feature
        feature_ds = dataset.map(lambda x, y: x[name])
        
        #learn statistics
        encoder.adapt(feature_ds)
        return encoder

    def tf_set_category_to_integer(self, columns, dataset, all_inputs, encoded_features, max_tokens=None):
         #set cat_to_integer to defined columns and initialize all_inputs and encoded_features
        for header in columns:
            str_col = tf.keras.Input(shape=(1,), name=header, dtype="string")
            layer = self.tf_layer_category_to_integer(header, dataset, max_tokens)
            encoded_str_col = layer(str_col)
            encoded_str_col = tf.cast(encoded_str_col, tf.float32) #change type int64 to float32
            all_inputs.append(str_col)
            encoded_features.append(encoded_str_col)

    def tf_dataframe_to_dataset(self, dataframe, label_column, train_split, val_split, test_split, batch_size=32, shuffle=True): 
        #copy dataset
        dataframe = dataframe.copy()
        #set label
        labels = dataframe.pop(label_column) #indicate label column(!)
        dataframe.columns = dataframe.columns.astype(str) #change columns to string (cause integers like 200 dont fit in keras.input)
        df = {key: value[:,tf.newaxis] for key, value in dataframe.items()}
        dataset = tf.data.Dataset.from_tensor_slices((dict(df), labels))

        #shuffle
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        #initialize splits
        train_size = int(train_split * len(dataset))
        val_size = int(val_split * len(dataset))
        test_size = int(test_split * len(dataset))

        #split dataset
        train_dataset = dataset.take(train_size)
        test_dataset = dataset.skip(train_size)
        val_dataset = test_dataset.skip(val_size)
        test_dataset = test_dataset.take(test_size)
        print("Using %d samples for training, %d for validation, %d for testing."% (len(train_dataset), len(val_dataset), len(test_dataset)))

        #batch and prefetch
        train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        return train_dataset, val_dataset, test_dataset


class PreprocessingBaseClass_Images(abc.ABC):
    @abc.abstractmethod
    def __init__(self, config):
        self.config = config

    def load_images(self, file_path, image_size, batch_size, split, seed=None):
        
        #e.g split is [.7,.2,.1] #train/val/test
        validation_split = split[1]+split[2] #e.g. 0.3 cause RELATIVE(!)
        relative_split = split[2] / (split[1]+split[2]) #e.g 0.33 cause RELATIVE(!)
        #-> size=100 -> train/val -> 70/30 (*0.3) -> val/test -> 20/10 (*0.33) -> now you got split [.7,.2,.1]
        
        #training
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        file_path,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=(image_size[0],image_size[1]),
        batch_size=batch_size,)

        #validation
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        file_path,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=(image_size[0],image_size[1]),
        batch_size=batch_size,)

        #testing
        #(not possible with "tf.keras.preprocessing.image_dataset_from_directory" in current version, 
        # need to take from val_ds or use different folders /train/class1, /test/class1, /val/class1)
        test_ds = val_ds.take(int(len(val_ds)*relative_split))
        val_ds = val_ds.skip(int(len(val_ds)*relative_split))
        
        print("Splitting validation into val/test set.")
        print("Batches Trainset:", len(train_ds))
        print("Batches Validationset:", len(val_ds))
        print("Batches Testset:", len(test_ds))
        print("Classes:", train_ds.class_names)

        return train_ds, val_ds, test_ds

    def rescaling_images(self, rescaling):
        model = tf.keras.Sequential(name="rescaling")
        if list(rescaling[0]) == [0,1]:
            #rescale RGB in range [0,1]
            model.add(tf.keras.layers.Rescaling(1./255))
        if list(rescaling[0]) == [-1,1]:
            #rescale RGB in range [-1,1]
            model.add(tf.keras.layers.Rescaling(1./127.5, offset= -1))
        return model

    def augmenting_images(self, random_crop, random_flip, random_translation, random_rotation, random_zoom):
        #data augmentation of images (RandomCrop, RandomFlip, RandomTransformation, RandomRotation, RandomZoom)
        model = tf.keras.Sequential(name="augmentation")

        #cropes image randomly to defined size
        if random_crop[0] is True: 
            model.add(tf.keras.layers.RandomCrop(random_crop[1][0],random_crop[1][1]))
        #flip image horizontal/vertical
        if random_flip[0] is True: 
            model.add(tf.keras.layers.RandomFlip(mode=random_flip[1]))
        #translate image in random directions
        if random_translation[0] is True: 
            model.add(tf.keras.layers.RandomTranslation(random_translation[1][0],random_translation[1][1]))
        #rotate image
        if random_rotation[0] is True:
            model.add(tf.keras.layers.RandomRotation(random_rotation[1]))
        #zoom image 
        if random_zoom[0] is True:
            model.add(tf.keras.layers.RandomZoom(random_zoom[1][0],random_zoom[1][1]))
        
        return model

    #for testing 
    def test_output(self, image):
        plt.imshow(image.numpy().astype("uint8"))
        plt.show()
