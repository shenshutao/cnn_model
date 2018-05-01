import os

from keras import backend as K
from keras.applications.resnet50 import ResNet50, Flatten
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from custom_predict import do_predict
from custom_split_data import split_data
from custom_getFile import download_file_from_google_drive
import zipfile

import pandas as pd


def preprocess_input(x):
    x /= 255.    # normalization
    x -= 0.5
    x *= 2.
    return x


def train_resNet(train_data_dir, validate_data_dir, categories, res_dir, model_file_name, weight_file_name, img_width=224,
                 img_height=224):
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    batch_size = 40

    categorie_size = len(categories)

    # # 2.augmentation (may try more)
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        # samplewise_std_normalization=True,  # divide each input by its std
        # zca_whitening=True,  # apply ZCA whitening
        channel_shift_range=100,
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,
        shear_range=0.05,
        zoom_range=0.05,
        fill_mode='nearest')

    validate_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        classes=categories,
        class_mode='categorical')

    validate_generator = validate_datagen.flow_from_directory(
        validate_data_dir,
        target_size=(img_width, img_height),
        classes=categories,
        batch_size=batch_size,
        class_mode='categorical')

    # # 3. model structure
    # # Base model Conv layers + Customize FC layers
    # # create the base pre-trained model with weights
    if K.image_data_format() == 'channels_first':
        the_input_shape = (3, img_width, img_height)
    else:
        the_input_shape = (img_width, img_height, 3)
    base_model = ResNet50(weights='imagenet', include_top=False,
                          input_shape=the_input_shape) # don't include the top (final FC) layers.

    x = base_model.output
    x = Flatten(input_shape=base_model.output_shape[1:])(x)
    predictions = Dense(categorie_size, activation='softmax', name='fc05')(x)

    # first: train only the FC layers (which were randomly initialized)
    # i.e. freeze all convolutional resnet layers
    for layer in base_model.layers:
        layer.trainable = False

    # this is the final model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    #     model = multi_gpu_model(model, gpus=1)
    #     model.summary()

    # # 4.compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # # 5.train the model on the new data for a few epochs
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=2,
        validation_data=validate_generator,
        validation_steps=validate_generator.n // batch_size)

    # model.load_weights('weights_resnet_224_before_finetune.h5')
    # # 6.start fine tune.
    NO_OF_LAYERS_TO_FREEZE = 0  # currently 0 freeze layers with low learning rate works best.
    for layer in model.layers[:NO_OF_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NO_OF_LAYERS_TO_FREEZE:]:
        layer.trainable = True

    # fine tune: stochastic gradient descent optimizer
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # fine tune: train again for fine tune
    # save best weights
    check_pointer1 = ModelCheckpoint(monitor='val_acc', filepath=os.path.join(res_dir, weight_file_name),
                                     verbose=1, save_best_only=True, mode='auto', period=1)
    # save the last model
    check_pointer2 = ModelCheckpoint(monitor='val_acc', filepath=os.path.join(res_dir, model_file_name),
                                     verbose=1, save_best_only=False, save_weights_only=False, mode='auto',
                                     period=1)
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=10,
        validation_data=validate_generator,
        validation_steps=validate_generator.n // batch_size,
        callbacks=[check_pointer1, check_pointer2])

    model.save(os.path.join(res_dir, model_file_name))


if __name__ == "__main__":
    # os.chdir('/Users/shutao/Desktop')

    # For big image.
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    try:
        print('======== Download data sets =========')
        download_file_from_google_drive('1l_EmtfJ3QdH2S0QFhFsE22sMKn_vMAAn', 'train_data.zip')
        zip_ref = zipfile.ZipFile('train_data.zip', 'r')
        zip_ref.extractall('train_data')
        zip_ref.close()

        download_file_from_google_drive('1wKH68-hx8doGewV0n56l6G4j88tgowL7', 'pictures.zip')
        zip_ref = zipfile.ZipFile('pictures.zip', 'r')
        zip_ref.extractall('pictures')
        zip_ref.close()

        print('======== Split data into train / validate sets =========') # Just because Keras 2.14 don't support this function.
        split_data('train_data', 'validate_data', 0.3)

        print('======== Train resNet Model !!! =========')
        categories = ['Beach', 'City', 'Forest', 'Mountain', 'Village']
        train_resNet('train_data', 'validate_data', categories, 'resNet', 'resnet_last_model.h5',
                     'weights_resnet_best.h5', 224, 224)
        do_predict('resNet/resnet_last_model.h5', 'resNet/weights_resnet_best.h5', categories, 224,
                   224, 'pictures', 'resNet/output_result.csv')
        # do_predict('resNet/resnet_last_model.h5', None, 224, 224, 'predict',
        #            'resNet/output_result_final.csv')

        print('======== Reformat the output for assignment. =========')
        df = pd.read_csv('resNet/output_result.csv')
        df['destination'] = df['id'].apply(lambda jpgname: jpgname.split('_')[1])

        agg_data = df.groupby(['destination', 'category']).agg({'category': ['count']})
        percent_data = agg_data / agg_data.groupby(level=0).sum()
        rows = []
        column_names = ['Destination', 'Mountain', 'Beach', 'Forest', 'City', 'Village']
        for city in percent_data.index.levels[0]:
            row = [city, 0, 0, 0, 0, 0]
            rows.append(row)

        df = pd.DataFrame(rows, columns=column_names)
        df = df.set_index('Destination')
        for index, row in percent_data.iterrows():
            df.loc[index[0], index[1]] = str(row[0])

        print('======== Done, store the result ! =========')
        df.to_csv('geofile_bigdata.csv', index=True, header=True)

    except Exception as e:
        print(e.message)
