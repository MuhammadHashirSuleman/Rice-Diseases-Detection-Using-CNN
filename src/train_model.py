# # import tensorflow as tf
# # from tensorflow.keras.applications import ResNet50
# # from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
# # from tensorflow.keras.models import Model
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator


# # def create_model():
# #     weight_path = 'E:/weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
# #     base_model = ResNet50(weights=weight_path, include_top=False, input_shape=(224, 224, 3))
# #     x = base_model.output
# #     x = GlobalAveragePooling2D()(x)
# #     x = Dense(3, activation='softmax')(x)  # 3 classes
# #     model = Model(inputs=base_model.input, outputs=x)

# #     for layer in base_model.layers:
# #         layer.trainable = False

# #     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# #     return model

# # def train_model():
# #     train_dir = "data/processed/train"
# #     val_dir = "data/processed/validation"
# #     train_datagen = ImageDataGenerator(rescale=1./255)
# #     val_datagen = ImageDataGenerator(rescale=1./255)
# #     train_generator = train_datagen.flow_from_directory(
# #         train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
# #     val_generator = val_datagen.flow_from_directory(
# #         val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
# #     model = create_model()
# #     checkpoint = tf.keras.callbacks.ModelCheckpoint('models/trained_model.h5', save_best_only=True, monitor='val_accuracy')
# #     model.fit(train_generator, epochs=20, validation_data=val_generator, callbacks=[checkpoint])

# # if __name__ == "__main__":
# #     train_model()



# # Again

# import tensorflow as tf
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.utils.class_weight import compute_class_weight
# import numpy as np

# def create_model():
#     weight_path = 'E:/weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
#     base_model = ResNet50(weights=weight_path, include_top=False, input_shape=(224, 224, 3))
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(3, activation='softmax')(x)  # 3 classes: Bacterial Leaf Blight, Brown Spot, Leaf Smut
#     model = Model(inputs=base_model.input, outputs=x)

#     # Unfreeze last ~30 layers for fine-tuning (adjust based on ResNet50 architecture)
#     for layer in base_model.layers:
#         layer.trainable = False
#     for layer in base_model.layers[-30:]:
#         layer.trainable = True

#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), 
#                   loss='categorical_crossentropy', 
#                   metrics=['accuracy'])
#     return model

# def train_model():
#     train_dir = "data/processed/train"
#     val_dir = "data/processed/validation"
    
#     # Enhanced data augmentation for training
#     train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         rotation_range=30,
#         width_shift_range=0.3,
#         height_shift_range=0.3,
#         horizontal_flip=True,
#         zoom_range=0.3,
#         shear_range=0.2
#     )
#     val_datagen = ImageDataGenerator(rescale=1./255)
    
#     train_generator = train_datagen.flow_from_directory(
#         train_dir, target_size=(224, 224), batch_size=16, class_mode='categorical')
#     val_generator = val_datagen.flow_from_directory(
#         val_dir, target_size=(224, 224), batch_size=16, class_mode='categorical')
    
#     # Compute class weights to handle imbalance
#     class_weights = compute_class_weight(
#         class_weight='balanced',
#         classes=np.unique(train_generator.classes),
#         y=train_generator.classes
#     )
#     class_weight_dict = dict(enumerate(class_weights))
    
#     model = create_model()
#     model.load_weights('models/trained_model.h5')  # Load existing trained weights
    
#     # Callbacks for checkpointing and learning rate scheduling
#     checkpoint = tf.keras.callbacks.ModelCheckpoint(
#         'models/finetuned_model.h5', save_best_only=True, monitor='val_accuracy')
#     lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
#         monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6)
    
#     # Fine-tune for 3 epochs
#     model.fit(
#         train_generator,
#         epochs=3,
#         validation_data=val_generator,
#         class_weight=class_weight_dict,
#         callbacks=[checkpoint, lr_scheduler]
#     )

# if __name__ == "__main__":
#     train_model()


import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def create_model():
    weight_path = 'E:/weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    base_model = ResNet50(weights=weight_path, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(3, activation='softmax')(x)  # 3 classes: Bacterial Leaf Blight, Brown Spot, Leaf Smut
    model = Model(inputs=base_model.input, outputs=x)

    # Unfreeze last 30 layers for fine-tuning
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000005), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def train_model():
    train_dir = "data/processed/train"
    val_dir = "data/processed/validation"
    
    # Enhanced data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        zoom_range=0.3,
        shear_range=0.2
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=16, class_mode='categorical')
    val_generator = val_datagen.flow_from_directory(
        val_dir, target_size=(224, 224), batch_size=16, class_mode='categorical')
    
    # Compute class weights to handle imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    # Load the existing model
    model = create_model()
    model.load_weights('models/finetuned_model_v2.h5')
    
    # Callbacks for checkpointing and learning rate scheduling
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'models/finetuned_model_v2.h5', save_best_only=True, monitor='val_accuracy')
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, restore_best_weights=True)
    
    # Fine-tune for additional 10 epochs
    model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        class_weight=class_weight_dict,
        callbacks=[checkpoint, lr_scheduler, early_stopping]
    )

if __name__ == "__main__":
    train_model()