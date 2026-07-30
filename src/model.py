# src/model.py
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_model(input_shape=(224, 224, 3), num_classes=3):
    """
    Build and return the ResNet50-based transfer learning model.
    Matches the architecture used in create_resnet50_base_model /
    create_resnet50_finetuned_model during training.

    NOTE: This rebuilds the architecture only. If your saved .keras file
    is a FULL model (via model.save(...), not just weights), you do NOT
    need this function for inference — use tf.keras.models.load_model()
    directly instead, which loads architecture + weights together and
    is more robust to mismatches.
    """
    input_tensor = Input(shape=input_shape)

    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
    base_model.trainable = False  # matches base training stage; fine-tuning unfreezes conv5 later

    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)  # multi-class

    model = Model(inputs=input_tensor, outputs=output_layer)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
