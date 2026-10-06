import numpy as np
import tensorflow as tf
import cv2

class GradCAM:
    """Implements Gradient-weighted Class Activation Mapping."""
    
    def compute(self, model, image, layer_name):
        """
        Compute GradCAM heatmap by reconstructing the model graph.
        This version handles nested models (like MobileNetV2 inside Sequential).
        """
        img_tensor = tf.cast(image, tf.float32)

        # 1. Flatten the model structure to find the specific layer
        # Since we use Transfer Learning, 'model' is Sequential [MobileNetV2, GAP, Dense...]
        
        # We need a model that maps from the input to both the target conv layer and the final output
        # To do this correctly with nested models, we can't easily use the Sequential 'loop'
        # if the layer is inside the base_model.
        
        base_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) or hasattr(layer, 'layers'):
                base_model = layer
                break
        
        if base_model is None:
            # Fallback for simple models
            grad_model = tf.keras.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
        else:
            # For Transfer Learning models:
            # target_layer is inside base_model
            # output is from the top model
            target_layer = base_model.get_layer(layer_name)
            
            # Create a model that maps from base_model input to target_layer output
            # and then we wrap the rest of the top model logic.
            # Simpler approach: Create a temporary model that stitches them
            
            # Define intermediate model for the base part
            intermediate_base = tf.keras.Model(base_model.inputs, [target_layer.output, base_model.output])
            
            # Now we need to pass base_model.output through the rest of the Sequential layers
            # This is tricky in Keras 3. Let's try the simpler "functional reconstruction"
            
            inputs = tf.keras.Input(shape=image.shape[1:])
            target_out, base_out = intermediate_base(inputs)
            
            x = base_out
            found_base = False
            for layer in model.layers:
                if layer == base_model:
                    found_base = True
                    continue
                if found_base:
                    x = layer(x)
            
            grad_model = tf.keras.Model(inputs, [target_out, x])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            loss = predictions[:, 0]

        # Extract gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        # Multiply each channel in the feature map by the corresponding gradient importance
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
            
        heatmap = heatmap.numpy()
        
        # Scale to [0, 255] and convert to uint8
        heatmap = np.uint8(255 * heatmap)
        return heatmap

if __name__ == "__main__":
    print("GradCAM utility ready.")